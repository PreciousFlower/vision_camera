#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同卡多 worker 的轮询网关（stdlib 实现，跑在计算节点上）。

为什么需要它（用户反馈"AI 给出结论的时间也很久"）：
    实测单窗口推理要 10~12 秒，而这张 A800 有 80GB 显存，
    4B 模型 bf16 只占约 8~9GB —— **完全放得下 2~3 份**。
    于是同一张卡上跑多个 worker 进程、并行吃窗口，
    吞吐就能接近翻倍，而**计费不变**（还是一张卡）。

为什么做成"网关"而不是在本机轮流连多个端口：
    本机到计算节点是两级 SSH 隧道（direct-tcpip），每多一个端口就要多一条隧道，
    连接管理、重连、健康检查全都要跟着改。
    在计算节点内部放一个只做 TCP 转发的网关，本机仍然只连一个端口 ——
    隧道、健康检查、作业脚本的对外契约完全不变。

实现要点：
    · 轮询（round-robin）分配新连接；HTTP 是短连接（每个请求一条），
      所以"按连接轮询"就等于"按请求轮询"
    · 双向转发用两个线程；任一侧关闭就收掉另一侧
    · 某个后端连不上就自动跳到下一个（不能让一个坏 worker 拖垮整体）
    · 健康检查：把 /health 转给第一个后端即可（各 worker 模型相同）

用法（在计算节点上）：
    python vlm_gateway.py --port 18080 --backends 18081,18082
    python vlm_gateway.py --port 18080 --backends 18081,18082,18083 --verbose
"""
from __future__ import annotations

import argparse
import socket
import sys
import threading
import time

HOST = "0.0.0.0"
BUF = 65536
BACKEND_TIMEOUT = 20.0
# 健康状态缓存时长（秒）：太短会每请求都探测拖慢转发，太长则故障恢复慢
HEALTH_TTL = 5.0


class Gateway:
    def __init__(self, port: int, backends: list[tuple[str, int]],
                 verbose: bool = False) -> None:
        self.port = port
        self.backends = backends
        self.verbose = verbose
        self.rr = 0
        self.lock = threading.Lock()
        self.stats = {i: 0 for i in range(len(backends))}
        self.fails = {i: 0 for i in range(len(backends))}
        # 每个后端的健康状态缓存 {i: (探测时刻, 是否就绪)}
        self.health: dict[int, tuple[float, bool]] = {}

    # ── 轮询挑一个后端：不健康或连不上就顺延到下一个 ──
    def pick(self):
        with self.lock:
            start = self.rr
            self.rr = (self.rr + 1) % len(self.backends)
        n = len(self.backends)
        for off in range(n):
            i = (start + off) % n
            # ★ 只挑**健康**的后端：worker 的 HTTP 服务先起来、模型在后台加载，
            #   所以"端口能连"≠"能推理"。不做这个判断的话，网关会把请求送给
            #   还在加载的 worker，客户端收到 ConnectionError/503 ——
            #   实测表现是"开头一批请求全部失败，然后突然全好了"。
            if not self._healthy(i):
                continue
            host, port = self.backends[i]
            try:
                s = socket.create_connection((host, port), timeout=BACKEND_TIMEOUT)
            except OSError as e:
                with self.lock:
                    self.fails[i] += 1
                    self.health[i] = (time.time(), False)
                if self.verbose:
                    print(f"[gw] 后端 {host}:{port} 连不上（{e}），换下一个",
                          flush=True)
                continue
            with self.lock:
                self.stats[i] += 1
            return s, i
        return None, -1

    def _healthy(self, i: int) -> bool:
        """后端模型是否就绪（带 HEALTH_TTL 秒缓存，避免每请求都探测）。"""
        now = time.time()
        with self.lock:
            ts, ok = self.health.get(i, (0.0, False))
        if now - ts < HEALTH_TTL:
            return ok
        ok = self._probe(i)
        with self.lock:
            self.health[i] = (now, ok)
        return ok

    def _probe(self, i: int, timeout: float = 4.0) -> bool:
        host, port = self.backends[i]
        try:
            with socket.create_connection((host, port), timeout=timeout) as s:
                s.settimeout(timeout)
                s.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                          b"Connection: close\r\n\r\n")
                buf = b""
                while len(buf) < 16384:
                    d = s.recv(4096)
                    if not d:
                        break
                    buf += d
        except OSError:
            return False
        low = buf.lower()
        # /health 返回 {"ready": true, ...}；只认 ready=true
        return b'"ready"' in low and b"true" in low

    def _pipe(self, src: socket.socket, dst: socket.socket) -> None:
        try:
            while True:
                data = src.recv(BUF)
                if not data:
                    break
                dst.sendall(data)
        except OSError:
            pass
        finally:
            for s in (src, dst):
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    s.close()
                except OSError:
                    pass

    def handle(self, cli: socket.socket, addr) -> None:
        t0 = time.time()
        srv, idx = self.pick()
        if srv is None:
            try:
                cli.sendall(b"HTTP/1.1 503 Service Unavailable\r\n"
                            b"Content-Length: 12\r\n\r\nno backend  ")
            except OSError:
                pass
            cli.close()
            return
        if self.verbose:
            print(f"[gw] 转发到后端 #{idx} {self.backends[idx]}", flush=True)
        threading.Thread(target=self._pipe, args=(cli, srv), daemon=True).start()
        threading.Thread(target=self._pipe, args=(srv, cli), daemon=True).start()
        if self.verbose:
            print(f"[gw] 连接结束，用时 {time.time() - t0:.1f}s", flush=True)

    def serve(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, self.port))
        srv.listen(64)
        print(f"[gw] 网关监听 {HOST}:{self.port} → "
              f"{', '.join(f'{h}:{p}' for h, p in self.backends)}", flush=True)

        def reporter() -> None:
            while True:
                time.sleep(60)
                with self.lock:
                    print(f"[gw] 每分钟统计：{self.stats}（失败次数 {self.fails}）",
                          flush=True)

        threading.Thread(target=reporter, daemon=True).start()
        while True:
            try:
                cli, addr = srv.accept()
            except OSError:
                continue
            threading.Thread(target=self.handle, args=(cli, addr),
                             daemon=True).start()


def main() -> int:
    ap = argparse.ArgumentParser(description="VLM 多 worker 轮询网关")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--backends", required=True,
                    help="worker 端口列表，逗号分隔，如 18081,18082（都在本机）")
    ap.add_argument("--host", default="127.0.0.1", help="后端所在主机")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ports = [int(p.strip()) for p in args.backends.split(",") if p.strip()]
    if not ports:
        print("❌ 没有后端端口")
        return 2
    Gateway(args.port, [(args.host, p) for p in ports],
            verbose=args.verbose).serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
