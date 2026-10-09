#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""等若干个推理 worker 的**模型**真正就绪（不是端口能连上就算）。

为什么必须有这一步（实测踩到的坑）：
    多 worker 场景下，`serve-vlm.sbatch` 以前只等"端口开始监听"，
    然后在前面挂网关就完事了。可是**端口监听 ≠ 模型加载完**：
    vlm_service 是先起 HTTP 服务、再在后台线程里加载模型的
    （这样页面能立刻看到"正在加载"）。
    结果：网关一启动就按轮询把请求分给了还在加载的 worker，
    第一批请求全部以 `ConnectionError: RemoteDisconnected` 失败 ——
    用户看到的是"开头几分钟一个结果都没有，然后突然全好了"。

用法（在计算节点上，由 sbatch 调用）：
    python wait_ready.py --ports 18081,18082,18083 --timeout 900
退出码 0 = 全部就绪；1 = 超时（会把各端口的最后状态打出来，便于排查）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


def state_of(port: int, timeout: float = 5.0) -> tuple[bool, str]:
    """返回 (是否就绪, 状态描述)。"""
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
    except Exception as e:                          # noqa: BLE001
        return False, f"{type(e).__name__}"
    if d.get("ready"):
        return True, "ready"
    return False, str(d.get("state") or d.get("error") or "loading")[:40]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", required=True, help="逗号分隔，如 18081,18082")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    ports = [int(p.strip()) for p in args.ports.split(",") if p.strip()]
    if not ports:
        print("❌ 没有端口")
        return 1

    t0 = time.time()
    last_print = 0.0
    while time.time() - t0 < args.timeout:
        states = {p: state_of(p) for p in ports}
        ready = [p for p, (ok, _) in states.items() if ok]
        if len(ready) == len(ports):
            print(f"[wait] ✅ {len(ready)}/{len(ports)} 个 worker 模型就绪"
                  f"（用时 {time.time() - t0:.0f}s）", flush=True)
            return 0
        now = time.time()
        if not args.quiet and now - last_print >= 20:
            last_print = now
            detail = "、".join(f"{p}:{s}" for p, (_, s) in states.items())
            print(f"[wait] {len(ready)}/{len(ports)} 就绪，"
                  f"已等 {now - t0:.0f}s（{detail}）", flush=True)
        time.sleep(3)

    print(f"[wait] ❌ 超时 {args.timeout}s：仍无全部就绪", flush=True)
    for p in ports:
        ok, s = state_of(p)
        print(f"       {p}: {'就绪' if ok else s}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
