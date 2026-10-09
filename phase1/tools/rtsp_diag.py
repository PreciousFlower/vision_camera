#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RTSP 协议诊断 —— 直接观察 FFmpeg 客户端发了什么、我们的服务答了什么。

为什么需要它：
    自建 RTSP 服务器一直报 "Too short RTP/JPEG packet"，但光看这句日志
    判断不出**是我们发包不对，还是 ffmpeg 根本没走到解包那一步**。
    猜了好几轮都没修好（改 Q 字段、加分片、查 RFC 2435），
    所以改成把协议交互原样打出来对照。

本脚本扮演一个**极简 RTSP 客户端**：
    连上服务器 → 依次发 OPTIONS / DESCRIBE / SETUP / PLAY → 收 RTP 包
    把每一步的请求与响应原文打印出来，并解析收到的交织帧。

用法：
    python tools/mini_rtsp_server.py --port 8554     # 另开窗口
    python tools/rtsp_diag.py                        # 本脚本
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


def recv_response(sock) -> str:
    """收一条 RTSP 响应（到 \\r\\n\\r\\n 为止，含可能的 body）。"""
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
    text = data.decode("utf-8", "replace")
    # 若声明了 Content-Length，继续收 body
    clen = 0
    for line in text.split("\r\n"):
        if line.lower().startswith("content-length:"):
            try:
                clen = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    head, _, body = text.partition("\r\n\r\n")
    while len(body.encode()) < clen:
        chunk = sock.recv(4096)
        if not chunk:
            break
        body += chunk.decode("utf-8", "replace")
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8554)
    ap.add_argument("--path", default="live")
    args = ap.parse_args()

    import socket

    url = f"rtsp://{args.host}:{args.port}/{args.path}"
    print("=" * 74)
    print(f"RTSP 协议诊断 → {url}")
    print("=" * 74)

    s = socket.create_connection((args.host, args.port), timeout=15)
    s.settimeout(10)
    cseq = 0

    def send(method: str, extra: str = "", with_session: bool = False) -> str:
        nonlocal cseq
        cseq += 1
        lines = [f"{method} {url} RTSP/1.0", f"CSeq: {cseq}",
                 "User-Agent: rtsp-diag/1.0"]
        if extra:
            lines.append(extra)
        req = "\r\n".join(lines) + "\r\n\r\n"
        print(f"\n──→ 请求 ──\n{req.strip()}")
        s.sendall(req.encode())
        resp = recv_response(s)
        print(f"←── 响应 ──\n{resp.strip()}")
        return resp

    try:
        send("OPTIONS")
        sdp = send("DESCRIBE", "Accept: application/sdp")
        # 关键：用 TCP 交织。RTP 走同一条连接，不需要额外端口。
        send("SETUP", "Transport: RTP/AVP/TCP;unicast;interleaved=0-1")
        send("PLAY", "Range: npt=0.000-")

        print()
        print("=" * 74)
        print("开始收交织帧（最多 6 个 RTP 包）")
        print("=" * 74)
        s.settimeout(12)
        buf = b""
        got = 0
        rows = []
        limit = 24
        while got < limit:
            try:
                chunk = s.recv(65536)
            except socket.timeout:
                print("  ⏱ 收包超时")
                break
            if not chunk:
                print("  连接被关闭")
                break
            buf += chunk
            while len(buf) >= 4 and buf[0:1] == b"$":
                chan = buf[1]
                ln = struct.unpack("!H", buf[2:4])[0]
                if len(buf) < 4 + ln:
                    break
                pkt = buf[4:4 + ln]
                buf = buf[4 + ln:]
                got += 1
                if len(pkt) < 12:
                    print(f"  #{got} ch={chan} len={ln}  ⚠️ 小于 RTP 头")
                    continue
                b0, b1, seq, ts, ssrc = struct.unpack("!BBHII", pkt[:12])
                ver = b0 >> 6
                marker = (b1 >> 7) & 1
                pt = b1 & 0x7F
                payload = pkt[12:]
                if len(payload) >= 8:
                    ts_field, frag_hi, frag_lo, jtype, q, w8, h8 = struct.unpack(
                        "!BBHBBBB", payload[:8])
                    off = (frag_hi << 16) | frag_lo
                    rows.append((seq, ts, marker, off, q, len(payload) - 8, ts_field))
        print()
        print(f"{'seq':>6} {'timestamp':>10} {'M':>2} {'offset':>7} {'Q':>4} {'数据':>6}")
        prev = None
        gaps = 0
        for seq, ts, marker, off, q, n, tsf in rows:
            note = ""
            if prev is not None:
                d = (seq - prev) & 0xFFFF
                if d != 1:
                    note = f"  ← 序列号跳变 +{d}"
                    gaps += 1
            prev = seq
            print(f"{seq:>6} {ts:>10} {marker:>2} {off:>7} {q:>4} {n:>6}{note}")
        print()
        print(f"共 {len(rows)} 个包，序列号跳变 {gaps} 处")
        if gaps:
            print("❌ 序列号不连续 —— 客户端会判定丢包并丢帧")
        else:
            print("✅ 序列号严格连续（+1）")

        # 按时间戳分组，检查每帧的偏移是否从 0 开始、末尾是否置 marker
        print()
        frames = {}
        for seq, ts, marker, off, q, n, tsf in rows:
            frames.setdefault(ts, []).append((seq, marker, off, n, q))
        print(f"共 {len(frames)} 个不同时间戳（= 帧数）")
        for ts, pkts in list(frames.items())[:4]:
            offs = [p[2] for p in pkts]
            mk = [p[1] for p in pkts]
            qs = sorted({p[4] for p in pkts})
            ok_off = offs[0] == 0
            ok_mk = mk[-1] == 1 and all(m == 0 for m in mk[:-1])
            print(f"  ts={ts}: {len(pkts)} 片  首片offset={offs[0]} "
                  f"{'✅' if ok_off else '❌'}  末尾marker={mk[-1]} "
                  f"{'✅' if ok_mk else '❌'}  Q值={qs}")
        if len(frames) <= 1:
            print("  ⚠️ 只看到 1 帧 —— 若每帧重新开始，客户端会误判。"
                  "检查时间戳是否随帧递增。")
    finally:
        try:
            s.close()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
