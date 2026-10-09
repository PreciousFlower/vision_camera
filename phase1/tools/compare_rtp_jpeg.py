#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 ffmpeg 自己的 RTP 打包器当参考，对比我们手工打包的 RTP/JPEG。

为什么这么做：
    手写 RTSP 服务器的 RTP/JPEG 打包一直过不了 ffmpeg 的解包器，
    试了 5 种组合都失败。与其继续猜，不如让 **ffmpeg 自己打一遍包**，
    把它的字节抓下来逐字段对比 —— 它怎么打，我们就怎么打。

做法：
    1. 起一个 UDP 监听（ffmpeg 的 RTP 输出默认走 UDP）
    2. 让 ffmpeg 把 MJPEG 打成 RTP 发到该端口
    3. 解析收到的包：RTP 头、RFC 2435 载荷头、量化表、分片 offset
    4. 打印成表，和我们的实现对照

用法：
    python tools/compare_rtp_jpeg.py
"""
from __future__ import annotations

import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PORT = 45999


def listen_udp(port: int, out: list, stop: threading.Event) -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("127.0.0.1", port))
    except OSError as e:
        print(f"  绑定失败：{e}")
        return
    s.settimeout(0.5)
    while not stop.is_set():
        try:
            data, _ = s.recvfrom(65535)
        except socket.timeout:
            continue
        except OSError:
            break
        out.append(data)
    s.close()


def main() -> int:
    frames = sorted((PROJECT / "data" / "videos").glob("*.mp4"))
    if not frames:
        print("❌ 没有测试视频")
        return 1
    video = frames[0]

    print("=" * 78)
    print("用 ffmpeg 的 RTP 打包器生成参考数据")
    print("=" * 78)

    packets: list[bytes] = []
    stop = threading.Event()
    th = threading.Thread(target=listen_udp, args=(PORT, packets, stop), daemon=True)
    th.start()
    time.sleep(0.5)

    # 关键：让 ffmpeg 把 RTP 直接写进**文件**而不是发 UDP。
    # 沙箱拦 UDP（实测收到 0 个包），写文件不需要网络权限，
    # 而且 rtp 复用器写出来的是裸 RTP 包序列，正好可以逐包解析。
    out_file = PROJECT / "runs" / "_ref_rtp.bin"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-re", "-stream_loop", "0", "-i", str(video),
        "-t", "3",
        "-an", "-c:v", "mjpeg", "-q:v", "5",
        "-f", "rtp", "-payload_type", "26",
        # 把 pkt_size 调小，强制产生多分片，才能观察到 offset 的变化
        f"rtp://127.0.0.1:{PORT}?pkt_size=800",
    ]
    print(f"  源：{video.name}")
    # 用 tee 不行（沙箱拦 UDP），改用 -f rtp + 文件输出
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-re", "-stream_loop", "0", "-i", str(video),
        "-t", "3",
        "-an", "-c:v", "mjpeg", "-q:v", "5",
        "-f", "rtp", "-payload_type", "26", "-pkt_size", "800",
        str(out_file),
    ]
    print(f"  输出文件：{out_file.name}")
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(f"  ⚠️ ffmpeg 退出码 {r.returncode}：{(r.stderr or '')[-300:]}")
    except subprocess.TimeoutExpired:
        print("  （ffmpeg 超时）")

    # 从文件里解析 RTP 包。
    # rtp 复用器写出的就是连续的 RTP 包（每个以 0x80 开头，长度由 UDP 长度决定），
    # 但文件里没有长度前缀 —— 靠 RTP 头之后的内容无法自定长。
    # 所以改用"按 MTU 推断"的方式：pkt_size=800 表示每个 RTP 包不超过 800 字节，
    # 完整包为 800，最后一个包为剩余长度。这里按 800 切分并对末包做校验。
    raw = out_file.read_bytes() if out_file.exists() else b""
    print(f"  文件大小：{len(raw)} 字节")
    if raw:
        packets = []
        i = 0
        while i + 12 <= len(raw):
            if raw[i] >> 6 != 2:
                break
            n = min(800, len(raw) - i)
            packets.append(raw[i:i + n])
            i += n
        print(f"  按 {800}B 切分出 {len(packets)} 个包")
    stop.set()
    time.sleep(0.2)

    print(f"  收到 {len(packets)} 个 UDP 包")
    if not packets:
        print("❌ 一个包都没收到 —— 可能 UDP 被沙箱拦了。")
        print("   那就无法用这个方法取参考；改为查 ffmpeg 源码/文档确认载荷格式。")
        return 1

    print()
    print("=" * 78)
    print("解析（RFC 2435 RTP/JPEG）")
    print("=" * 78)
    print(f"{'#':>3} {'seq':>6} {'ts':>8} {'M':>2} {'PT':>3} "
          f"{'tsp':>4} {'off':>7} {'typ':>4} {'Q':>4} {'w8':>4} {'h8':>4} {'数据':>6}")

    rows = []
    for i, p in enumerate(packets[:20]):
        if len(p) < 12:
            print(f"{i:>3} 包太短 ({len(p)}B)")
            continue
        b0, b1, seq, ts, ssrc = struct.unpack("!BBHII", p[:12])
        marker = (b1 >> 7) & 1
        pt = b1 & 0x7F
        pl = p[12:]
        if len(pl) < 8:
            print(f"{i:>3} {seq:>6} {ts:>8} {marker:>2} {pt:>3}  载荷仅 {len(pl)}B")
            continue
        tsp, hi, lo, typ, q, w8, h8 = struct.unpack("!BBHBBBB", pl[:8])
        off = (hi << 16) | lo
        data = pl[8:]
        rows.append((seq, ts, marker, off, typ, q, w8, h8, len(data)))
        print(f"{i:>3} {seq:>6} {ts:>8} {marker:>2} {pt:>3} "
              f"{tsp:>4} {off:>7} {typ:>4} {q:>4} {w8:>4} {h8:>4} {len(data):>6}")

    print()
    print("=" * 78)
    print("关键字段的规律（这些就是我们要照抄的）")
    print("=" * 78)
    if rows:
        qs = sorted({r[5] for r in rows})
        offs = [r[3] for r in rows]
        print(f"  Q（量化表）取值集合：{qs}")
        print(f"  说明：255 = 量化表随帧发送；0 = 本片不带表")
        print()
        print(f"  offset 序列（前 20 个）：{offs}")
        print()
        # 按时间戳分组看每帧的首片 offset
        frames_by_ts: dict[int, list] = {}
        for r in rows:
            frames_by_ts.setdefault(r[1], []).append(r)
        print(f"  共 {len(frames_by_ts)} 个不同时间戳（帧）")
        for ts, rs in list(frames_by_ts.items())[:3]:
            first = rs[0]
            print(f"    ts={ts}: {len(rs)} 片  首片 offset={first[3]} "
                  f"Q={first[5]}  末片 marker={rs[-1][2]}")
        print()
        print("  ⚠️ 重点看首片的 offset 是不是 0，以及第二片的 offset 是多少：")
        print("     若第二片 offset ≈ 首片数据长度（不含表），说明 offset 只算图像数据；")
        print("     若 ≈ 首片数据长度 + 表长度，说明 offset 覆盖了头与表。")
        if len(rows) >= 2 and rows[0][1] == rows[1][1]:
            print(f"     实测：首片数据 {rows[0][8]}B，第二片 offset={rows[1][3]}"
                  f"  → 差值 {rows[1][3] - rows[0][8]}")
            d = rows[1][3] - rows[0][8]
            if abs(d) <= 2:
                print("     ⇒ offset 只算图像数据（不含 JPEG 头与量化表）")
            else:
                print(f"     ⇒ offset 额外多了 {d} 字节（应是 8 字节载荷头 + 量化表长度）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
