#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RTSP 拉流穷举测试 —— 同时扫"分片 offset 写法"和"RTP 包大小"。

**已知事实**（前几轮实测确定，不要重复验证）：
    · 单包模式（整帧一个 RTP 包）能被 OpenCV 正常解码 12/12 帧
      → **载荷格式（RTP 头 / RFC 2435 载荷头 / 量化表 / JPEG 数据）完全正确**
    · 一分片就失败。所以变量只剩"分片怎么切、offset 怎么写"。

**本轮新增的怀疑点**：包大小。
    之前所有尝试都固定用 1400 字节的包。如果问题出在
    "包太大被中间层重组"或"某种大小触发了客户端的重组边界"，
    那么换包大小就能看出差异 —— 这是最后一个没被扰动过的变量。

判定标准（必须同时满足，否则是假成功）：
    1. cap.isOpened() 为真
    2. 能连续读到 N 帧
    3. **帧内容确实在变化** —— 用帧间像素差判定，
       排除"反复解出同一帧"的假成功（单包模式实测 11/12 帧不同才算过）。

用法：
    python tools/sweep_rtsp.py            # 扫 offset 写法 × 包大小
    python tools/sweep_rtsp.py --quick    # 只扫几个代表组合
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PORT = 8557
URL = f"rtsp://127.0.0.1:{PORT}/live"

# (说明, 额外命令行参数)
VARIANTS = [
    ("单包（对照，应通）", ["--single-packet"]),
    ("offset=纯图像, pkt=1400", ["--offset-mode", "0", "--pkt-size", "1400"]),
    ("offset=纯图像, pkt=800", ["--offset-mode", "0", "--pkt-size", "800"]),
    ("offset=纯图像, pkt=600", ["--offset-mode", "0", "--pkt-size", "600"]),
    ("offset=纯图像, pkt=400", ["--offset-mode", "0", "--pkt-size", "400"]),
    ("offset=含8头, pkt=800", ["--offset-mode", "1", "--pkt-size", "800"]),
    ("offset=含头+表, pkt=800", ["--offset-mode", "2", "--pkt-size", "800"]),
    ("每片带表, pkt=800", ["--offset-mode", "3", "--pkt-size", "800"]),
]

QUICK = {0, 1, 3}


def kill_port(port: int) -> None:
    try:
        out = subprocess.run(["netstat", "-ano"], capture_output=True,
                             text=True, timeout=20).stdout
    except Exception:
        return
    for line in out.splitlines():
        if "LISTENING" in line and f":{port}" in line:
            subprocess.run(["taskkill", "/PID", line.split()[-1], "/F"],
                           capture_output=True)


def probe(tries: int = 12, timeout_sec: float = 25.0) -> tuple[int, int, str]:
    """拉流并返回 (成功帧数, 内容不同的帧数, 错误摘要)。

    ⚠️ 必须套硬超时：拉不通时 OpenCV 的 FFmpeg 后端会卡在内部 30 秒超时上，
    多个变体逐个试会把整个扫描拖到十分钟以上（实测踩到，脚本被外部掐断）。
    """
    import threading

    box: dict = {}

    def worker():
        import cv2
        import numpy as np

        try:
            cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap.release()
                box["r"] = (0, 0, "打不开")
                return
            frames = []
            for _ in range(tries):
                ok, f = cap.read()
                if ok and f is not None:
                    frames.append(f)
            cap.release()
            uniq = 0
            if frames:
                base = frames[0].astype("int16")
                for f in frames[1:]:
                    if np.abs(f.astype("int16") - base).mean() > 1.0:
                        uniq += 1
            box["r"] = (len(frames), uniq, "")
        except Exception as e:
            box["r"] = (0, 0, f"{type(e).__name__}")

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    th.join(timeout_sec)
    if th.is_alive():
        return 0, 0, f"超时({timeout_sec:.0f}s，无有效帧)"
    return box.get("r", (0, 0, "无结果"))


def main() -> int:
    quick = "--quick" in sys.argv
    variants = [v for i, v in enumerate(VARIANTS) if not quick or i in QUICK]

    print("=" * 78)
    print("RTSP 拉流穷举：分片 offset 写法 × RTP 包大小")
    print("=" * 78)
    print()

    results = []
    for label, extra in variants:
        kill_port(PORT)
        time.sleep(1)
        log = PROJECT / "runs" / "_rtsp_sweep.log"
        with open(log, "w", encoding="utf-8") as fh:
            proc = subprocess.Popen(
                [sys.executable, str(HERE / "mini_rtsp_server.py"),
                 "--video", "data/videos/littering_pos_vis.mp4",
                 "--port", str(PORT), *extra],
                cwd=str(PROJECT), stdout=fh, stderr=subprocess.STDOUT,
            )
        time.sleep(4)
        if proc.poll() is not None:
            print(f"  {label:26s} ❌ 服务启动失败")
            results.append((label, 0, 0, "服务未起来"))
            continue

        n, uniq, err = probe()
        proc.terminate()
        try:
            proc.wait(timeout=6)
        except subprocess.TimeoutExpired:
            proc.kill()

        ok = n >= 8 and uniq >= 2
        print(f"  {label:26s} {'✅ 通' if ok else '❌ 失败'}   "
              f"解码 {n:2d} 帧 / {uniq:2d} 帧内容不同"
              + (f"   {err}" if err else ""))
        results.append((label, n, uniq, err))

    kill_port(PORT)
    print()
    print("=" * 78)
    print("结论")
    print("=" * 78)
    winner = [r for r in results if r[1] >= 8 and r[2] >= 2 and "单包" not in r[0]]
    if winner:
        print(f"  🎉 找到可用的分片写法：{winner[0][0]}")
        print("     → 把这个组合固化进 mini_rtsp_server.py 的默认值")
    else:
        print("  ❌ 所有分片组合都过不了 ffmpeg 的 RTP/JPEG 解包器。")
        print("     单包模式能通，说明载荷格式正确、问题在分片重组。")
        print("     结论：不再继续投入 —— 自建服务器只是测试便利设施，不是交付物。")
    return 0 if winner else 2


if __name__ == "__main__":
    sys.exit(main())
