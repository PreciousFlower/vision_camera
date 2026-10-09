#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比 open_source 与直接 VideoCapture 在 H.264 RTSP 上的行为差异。

背景：
    探测工具（probe_source → open_source）连续 3 次成功读到 640×360；
    但监测流程走同一个 open_source 却一个窗口都采不到（帧目录为空）。
    两者都用 open_source，差异只可能在**调用参数**或**环境状态**上。

这个脚本把可能的差异逐个测一遍：
    A. open_source 默认参数
    B. open_source + CAP_PROP_BUFFERSIZE=1（capture_windows 会设）
    C. cv2.VideoCapture 直接开（不设任何环境变量）
    D. 连续开两次（第二次是否受第一次残留影响）

用法：
    python tools/mini_rtsp_h264.py --h264 runs/_ref.h264 --port 8554   # 另开窗口
    python tools/diag_rtsp_read.py
"""
from __future__ import annotations

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
sys.path.insert(0, str(PROJECT))

URL = "rtsp://127.0.0.1:8554/live"


def read_some(cap, n=8, budget=8.0):
    """在时间预算内尽量读 n 帧，返回实际读到的帧数。"""
    got = 0
    t0 = time.time()
    while got < n and time.time() - t0 < budget:
        ok, f = cap.read()
        if ok and f is not None:
            got += 1
    return got, time.time() - t0


def main() -> int:
    import cv2

    print("=" * 74)
    print(f"目标 {URL}")
    print("=" * 74)

    # A. open_source 默认参数
    print("\n[A] open_source 默认参数")
    from webui.capture import open_source

    t0 = time.time()
    cap, kind, src, err = open_source(URL, timeout_sec=10, transport="tcp")
    if cap is None:
        print(f"    打开失败（{time.time() - t0:.1f}s）：{err}")
    else:
        print(f"    打开用时 {time.time() - t0:.1f}s  kind={kind}")
        n, dt = read_some(cap)
        print(f"    读到 {n} 帧，用时 {dt:.1f}s")
        cap.release()

    # B. open_source + BUFFERSIZE（capture_windows 会设这个）
    print("\n[B] open_source + CAP_PROP_BUFFERSIZE=1（监测流程会设）")
    cap, kind, src, err = open_source(URL, timeout_sec=10, transport="tcp")
    if cap is None:
        print(f"    打开失败：{err}")
    else:
        try:
            cap.set(getattr(cv2, "CAP_PROP_BUFFERSIZE", 38), 1)
        except Exception as e:
            print(f"    设 BUFFERSIZE 异常：{e}")
        n, dt = read_some(cap)
        print(f"    读到 {n} 帧，用时 {dt:.1f}s")
        cap.release()

    # C. 直接 VideoCapture
    print("\n[C] 直接 cv2.VideoCapture（不额外设参数）")
    t0 = time.time()
    cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"    打不开（{time.time() - t0:.1f}s）")
    else:
        n, dt = read_some(cap)
        print(f"    读到 {n} 帧，用时 {dt:.1f}s")
        cap.release()

    # D. 在子线程里读（监测流程在后台线程执行）
    print("\n[D] 在子线程里 open_source + 读帧（模拟监测的执行方式）")
    box = {}

    def worker():
        try:
            c, _k, _s, e = open_source(URL, timeout_sec=10, transport="tcp")
            if c is None:
                box["r"] = (0, 0.0, e or "打不开")
                return
            try:
                c.set(getattr(cv2, "CAP_PROP_BUFFERSIZE", 38), 1)
            except Exception:
                pass
            n, dt = read_some(c)
            c.release()
            box["r"] = (n, dt, "")
        except Exception as ex:
            box["r"] = (0, 0.0, f"{type(ex).__name__}: {ex}")

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    th.join(25)
    if th.is_alive():
        print("    ⏱ 子线程 25 秒未返回")
    else:
        n, dt, e = box.get("r", (0, 0.0, "无结果"))
        print(f"    读到 {n} 帧，用时 {dt:.1f}s  {e}")

    print()
    print("=" * 74)
    print("判读")
    print("=" * 74)
    print("  A 通、B 不通 → CAP_PROP_BUFFERSIZE 是元凶（监测流程就设了它）")
    print("  A/B/C 通、D 不通 → 子线程里读流有问题")
    print("  全通 → 监测采不到帧的原因在别处（时序/停止逻辑），不在读取本身")
    return 0


if __name__ == "__main__":
    sys.exit(main())
