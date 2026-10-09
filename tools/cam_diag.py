"""摄像头诊断脚本 —— 逐个后端、逐个索引尝试打开本机摄像头，并打印详细原因。

用法：python tools/cam_diag.py
"""
from __future__ import annotations

import sys
import time

import cv2

print("python:", sys.executable, sys.version.split()[0])
print("cv2   :", cv2.__version__)

try:
    backends = cv2.videoio_registry.getBackends()
    print("可用后端:", [(b, cv2.videoio_registry.getBackendName(b)) for b in backends])
except Exception as e:  # pragma: no cover
    print("枚举后端失败:", e)

CANDIDATES = [
    ("CAP_MSMF", cv2.CAP_MSMF),
    ("CAP_DSHOW", cv2.CAP_DSHOW),
    ("CAP_ANY", cv2.CAP_ANY),
]

for name, be in CANDIDATES:
    if be not in (cv2.CAP_ANY, *cv2.videoio_registry.getBackends()):
        print(f"\n=== {name}: 当前 OpenCV 构建不支持，跳过 ===")
        continue
    print(f"\n=== 后端 {name} ===")
    for i in range(4):
        t0 = time.time()
        cap = cv2.VideoCapture()
        # 显式传 backend，并用 open() 而不是构造函数，便于拿到失败细节
        try:
            ok_open = cap.open(i, be)
        except Exception as e:
            print(f"  index {i}: open() 抛异常 {type(e).__name__}: {e}")
            cap.release()
            continue
        dt = time.time() - t0
        if not ok_open or not cap.isOpened():
            print(f"  index {i}: 打开失败 ({dt*1000:.0f} ms)")
            cap.release()
            continue
        ok, frame = cap.read()
        shape = None if frame is None else frame.shape
        print(f"  index {i}: 打开成功 ({dt*1000:.0f} ms) read={ok} shape={shape} "
              f"reported={cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
              f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} "
              f"fps={cap.get(cv2.CAP_PROP_FPS):.1f} backend={cap.getBackendName()}")
        cap.release()

print("\n完成。")
