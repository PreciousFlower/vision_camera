"""按设备名（而不是索引）打开摄像头，绕过索引枚举问题。

用法：python tools/cam_diag_name.py
"""
from __future__ import annotations

import sys
import time

import cv2

NAME = "USB2.0 HD UVC WebCam"

specs = [
    ("FFMPEG + video=名字", f"video={NAME}", cv2.CAP_FFMPEG),
    ("MSMF   + 名字",       NAME,                      cv2.CAP_MSMF),
    ("DSHOW  + 名字",       NAME,                      cv2.CAP_DSHOW),
    ("FFMPEG 索引 0",       "0",                       None),  # 走 OpenCV 自己的解析
]

for label, src, be in specs:
    t0 = time.time()
    try:
        if be is None:
            cap = cv2.VideoCapture(int(src))
        else:
            cap = cv2.VideoCapture(src, be)
        opened = cap.isOpened()
        dt = (time.time() - t0) * 1000
        if not opened:
            print(f"{label:22s}: 打开失败 ({dt:.0f} ms)")
            cap.release()
            continue
        ok, frame = cap.read()
        print(f"{label:22s}: 成功 ({dt:.0f} ms) read={ok} "
              f"shape={None if frame is None else frame.shape} "
              f"backend={cap.getBackendName()}")
        cap.release()
    except Exception as e:
        print(f"{label:22s}: 异常 {type(e).__name__}: {e}")

print("\n完成。")
