#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断：预览帧到底有没有在被写。

现象：页面上"实时画面"（/api/job/{id}/preview）3 秒里只拿到 1 张不同的图，
看上去是冻住的。可能的原因有三个，必须分开验证：
    ① 采集线程根本没走到写预览的那段（取样等待循环的路径问题）
    ② 写了但写失败（Windows 上 os.replace 遇到"文件被 HTTP 读开着"会报错，
       而我们把异常吞掉了 —— 这类"静默失败"最难查）
    ③ 写了、也能读，但内容是同一张（源本身静止）

这个脚本直接在进程内跑 capture_windows，绕开 HTTP，
逐条回答上面三个问题。
"""
from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from webui.capture import capture_windows  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else "rtsp://127.0.0.1:8554/live"
OUT = HERE.parent / "runs" / "_preview_diag"


def newest(out_dir: Path) -> Path | None:
    """最新的预览帧（预览是 preview_000012.jpg 这样的一串文件，不是固定名字）。"""
    c = sorted(out_dir.glob("preview_*.jpg"))
    return c[-1] if c else None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for p in OUT.glob("*.jpg"):
        p.unlink()
    pv = OUT / "preview.jpg"

    hits: list[tuple[float, str, int]] = []
    stop = threading.Event()

    def watch() -> None:
        while not stop.is_set():
            p = newest(OUT)
            if p is not None:
                raw = p.read_bytes()
                hits.append((time.time(), hashlib.md5(raw).hexdigest()[:8], len(raw)))
            time.sleep(0.2)

    th = threading.Thread(target=watch, daemon=True)
    th.start()

    print(f"源：{SRC}")
    print("跑一个窗口（stride=6s / 4 帧），同时每 0.2 秒记录一次 preview.jpg 的指纹")
    t0 = time.time()
    windows = capture_windows(
        SRC, OUT, window_stride_sec=6.0, frames_per_window=4, max_windows=1,
        max_side=640, jpeg_quality=82,
    )
    stop.set()
    time.sleep(0.3)
    th.join(timeout=2)

    uniq = len({h for _, h, _ in hits})
    print(f"\n采集结束：{len(windows)} 个窗口，用时 {time.time() - t0:.1f}s")
    print(f"观察点 {len(hits)} 次，不同内容 {uniq} 种")
    if hits:
        print(f"首次写于 t={hits[0][0] - t0:.1f}s，末次 t={hits[-1][0] - t0:.1f}s，"
              f"大小样本 {[n for _, _, n in hits[:3]]}")
    print(f"目录内容：{sorted(p.name for p in OUT.iterdir())}")
    print(f"预览帧每秒约 {uniq / max(0.1, time.time() - t0):.1f} 张")

    # 直接测 os.replace 在"文件被读开着"时会不会失败（Windows 特有陷阱）
    a, b = OUT / "t_a.jpg", OUT / "t_b.jpg"
    b.write_bytes(BENCH)
    a.write_bytes(BENCH)
    fh = open(a, "rb")
    try:
        os.replace(b, a)
        print("os.replace：目标文件被读开着时 **成功**")
    except Exception as e:
        print(f"os.replace：目标文件被读开着时 **失败** → {type(e).__name__}: {e}")
    finally:
        fh.close()

    print("\n判读：")
    if uniq >= 5:
        print("  ✅ 预览确实在写、内容在变 —— 问题在 HTTP 层或页面轮询")
    elif len(hits) <= 1:
        print("  ❌ 预览文件几乎没被写过 —— 采集循环没走到写预览的分支（①）")
    else:
        print("  ❌ 写了但内容不变（②/③），看上面的 os.replace 结论")
    return 0


BENCH = b"\xff\xd8" + b"0" * 512 + b"\xff\xd9"

if __name__ == "__main__":
    raise SystemExit(main())
