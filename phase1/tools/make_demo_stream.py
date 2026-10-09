#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把任意视频转成"模拟摄像头"用的 H.264 裸流（Annex-B），供自建 RTSP 服务器播放。

为什么要这个东西（用户实测反馈："模拟的 RTSP 明显卡顿"）：
    原来演示用的 `runs/_ref.h264` 只有 **60 帧 / 3 秒**，
    播完立刻跳回开头 —— 3 秒循环里也不可能出现一个完整动作，
    所以"AI 没识别出丢垃圾"这件事，一部分原因就在于**素材里根本没有那个动作**。
    演示流必须满足两条：① 足够长（≥ 一个完整动作 + 若干秒上下文）；
    ② 关键时刻有 IDR（关键帧），否则客户端从中途接入时解不出画面。

产出两条：
    runs/demo_cam.h264      —— 真实工地画面（默认取 data/real/smoking_construction.mp4）
    runs/demo_litter.h264   —— 含完整"丢垃圾"动作（默认取 data/videos/littering_pos_vis.mp4）
    runs/demo_queue.h264    —— 含完整"插队"动作（可选，默认关）

用法：
    python tools/make_demo_stream.py                      # 生成上面两条
    python tools/make_demo_stream.py --seconds 60
    python tools/make_demo_stream.py --src 你的视频.mp4 --out runs/demo_cam.h264
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def build(src: Path, out: Path, *, seconds: float, width: int = 640,
          fps: int = 20, keyframe_sec: float = 1.0) -> bool:
    """转码成 Annex-B H.264。

    几个参数不是随便定的：
    · `-g`（GOP）：每 keyframe_sec 秒一个关键帧。
      如果不强制打关键帧，ffmpeg 可能整个片段只有一个 IDR，
      客户端（尤其是 OpenCV/FFmpeg 的 RTSP 拉流）**从中间接入时永远出不来画面**，
      表现就是"能连上但一直没图像"，很难查。
    · `-bf 0`（不用 B 帧）：B 帧依赖前后帧，丢包时花屏更明显，
      且部分解码器对 baseline 之外的 profile 支持不好。
    · `-profile:v baseline`：兼容性最好，摄像头流基本都落在这个档。
    · 只输出裸流（`-f h264`）：RTSP 服务器直接按 NAL 打包，不需要 MP4 容器。
    """
    if not src.exists():
        print(f"  ❌ 找不到源文件 {src}")
        return False

    # ⚠️ 先量源视频有多长。
    #    踩过的坑：脚本里写死"要 45 秒"，可源视频其实只有 10 秒 ——
    #    `-t 45` 不会报错，只是拿到 10 秒的流，
    #    而按 45 秒去校验帧数就会误报"不合格"（第一版就是这么错的）。
    src_sec = 0.0
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(src)], capture_output=True, text=True)
    try:
        src_sec = float((p.stdout or "0").strip())
    except ValueError:
        src_sec = 0.0
    take = seconds if (src_sec <= 0 or src_sec >= seconds) else src_sec
    if src_sec > 0 and src_sec < seconds:
        print(f"  ℹ️ 源视频只有 {src_sec:.1f} 秒（不足 {seconds:.0f} 秒），按全片转")

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(src),
        "-t", f"{take:.3f}",
        "-vf", f"scale={width}:-2,fps={fps}",
        "-c:v", "libx264", "-preset", "veryfast",
        "-profile:v", "baseline", "-level", "3.1", "-pix_fmt", "yuv420p",
        "-g", str(max(1, int(fps * keyframe_sec))), "-bf", "0",
        "-f", "h264", str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ❌ ffmpeg 失败：{(r.stderr or '').strip()[:300]}")
        return False

    # 校验：帧数、能否解码（只看文件大小不算验证）
    info = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0",
         str(out)], capture_output=True, text=True).stdout.strip()
    size_mb = out.stat().st_size / 1048576
    print(f"  ✅ {out.name}: {size_mb:.2f} MB  分辨率 {info}  时长约 {take:.1f}s")

    # 用 OpenCV（与网页同一条解码路径）实测能不能读出足够多的帧
    try:
        import cv2
        cap = cv2.VideoCapture(str(out))
        n = 0
        uniq = set()
        while True:
            ok, f = cap.read()
            if not ok or f is None:
                break
            n += 1
            if n <= 40:
                uniq.add(int(f[::40, ::40].sum()))
        cap.release()
        want = int(take * fps * 0.9)
        print(f"     实测解码 {n} 帧（≥{want} 合格），前 40 帧里不同画面 {len(uniq)} 种"
              f"{'（说明画面在动，不是静止图）' if len(uniq) > 5 else '（⚠️ 画面几乎不动）'}")
        return n >= want
    except ImportError:
        print("     （未装 opencv，跳过多媒体解码实测）")
        return True


def main() -> int:
    ap = argparse.ArgumentParser(description="生成模拟摄像头用的 H.264 裸流")
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--src", default=None, help="只转一条：源视频路径")
    ap.add_argument("--out", default=None, help="配合 --src 指定输出")
    args = ap.parse_args()

    if args.src:
        out = Path(args.out) if args.out else ROOT / "runs" / "demo_custom.h264"
        print(f"转码 {args.src} → {out}")
        return 0 if build(Path(args.src), out, seconds=args.seconds,
                          width=args.width, fps=args.fps) else 1

    jobs = [
        (ROOT / "data" / "real" / "smoking_construction.mp4",
         ROOT / "runs" / "demo_cam.h264",
         "真实工地画面（吸烟场景）"),
        (ROOT / "data" / "videos" / "littering_pos_vis.mp4",
         ROOT / "runs" / "demo_litter.h264",
         "含完整『丢垃圾』动作"),
    ]
    if not shutil.which("ffmpeg"):
        print("❌ 找不到 ffmpeg，请先安装并加入 PATH")
        return 2

    print("=" * 70)
    print("生成模拟摄像头演示流（Annex-B H.264，供 tools/mini_rtsp_h264.py 播放）")
    print("=" * 70)
    ok_all = True
    for src, out, desc in jobs:
        print(f"\n[{desc}]  {src.name} → {out.name}")
        ok_all &= build(src, out, seconds=args.seconds,
                        width=args.width, fps=args.fps)

    print("\n" + "=" * 70)
    if ok_all:
        print("✅ 全部生成完成。用法：")
        print("   python tools/mini_rtsp_h264.py --h264 runs/demo_litter.h264 --port 8554")
        print("   网页「摄像头监测」源填 rtsp://127.0.0.1:8554/live")
        return 0
    print("❌ 有文件生成失败，见上面的错误")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
