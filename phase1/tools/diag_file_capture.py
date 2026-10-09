#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断：为什么"用视频文件冒充实时源做持续监测"采不到窗口？

背景（实测踩到）：
    tools/diag_http_responsive.py 里起了持续监测（源 = runs/demo_cam.h264），
    服务端日志一直在刷：
        [31fe8c13] 实时流已重新连接（第 4 次），用时 0.0s
        [31fe8c13] 采集完成：0 个窗口
        [31fe8c13] 实时流断开（没有采集到任何窗口…），将自动重连
    最后以 ClusterError 结束。也就是说**这条"没有摄像头时的主要验证手段"
    是坏的** —— 而 tools/test_frame_cache.py 正建立在它上面。

为什么不能靠猜：
    这条路径上有好几个分支（文件源按帧号跳 vs 实时源按墙钟取样、
    StreamReader 是否存在、loop_file 取模回绕、clean_dir 清理），
    任何一处不对都可能表现成"0 个窗口"。所以本脚本**逐层**验证：
        1. 直接调用 capture_windows（不带 reader、不带 loop_file）能拿到窗口吗？
        2. 带上 loop_file=True（持续监测用的组合）呢？
        3. 视频文件到底能被 OpenCV 打开吗（帧数/帧率读到了什么）？
        4. 文件读完后 cap.read() 是什么行为（EOF 还是继续出帧）？
    把"哪一层先坏"钉死，再决定怎么修。
"""
from __future__ import annotations

import shutil
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TMP = ROOT / "runs" / "_tmp_filecap_test"


def main() -> int:
    import argparse

    import cv2

    from webui.capture import capture_windows, open_source, parse_source

    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(ROOT / "runs" / "demo_cam.h264"),
                    help="要用作'伪装实时源'的视频文件")
    ap.add_argument("--stride", type=float, default=2.0)
    args = ap.parse_args()

    demo = Path(args.file)
    print(f"素材：{demo}  存在={demo.exists()}  "
          f"大小={demo.stat().st_size if demo.exists() else 0} 字节")
    if not demo.exists():
        print("❌ 素材不存在，无法诊断（先跑 tools/make_demo_stream.py）")
        return 1

    if TMP.exists():
        shutil.rmtree(TMP, ignore_errors=True)

    print("\n[1] parse_source / open_source")
    kind, src = parse_source(str(demo))
    print(f"    parse_source → kind={kind!r} source={src!r}")
    cap, k2, source2, err = open_source(str(demo))
    if cap is None:
        print(f"    ❌ open_source 失败：{err}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"    open_source OK：kind={k2!r} fps={fps} 帧数={total}")

    print("\n[2] 文件读完后 cap.read() 是什么行为（EOF 语义）")
    # 直接把位置设到末尾之后，看 read 返回什么
    ok, fr = cap.read()
    print(f"    第 1 帧：ok={ok} shape={None if fr is None else fr.shape}")
    if total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total))
        ok2, fr2 = cap.read()
        print(f"    跳到末尾后 read：ok={ok2} frame={'None' if fr2 is None else fr2.shape}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok3, fr3 = cap.read()
        print(f"    再跳回 0 后 read：ok={ok3} frame={'None' if fr3 is None else fr3.shape}")
    cap.release()

    print("\n[3] capture_windows 基线（不带 reader / 不带 loop_file）")
    try:
        out = TMP / "base"
        wins = capture_windows(str(demo), out, window_stride_sec=2.0,
                               frames_per_window=4, max_windows=3,
                               max_side=640)
        print(f"    → {len(wins)} 个窗口；每窗帧数 "
              f"{[len(w) for w in wins]}")
        if wins:
            print(f"    首窗时间戳：{[round(t,2) for t,_ in wins[0]]}")
    except Exception as e:                                  # noqa: BLE001
        print(f"    ❌ {type(e).__name__}: {e}")
        traceback.print_exc()

    print("\n[4] 持续监测的组合（loop_file=True，模拟第二批起）")
    try:
        out = TMP / "loop"
        wins = capture_windows(str(demo), out, window_stride_sec=2.0,
                               frames_per_window=4, max_windows=2,
                               max_side=640, loop_file=True,
                               t_offset=0.0, idx_offset=0,
                               clean_dir=True)
        print(f"    → {len(wins)} 个窗口；每窗帧数 {[len(w) for w in wins]}")
        wins2 = capture_windows(str(demo), out, window_stride_sec=2.0,
                                frames_per_window=4, max_windows=2,
                                max_side=640, loop_file=True,
                                t_offset=8.0, idx_offset=8,
                                clean_dir=False)
        print(f"    第二批 → {len(wins2)} 个窗口；每窗帧数 "
              f"{[len(w) for w in wins2]}")
    except Exception as e:                                  # noqa: BLE001
        print(f"    ❌ {type(e).__name__}: {e}")
        traceback.print_exc()

    print("\n[5] 复用外部 cap（持续监测真的这么用：cap=stream.cap）")
    try:
        cap2, kind2, _, err2 = open_source(str(demo))
        out = TMP / "reuse"
        wins = capture_windows(str(demo), out, window_stride_sec=2.0,
                               frames_per_window=4, max_windows=2,
                               max_side=640, cap=cap2, kind="file",
                               loop_file=True, clean_dir=True)
        print(f"    → {len(wins)} 个窗口；每窗帧数 {[len(w) for w in wins]}")
        cap2.release()
    except Exception as e:                                  # noqa: BLE001
        print(f"    ❌ {type(e).__name__}: {e}")
        traceback.print_exc()

    print("\n[6] 关键差异：LiveStream 传的 kind 是什么？")
    # webui 的 LiveStream 会用自己的 kind 覆盖调用方的判断 —— 这对
    # "视频文件冒充实时源"影响极大：kind 一旦不是 'file'，
    # 就会走**实时源的墙钟取样**分支，而文件能瞬间读完 → 采到 0 个窗口。
    from webui.capture import probe_source
    info = probe_source(str(demo), timeout_sec=10)
    print(f"    probe_source → kind={info.kind!r} ok={info.ok} "
          f"fps={info.measured_fps} {info.width}x{info.height}")
    if info.kind != "file":
        print(f"    ⚠️ 注意：probe_source 把 .h264 当成了 {info.kind!r} ——"
              f"如果 LiveStream 也这么判，就会走错分支")

    shutil.rmtree(TMP, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
