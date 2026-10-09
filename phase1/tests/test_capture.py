#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实时流采集模块自检 —— 在没有摄像头的机器上也能验证大部分逻辑。

为什么需要它：
    开发机（以及很多服务器）没有摄像头，但实时流的代码路径必须验证。
    本脚本用**视频文件**当输入源走一遍 capture_windows ——
    文件与摄像头/RTSP 在 OpenCV 里是同一套 VideoCapture 代码，
    所以能覆盖切窗、取样、缩放、落盘、进度回调、中止标志等全部逻辑，
    唯一没覆盖的只有"设备打开"这一步。

不测什么：
    · 真实摄像头的打开（需要设备）
    · RTSP 握手与认证（需要真实摄像头地址）
    这两项要在有设备的机器上用网页的「探测源」按钮验证。

用法：
    python tests/test_capture.py
"""
from __future__ import annotations

import shutil
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
sys.path.insert(0, str(PROJECT))

from webui.capture import (  # noqa: E402
    capture_windows,
    mask_credentials,
    parse_source,
    probe_source,
)

FAILS: list[str] = []

# ⚠️ 临时目录必须放在工作区内。
#    第一版用了 tempfile.mkdtemp()（系统临时目录），在受限沙箱下
#    cv2.imwrite 直接 "permission denied" —— 而且是 OpenCV 的 C++ 层
#    只打 WARN 不抛异常，测试里看起来像"帧文件异常"，排查方向会被带偏。
TMPROOT = PROJECT / "runs" / "_captest"


def check(name: str, fn) -> None:
    try:
        detail = fn()
        print(f"  ✅ {name}" + (f" — {detail}" if detail else ""))
    except Exception as e:
        print(f"  ❌ {name} — {type(e).__name__}: {e}")
        FAILS.append(name)


def main() -> int:
    print("=" * 70)
    print("实时流采集模块自检")
    print("=" * 70)

    # ── 1. 源解析 ──
    def t_parse():
        cases = {
            "0": ("local", 0),
            "1": ("local", 1),
            "rtsp://admin:pass@10.0.0.5:554/stream1": ("rtsp", "rtsp://admin:pass@10.0.0.5:554/stream1"),
            "D:/videos/a.mp4": ("file", "D:/videos/a.mp4"),
        }
        for spec, want in cases.items():
            got = parse_source(spec)
            assert got == want, f"{spec!r} → {got}，期望 {want}"
        # 空白应当报错，而不是静默当作索引 0
        try:
            parse_source("   ")
            raise AssertionError("空字符串应当报错")
        except ValueError:
            pass
        return f"{len(cases)} 种输入解析正确，空值被拒"

    check("源解析（索引/RTSP/文件/空值）", t_parse)

    # ── 2. 凭据打码 ──
    def t_mask():
        masked = mask_credentials("rtsp://admin:Secret123@10.0.0.5:554/s1")
        assert "Secret123" not in masked, f"密码没被打码：{masked}"
        assert "admin" in masked and "10.0.0.5" in masked
        # 无密码的 URL 不应被破坏
        plain = "rtsp://10.0.0.5:554/s1"
        assert mask_credentials(plain) == plain
        return masked

    check("RTSP 凭据打码（避免日志泄露密码）", t_mask)

    # ── 3. 失败路径：不存在的摄像头索引 ──
    def t_bad_index():
        info = probe_source("97")
        assert not info.ok, "不存在的索引竟然探测成功"
        assert info.error, "失败时应当给出可读的原因"
        return f"正确报错：{info.error[:40]}…"

    check("探测失败路径（不存在的设备索引）", t_bad_index)

    # ── 4. 失败路径：连不上的 RTSP（必须在超时内返回，不能永久阻塞）──
    def t_bad_rtsp():
        t0 = time.time()
        info = probe_source("rtsp://127.0.0.1:59999/nonexistent", timeout_sec=3)
        dt = time.time() - t0
        assert not info.ok, "连不上的 RTSP 竟然成功了"
        assert dt < 60, f"阻塞太久（{dt:.0f}s），超时设置没生效"
        return f"按时返回（{dt:.1f}s）并报错"

    check("探测失败路径（不可达 RTSP，验证不阻塞）", t_bad_rtsp)

    # ── 5. 真实采集：用视频文件当输入源 ──
    def t_capture():
        vids = sorted((PROJECT / "data" / "videos").glob("*.mp4"))
        if not vids:
            raise RuntimeError("data/videos 下没有 mp4，先跑 tools/make_visible_testdata.py")
        src = vids[0]
        tmp = TMPROOT / "cap_test"
        try:
            seen = []
            windows = capture_windows(
                str(src), tmp,
                window_stride_sec=2.0, frames_per_window=3,
                max_windows=3, max_side=320,
                on_progress=lambda d, t: seen.append((d, t)),
            )
            assert windows, "一个窗口都没采集到"
            # 每个窗口应有 2-3 帧
            sizes = [len(w) for w in windows]
            assert all(2 <= s <= 3 for s in sizes), f"窗口帧数异常：{sizes}"
            # 帧文件应真的存在且非空
            for w in windows:
                for _t, p in w:
                    f = Path(p)
                    assert f.exists() and f.stat().st_size > 200, f"帧文件异常：{p}"
            # 缩放应生效（长边 ≤ 320）
            from PIL import Image

            im = Image.open(windows[0][0][1])
            assert max(im.size) <= 320, f"缩放没生效：{im.size}"
            # 进度回调应被调用
            assert seen, "进度回调没被调用"
            # 时间戳应递增
            ts = [t for w in windows for t, _ in w]
            assert ts == sorted(ts), f"时间戳非递增：{ts}"
            return (f"{len(windows)} 个窗口，帧数 {sizes}，"
                    f"缩放后 {im.size}，进度回调 {len(seen)} 次")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    check("真实采集（用视频文件走同一套 VideoCapture 路径）", t_capture)

    # ── 6. 中止标志应能及时生效 ──
    def t_stop():
        vids = sorted((PROJECT / "data" / "videos").glob("*.mp4"))
        if not vids:
            raise RuntimeError("没有测试视频")
        tmp = TMPROOT / "cap_stop"
        state = {"n": 0}

        def should_stop():
            state["n"] += 1
            return state["n"] > 3   # 前几次不叫停，之后叫停

        try:
            t0 = time.time()
            windows = capture_windows(
                str(vids[0]), tmp,
                window_stride_sec=2.0, frames_per_window=3,
                max_windows=50, max_side=320, should_stop=should_stop,
            )
            dt = time.time() - t0
            assert len(windows) < 50, "中止标志没生效，采满了 50 个窗口"
            assert dt < 60, f"中止后仍跑了 {dt:.0f}s"
            return f"中止生效：只采了 {len(windows)} 个窗口（{dt:.1f}s）"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    check("中止标志（用户点「中止」时应尽快停下）", t_stop)

    print()
    if FAILS:
        print(f"❌ {len(FAILS)} 项失败：{FAILS}")
        return 1
    print("✅ 全部通过 —— 采集逻辑可用。")
    print()
    print("⚠️ 仍未验证（需要真实设备，请在网页「实时流」页用「探测源」按钮测）：")
    print("   · 本机摄像头的打开（开发机没有摄像头则跳过）")
    print("   · RTSP 握手与认证")
    return 0


if __name__ == "__main__":
    sys.exit(main())
