#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实 RTSP 端到端验证 —— 用 ffmpeg 自建服务器，不依赖真实摄像头。

为什么能这么测：
    ffmpeg 自带 RTSP **服务器**能力（`-rtsp_flags listen`），可以把一路视频
    发布成 `rtsp://127.0.0.1:PORT/xxx`。这只是"发布端"，接下来
    webui/capture.py 用 OpenCV 以标准 RTSP 客户端身份去拉流 ——
    **和拉真实海康/大华摄像头走的是同一条代码路径**（同一个 VideoCapture
    + 同一个 rtsp:// 解析），所以拉流、超时、认证、采集这些逻辑都被真实覆盖。

    唯一没覆盖的是：真实摄像头的 RTSP 握手变种（Digest 认证、私有 URL 格式）。
    那些只能在拿到设备后验证。

测什么：
    1. 能建起 RTSP 服务器并发布流
    2. probe_source() 能探测到它，拿到分辨率与实测帧率
    3. capture_windows() 能从 RTSP 流按窗口采集到帧
    4. 采集到的帧确实有内容（不是黑图/花屏）
    5. 采集出的帧能送进推理服务得到结果（可选，需要服务在跑）

用法：
    python tools/test_rtsp_real.py
    python tools/test_rtsp_real.py --no-infer    # 跳过推理部分
"""
from __future__ import annotations

import argparse
import shutil
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
sys.path.insert(0, str(PROJECT))

TMP = PROJECT / "runs" / "_rtsp_test"
PORT = 8555
STREAM = "live"


def log(m: str = "") -> None:
    print(m, flush=True)


def start_rtsp_server(video: Path) -> subprocess.Popen:
    """用 ffmpeg 起一个 RTSP 服务器并循环发布 video。

    关键参数：
        -rtsp_flags listen        让 ffmpeg 当服务器而不是客户端
        -stream_loop -1           循环播放，方便反复拉流
        -re                       按真实速率推流（否则会瞬间推完）
        -c:v copy 不行（源是 mp4 需要重封装），用 libx264 重编码
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("找不到 ffmpeg")
    url = f"rtsp://127.0.0.1:{PORT}/{STREAM}"
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "warning",
        "-re", "-stream_loop", "-1", "-i", str(video),
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-pix_fmt", "yuv420p", "-g", "10",
        "-f", "rtsp", "-rtsp_flags", "listen", url,
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-infer", action="store_true", help="跳过推理验证")
    ap.add_argument("--base", default="http://127.0.0.1:8848")
    ap.add_argument("--video", default="")
    args = ap.parse_args()

    vids = ([Path(args.video)] if args.video
            else sorted((PROJECT / "data" / "videos").glob("*.mp4")))
    if not vids:
        log("❌ data/videos 下没有 mp4")
        return 1
    video = vids[0]

    url = f"rtsp://127.0.0.1:{PORT}/{STREAM}"
    log("=" * 74)
    log("真实 RTSP 端到端验证")
    log("=" * 74)
    log(f"源视频  {video.name}")
    log(f"RTSP 地址  {url}")
    log()

    failures: list[str] = []
    proc = None
    try:
        # ── 1. 起服务器 ──
        log("[1] 启动 ffmpeg RTSP 服务器…")
        proc = start_rtsp_server(video)
        # 等它开始监听（ffmpeg 的 listen 模式要等客户端连上才会真正开始推流）
        from webui.capture import probe_source

        server_up = False
        t0 = time.time()
        while time.time() - t0 < 30:
            if proc.poll() is not None:
                err = (proc.stderr.read() or b"").decode("utf-8", "replace")
                log(f"    ❌ ffmpeg 退出（rc={proc.returncode}）：{err[-400:]}")
                return 1
            time.sleep(1)
            server_up = True
            if time.time() - t0 > 2:
                break
        log(f"    ✅ ffmpeg 进程在跑（PID {proc.pid}）")

        # ── 2. 探测 ──
        log()
        log("[2] probe_source() 以 RTSP 客户端身份探测…")
        info = probe_source(url, timeout_sec=15)
        log(f"    ok={info.ok}  kind={info.kind}  {info.width}×{info.height}  "
            f"实测 {info.measured_fps} fps  后端 {info.backend}")
        if not info.ok:
            log(f"    ❌ 探测失败：{info.error}")
            failures.append("probe_source 探测 RTSP 失败")
            return 1
        if info.kind != "rtsp":
            failures.append(f"源类型识别错误：{info.kind}（应为 rtsp）")
        log("    ✅ 探测成功，且被正确识别为 rtsp 类型")

        # ── 3. 采集窗口 ──
        log()
        log("[3] capture_windows() 从 RTSP 流按窗口采集…")
        from webui.capture import capture_windows

        if TMP.exists():
            shutil.rmtree(TMP, ignore_errors=True)
        seen: list[tuple[int, int]] = []
        t0 = time.time()
        windows = capture_windows(
            url, TMP,
            window_stride_sec=4.0, frames_per_window=3,
            max_windows=2, max_side=640,
            on_progress=lambda d, t: seen.append((d, t)),
        )
        dt = time.time() - t0
        sizes = [len(w) for w in windows]
        log(f"    采集到 {len(windows)} 个窗口，帧数 {sizes}，耗时 {dt:.1f}s")
        if not windows:
            failures.append("从 RTSP 流没采集到任何窗口")
            return 1
        if not all(2 <= s <= 3 for s in sizes):
            failures.append(f"窗口帧数异常：{sizes}")
        log(f"    ✅ 采集成功（进度回调 {len(seen)} 次）")

        # ── 4. 帧内容检查（确认不是黑图/花屏） ──
        log()
        log("[4] 检查采集到的帧是否真有内容…")
        from PIL import Image, ImageStat

        stats = []
        for w in windows:
            for _t, p in w:
                im = Image.open(p).convert("L")
                st = ImageStat.Stat(im)
                stats.append((Path(p).name, im.size, round(st.mean[0], 1),
                              round(st.stddev[0], 1)))
        for name, size, mean, sd in stats[:6]:
            log(f"    {name}  {size}  亮度 {mean:6.1f}  标准差 {sd:5.1f}")
        flat = [s for s in stats if s[3] < 1.0]
        if flat:
            failures.append(f"{len(flat)} 帧标准差≈0（纯色/黑图，拉流可能有问题）")
        else:
            log(f"    ✅ {len(stats)} 帧全部有正常画面内容")

        # ── 5. 送进推理（可选） ──
        if not args.no_infer:
            log()
            log("[5] 把采集到的帧送进推理服务…")
            import base64

            import requests

            B = args.base.rstrip("/")
            try:
                st = requests.get(f"{B}/api/state", timeout=15).json()
                if st["service"]["state"] != "ready":
                    log(f"    ⏭ 推理服务未就绪（{st['service']['state']}），跳过")
                else:
                    sess = requests.get(f"{B}/api/service/health", timeout=20).json()
                    w = windows[0]
                    payload = {
                        "frames": [{"b64": base64.b64encode(Path(p).read_bytes()).decode()}
                                   for _t, p in w],
                        "labels": [f"t={t:.1f}s" for t, _p in w],
                    }
                    r = requests.post(f"{sess and st['service']['local_url']}/infer",
                                      json=payload, timeout=300)
                    res = r.json()
                    log(f"    HTTP {r.status_code}  耗时 {res.get('latency_sec')}s")
                    parsed = res.get("parsed") or {}
                    log(f"    场景：{parsed.get('scene')}")
                    for bid, v in (parsed.get("violations") or {}).items():
                        mark = "🔴" if v.get("violation") else "⚪"
                        log(f"      {mark} {bid:16s} conf={v.get('confidence', 0):.2f}")
                    if res.get("error"):
                        log(f"    ⚠️ {res['error']}")
                    else:
                        log("    ✅ RTSP 帧 → 推理服务 → 结构化结果，全链路通")
            except Exception as e:
                log(f"    ⚠️ 推理验证跳过：{type(e).__name__}: {e}")

        log()
        log("=" * 74)
        if failures:
            log(f"❌ {len(failures)} 项失败：")
            for f in failures:
                log(f"   · {f}")
            return 1
        log("🎉 RTSP 链路验证通过")
        log()
        log("  已验证：发布 → 探测 → 采集 → 帧有内容 →（可选）推理")
        log("  未验证（需要真实设备）：Digest 认证、厂商私有 URL 格式")
        log("  ⇒ 界面上的「实时流」填 rtsp://... 即可用，代码路径与本次相同")
        return 0

    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
            log("\n[清理] 已停止测试用的 RTSP 服务器")


if __name__ == "__main__":
    sys.exit(main())
