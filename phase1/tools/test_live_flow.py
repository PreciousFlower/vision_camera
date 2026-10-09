#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实时流链路的网页级验证 —— 不依赖真实摄像头也能跑通全流程。

思路：把**视频文件**当作实时源喂进去。文件与摄像头/RTSP 在 OpenCV 里
是同一套 VideoCapture 代码，所以链路（采集 → 抽帧 → 推理 → SSE → 告警）
完全一致，只有"设备打开"这一步不同。

这样即使开发机没有摄像头，也能确认实时流功能是好的；
等有设备时，只需把源换成 `0` 或 `rtsp://...`，代码路径不用改。

用法：
    python -m webui.server          # 另开窗口，并已启动推理服务
    python tools/test_live_flow.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

import requests

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def log(m: str = "") -> None:
    print(m, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8848")
    ap.add_argument("--source", default="",
                    help="视频源；留空则自动用 data/videos 下的一个 mp4 充当实时源")
    ap.add_argument("--stride", type=float, default=2.0, help="窗口步长（秒）")
    ap.add_argument("--max-windows", type=int, default=3)
    args = ap.parse_args()
    B = args.base.rstrip("/")

    # ── 1. 后端与服务 ──
    log("=" * 72)
    log("步骤 1  后端与推理服务")
    log("=" * 72)
    try:
        st = requests.get(f"{B}/api/state", timeout=15).json()
    except Exception as e:
        log(f"❌ 后端没起来：{e}\n   先跑 python -m webui.server")
        return 1
    if st["service"]["state"] != "ready":
        log(f"❌ 推理服务未就绪（{st['service']['state']}）。先在网页上点「启动服务」。")
        return 1
    log(f"✅ 后端在线，推理服务就绪（{st['service']['node']}）")

    # ── 2. 摄像头枚举接口 ──
    log()
    log("=" * 72)
    log("步骤 2  本机摄像头枚举")
    log("=" * 72)
    r = requests.get(f"{B}/api/live/cameras", timeout=120).json()
    if r.get("ok"):
        cams = r.get("cameras") or []
        if cams:
            log(f"✅ 找到 {len(cams)} 个：{[c['index'] for c in cams]}")
        else:
            log("⚠️ 没找到摄像头（开发机常见）。用视频文件当实时源继续验证。")
    else:
        log(f"⚠️ 枚举失败：{r.get('error')}")

    # ── 3. 选源 ──
    spec = args.source
    if not spec:
        vids = sorted((PROJECT / "data" / "videos").glob("*.mp4"))
        if not vids:
            log("❌ data/videos 下没有 mp4，先跑 tools/make_visible_testdata.py")
            return 1
        spec = str(vids[0])
        log()
        log(f"（用 {vids[0].name} 当实时源 —— 同一套 VideoCapture 代码路径）")

    # ── 4. 探测源 ──
    log()
    log("=" * 72)
    log("步骤 3  探测源（网页的「探测源」按钮）")
    log("=" * 72)
    r = requests.post(f"{B}/api/live/probe", json={"source": spec}, timeout=180).json()
    if not r.get("ok"):
        log(f"❌ 探测失败：{r.get('error')}")
        return 1
    log(f"✅ {r['label']}  {r['width']}×{r['height']}  "
        f"实测 {r['measured_fps']} fps  后端 {r['backend']}  类型 {r['kind']}")

    # ── 5. 设置参数并启动 ──
    log()
    log("=" * 72)
    log("步骤 4  设置采集参数并开始实时检测")
    log("=" * 72)
    r = requests.post(f"{B}/api/live/config",
                      json={"window_stride_sec": args.stride,
                            "max_windows": args.max_windows}, timeout=30).json()
    log(f"  参数：{r.get('live')}")

    r = requests.post(f"{B}/api/live/start", json={"source": spec}, timeout=60)
    if r.status_code != 200:
        log(f"❌ 启动失败：{r.status_code} {r.text[:300]}")
        return 1
    jid = r.json()["job_id"]
    log(f"✅ 已启动 job_id={jid}  来源 {r.json()['label']}")

    # ── 6. SSE 实时接收 ──
    log()
    log("=" * 72)
    log("步骤 5  实时接收（SSE，每完成一个窗口立刻推一条）")
    log("=" * 72)
    n_win = 0
    t0 = time.time()
    summary = None
    with requests.get(f"{B}/api/job/{jid}/stream", stream=True, timeout=900) as resp:
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or raw.startswith(":") or not raw.startswith("data: "):
                continue
            ev = json.loads(raw[6:])
            t = ev.get("type")
            if t == "hello":
                log(f"  任务状态：{ev['summary']['state']} "
                    f"（来源类型 {ev['summary'].get('source_kind')}）")
            elif t == "window":
                n_win += 1
                hits = ev.get("hits") or []
                tag = "  ".join(f"🔴{h['name']}({h['confidence']:.2f})" for h in hits)
                log(f"  窗口 {ev['index']}/{ev['total']}  "
                    f"{ev['t_start']}s–{ev['t_end']}s  "
                    f"{ev.get('latency') or 0:.1f}s  {tag or '—'}")
                if ev.get("scene"):
                    log(f"      场景：{ev['scene']}")
                if ev.get("error"):
                    log(f"      ⚠️ {ev['error']}")
            elif t in ("done", "final"):
                summary = ev["summary"]
                if t == "done":
                    break
            elif t == "error":
                log(f"  ❌ {ev['error']}")
                return 1

    dt = time.time() - t0
    if not summary:
        summary = requests.get(f"{B}/api/job/{jid}", timeout=30).json()

    # ── 7. 结果 ──
    log()
    log("=" * 72)
    log("步骤 6  结果")
    log("=" * 72)
    log(f"  状态    {summary['state']}")
    log(f"  窗口    {summary['windows_done']}/{summary['windows_total']}")
    log(f"  耗时    {dt:.0f}s")
    log(f"  消息    {summary['message']}")
    log(f"  告警    {len(summary.get('alerts') or [])} 条")
    for a in (summary.get("alerts") or []):
        log(f"    🔴 {a['behavior_name']} {a['t_start']}s-{a['t_end']}s "
            f"命中 {a['hits']}/{a['span']}")

    det = requests.get(f"{B}/api/job/{jid}", timeout=30).json()
    wins = det.get("window_results") or []
    log(f"  逐窗口记录 {len(wins)} 条")
    if wins:
        wid = wins[0]["window_id"]
        fr = requests.get(f"{B}/api/job/{jid}/frames/{wid}/0", timeout=30)
        log(f"  帧图片接口 HTTP {fr.status_code}  {len(fr.content)} 字节")

    ok = summary["state"] == "done" and n_win > 0 and len(wins) == n_win
    log()
    log("=" * 72)
    if ok:
        log("🎉 实时流链路验证通过")
        log()
        log("  ⇒ 网页「实时流」页是可用的。换成真设备只需把源改成")
        log("     0                      （笔记本/ USB 摄像头）")
        log("     rtsp://user:pass@ip:554/...   （网络摄像头）")
        log("     代码路径完全相同，不需要改任何东西。")
    else:
        log("⚠️ 有异常，见上")
    log("=" * 72)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
