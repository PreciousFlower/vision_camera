#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通过网页 API 走完整流程 —— 验证"用户点几下就能用"这件事真的成立。

与 tools/e2e_verify.py 的区别：
    e2e_verify  直接调 Python 模块，验证"链路能不能通"
    本脚本      只调 HTTP 接口，验证"网页这套接口是否自洽"
                （上传 → 启动服务 → 检测 → SSE 实时结果 → 告警）

这是交付前最后一个必须过的关：模块级都对，接口层照样可能接错。

用法：
    python -m webui.server            # 另开一个窗口先起服务
    python tools/web_e2e.py           # 跑本脚本
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
    ap.add_argument("--video", default="")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--skip-start", action="store_true",
                    help="服务已在运行，跳过启动（省去排队等待）")
    ap.add_argument("--stop-after", action="store_true", help="测完停止服务")
    args = ap.parse_args()

    B = args.base.rstrip("/")

    # ── 1. 后端在线 ──
    log("=" * 66)
    log("步骤 1  网页后端")
    log("=" * 66)
    try:
        st = requests.get(f"{B}/api/state", timeout=15).json()
    except Exception as e:
        log(f"❌ 后端没起来：{e}")
        log("   先在另一个窗口跑：python -m webui.server")
        return 1
    log(f"✅ 后端在线   当前服务状态 = {st['service']['state']}")
    log(f"   启用行为 = {[b['id'] for b in st['behaviors']['enabled']]}")

    # ── 2. 集群连通 ──
    log()
    log("=" * 66)
    log("步骤 2  集群连通性（网页的「测试连通性」按钮）")
    log("=" * 66)
    c = requests.get(f"{B}/api/cluster/check", timeout=90).json()
    if not c.get("ok"):
        log(f"❌ 连不上：{c.get('error')}")
        log("   确认 VPN 已连，且 VLM_SSH_PASSWORD 或免密可用")
        return 1
    log(f"✅ 登录节点 {c['host']}")

    # ── 3. 启动服务 ──
    log()
    log("=" * 66)
    log("步骤 3  启动推理服务（网页的「启动服务」按钮）")
    log("=" * 66)
    if args.skip_start:
        log("  跳过（--skip-start）")
    else:
        if st["service"]["state"] in ("ready", "running", "pending", "submitting"):
            log(f"  已有服务在跑（{st['service']['state']}），先停掉")
            requests.post(f"{B}/api/service/stop", timeout=60)
            time.sleep(3)
        r = requests.post(f"{B}/api/service/start", json={"model": args.model}, timeout=60)
        if r.status_code != 200:
            log(f"❌ 启动请求被拒：{r.status_code} {r.text[:300]}")
            return 1
        log(f"  {r.json().get('message')}")

    # 轮询状态，把每一步打出来（等价于用户盯着页面看）
    t0 = time.time()
    ready = False
    while time.time() - t0 < 1800:
        try:
            s = requests.get(f"{B}/api/state", timeout=20).json()["service"]
        except Exception:
            time.sleep(5)
            continue
        if s["state"] == "ready":
            ready = True
            break
        if s["state"] == "error":
            log(f"❌ 服务启动失败：{s['message']}")
            return 1
        el = int(time.time() - t0)
        if el % 30 < 6:
            log(f"  [{el:4d}s] {s['state']}: {s['message']}")
        time.sleep(6)

    if not ready:
        log("❌ 等待就绪超时")
        return 1
    s = requests.get(f"{B}/api/state", timeout=20).json()["service"]
    log(f"✅ 服务就绪   节点 {s['node']}   本机隧道 {s['local_url']}")

    # ── 4. 上传视频 ──
    log()
    log("=" * 66)
    log("步骤 4  上传视频（网页的拖拽上传）")
    log("=" * 66)
    video = Path(args.video) if args.video else None
    if not video:
        cands = sorted((PROJECT / "data" / "videos").glob("*.mp4"))
        if not cands:
            log("❌ data/videos 下没有 mp4")
            return 1
        # 优先选正例视频，这样能验到"命中"路径而不只是"无告警"路径
        video = next((p for p in cands if "pos" in p.name), cands[0])
    log(f"  使用 {video.name}（{video.stat().st_size / 1024:.0f} KB）")
    with open(video, "rb") as f:
        r = requests.post(f"{B}/api/upload", files={"file": (video.name, f)}, timeout=300)
    if r.status_code != 200:
        log(f"❌ 上传失败：{r.status_code} {r.text[:300]}")
        return 1
    up = r.json()
    jid = up["job_id"]
    log(f"✅ 已上传 job_id={jid}  {up['size_mb']} MB")

    # ── 5. 检测 + SSE ──
    log()
    log("=" * 66)
    log("步骤 5  开始检测（网页的「开始检测」按钮 + 实时推送）")
    log("=" * 66)
    r = requests.post(f"{B}/api/detect/{jid}", json={}, timeout=60)
    if r.status_code != 200:
        log(f"❌ 启动检测失败：{r.status_code} {r.text[:300]}")
        return 1
    log("  ✅ 已开始，接收实时推送…")

    n_win = 0
    t1 = time.time()
    summary = None
    try:
        with requests.get(f"{B}/api/job/{jid}/stream", stream=True, timeout=1800) as resp:
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or raw.startswith(":"):
                    continue
                if not raw.startswith("data: "):
                    continue
                ev = json.loads(raw[6:])
                t = ev.get("type")
                if t == "hello":
                    log(f"  任务概况：{ev['summary']['state']}")
                elif t == "window":
                    n_win += 1
                    hits = ev.get("hits") or []
                    tag = "  ".join(f"🔴{h['name']}({h['confidence']:.2f})" for h in hits)
                    log(f"  窗口 {ev['index']}/{ev['total']}  "
                        f"{ev['t_start']}s-{ev['t_end']}s  "
                        f"{ev.get('latency') or 0:.1f}s  {tag or '—'}")
                    if ev.get("scene"):
                        log(f"        场景：{ev['scene']}")
                    if ev.get("error"):
                        log(f"        ⚠️ {ev['error']}")
                elif t in ("done", "final"):
                    summary = ev["summary"]
                    if t == "done":
                        break
                elif t == "error":
                    log(f"  ❌ {ev['error']}")
                    return 1
    except Exception as e:
        log(f"  ⚠️ SSE 中断：{type(e).__name__}: {e}")

    if not summary:
        summary = requests.get(f"{B}/api/job/{jid}", timeout=30).json()
    dt = time.time() - t1

    # ── 6. 结果 ──
    log()
    log("=" * 66)
    log("步骤 6  结果")
    log("=" * 66)
    log(f"  状态      {summary['state']}")
    log(f"  窗口      {summary['windows_done']}/{summary['windows_total']}")
    log(f"  耗时      {dt:.0f}s（平均 {summary.get('avg_latency') or 0:.1f}s/窗口）")
    log(f"  消息      {summary['message']}")

    alerts = summary.get("alerts") or []
    log(f"  告警      {len(alerts)} 条")
    for a in alerts:
        log(f"    🔴 {a['behavior_name']}  {a['t_start']}s-{a['t_end']}s  "
            f"命中 {a['hits']}/{a['span']}  置信度 {a['max_confidence']:.2f}")
        for e in (a.get("evidences") or [])[:2]:
            log(f"       证据：{e}")

    # 逐窗口详情也要能读到（网页的「逐窗口详情」区）
    det = requests.get(f"{B}/api/job/{jid}", timeout=30).json()
    log()
    log(f"  逐窗口记录 {len(det.get('window_results') or [])} 条")
    # 验证帧图片接口（网页缩略图靠它）
    if det.get("window_results"):
        wid = det["window_results"][0]["window_id"]
        fr = requests.get(f"{B}/api/job/{jid}/frames/{wid}/0", timeout=30)
        log(f"  帧图片接口 HTTP {fr.status_code}  "
            f"{len(fr.content)} 字节  {fr.headers.get('content-type')}")

    if args.stop_after:
        log()
        log("  停止服务…")
        requests.post(f"{B}/api/service/stop", timeout=60)
        log("  ✅ 已释放计算节点")

    ok = summary["state"] == "done" and n_win > 0 and bool(det.get("window_results"))
    log()
    log("=" * 66)
    log("🎉 网页全流程验证通过" if ok else "⚠️ 流程走通但有异常，见上")
    log("=" * 66)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
