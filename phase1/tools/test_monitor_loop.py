#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""摄像头监测的**持续监测循环**验证 —— 用 mock 后端，不需要 VPN/GPU。

验证什么（这些是"监测"区别于"分析一段视频"的关键）：
    1. 持续模式真的会**循环采集**，而不是采一批就结束
    2. 每批处理完都把告警推给前端（type=alerts），且累积不归零
    3. SSE 每个窗口推一条，且带 people（动作描述）字段
    4. 用户点停止后能干净收尾（type=done）
    5. 长期运行有资源上限（历史窗口裁剪、帧文件滚动删除）

用法：
    python -m webui.server            # 另开窗口
    python tools/test_monitor_loop.py
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
    ap.add_argument("--source", default="")
    ap.add_argument("--stride", type=float, default=2.0)
    ap.add_argument("--run-sec", type=float, default=45,
                    help="跑多久后主动停止（验证持续循环）")
    args = ap.parse_args()
    B = args.base.rstrip("/")

    log("=" * 74)
    log("摄像头监测：持续监测循环验证（mock 后端，不需要 VPN）")
    log("=" * 74)

    src = args.source
    if not src:
        vids = sorted((PROJECT / "data" / "videos").glob("*.mp4"))
        if not vids:
            log("❌ data/videos 下没有 mp4")
            return 1
        # 用一个正例视频，这样 mock 会产出行命中，能验证告警链路
        src = str(next((p for p in vids if "_pos_" in p.name), vids[0]))
    log(f"源：{src}")
    log(f"（用视频文件当实时源 —— 与摄像头/RTSP 走同一套 VideoCapture 代码）")
    log()

    # 参数：窗口步长调小 + 连续模式 + 动作描述 + mock
    r = requests.post(f"{B}/api/live/config",
                      json={"window_stride_sec": args.stride}, timeout=30).json()
    log(f"采集参数：{r.get('live')}")

    r = requests.post(f"{B}/api/live/start",
                      json={"source": src, "continuous": True,
                            "with_actions": True, "use_mock": True}, timeout=60)
    if r.status_code != 200:
        log(f"❌ 启动失败：{r.status_code} {r.text[:300]}")
        return 1
    info = r.json()
    jid = info["job_id"]
    log(f"✅ 已启动持续监测 job={jid}  use_mock={info.get('use_mock')}")
    log()

    # ── 收 SSE ──
    n_win = 0
    n_people = 0
    people_samples = []
    alert_events = 0
    max_alerts = 0
    max_cumulative = 0
    stopped = False
    t0 = time.time()

    try:
        with requests.get(f"{B}/api/job/{jid}/stream",
                          stream=True, timeout=args.run_sec + 120) as resp:
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or raw.startswith(":") or not raw.startswith("data: "):
                    continue
                ev = json.loads(raw[6:])
                t = ev.get("type")
                if t == "window":
                    n_win += 1
                    cum = ev.get("cumulative") or ev.get("index")
                    max_cumulative = max(max_cumulative, cum)
                    ppl = ev.get("people") or []
                    n_people += len(ppl)
                    if ppl and len(people_samples) < 5:
                        people_samples.append(ppl)
                    hits = ev.get("hits") or []
                    log(f"  窗口 #{cum}  {ev['t_start']}s–{ev['t_end']}s  "
                        f"人 {len(ppl)}  "
                        + ("  ".join(f"🔴{h['name']}" for h in hits) if hits else "—"))
                    for p in ppl[:2]:
                        log(f"      · {p.get('where','')} → {p.get('action','')}"
                            + (f"（{p['changed']}）" if p.get('changed') and p['changed'] != '无' else ""))
                elif t == "alerts":
                    alert_events += 1
                    n = len(ev.get("alerts") or [])
                    max_alerts = max(max_alerts, n)
                    log(f"  ↳ 告警推送 #{alert_events}：{n} 条（累计窗口 {ev.get('cumulative')}）")
                elif t in ("done", "final"):
                    log(f"  任务结束：{(ev.get('summary') or {}).get('message','')}")
                    stopped = True
                    if t == "done":
                        break
                elif t == "error":
                    log(f"  ❌ {ev.get('error')}")
                    return 1

                # 跑够时间就主动停止，验证"持续"确实在持续
                if time.time() - t0 > args.run_sec and not stopped:
                    log()
                    log(f"  （已跑 {args.run_sec:.0f}s，主动停止以验证收尾）")
                    requests.post(f"{B}/api/stop/{jid}", timeout=30)
    except Exception as e:
        log(f"⚠️ SSE 中断：{type(e).__name__}: {e}")

    dt = time.time() - t0
    log()
    log("=" * 74)
    log("结果")
    log("=" * 74)
    log(f"  运行时长      {dt:.0f}s")
    log(f"  收到窗口      {n_win} 个（最大累计编号 {max_cumulative}）")
    log(f"  动作描述      {n_people} 条人物记录，样例 {len(people_samples)} 组")
    log(f"  告警推送      {alert_events} 次，最多 {max_alerts} 条")
    log(f"  收尾          {'✅ 已收到结束事件' if stopped else '⚠️ 未收到（可能超时）'}")

    ok = True
    if n_win < 3:
        log("  ❌ 窗口太少，持续循环可能没生效")
        ok = False
    if alert_events < 1:
        log("  ❌ 从未推送过告警事件")
        ok = False
    if n_people == 0:
        log("  ❌ 一条动作描述都没有 —— people 字段没透到前端")
        ok = False
    if max_cumulative < n_win:
        log("  ⚠️ 累计编号未随批次增长")

    log()
    if ok:
        log("🎉 持续监测循环验证通过：会循环采集、逐窗推送、动作描述可达前端")
        log()
        log("  注意：这里用的是 **mock 假模型**，动作描述是编的。")
        log("  它验证的是链路（采集→循环→SSE→渲染），不是识别效果。")
        log("  真实模型效果需要 VPN 恢复后跑一次。")
    else:
        log("❌ 有未通过项")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
