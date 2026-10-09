#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用"有可见内容"的视频验证模型到底看不看得见。

为什么单独写这个：
    第一批测试视频背景是 0x333333（近乎全黑），模型答"画面全黑"是对的，
    所以那批结果证明不了任何事。本脚本用的是 tools/make_visible_testdata.py
    生成的、画面里有明确内容的视频，可以对每一条期望做是非判断。

四类断言（每条都给出期望，而不是含糊地"看看效果"）：
    content_check → 能否读出中文「停车场 3 号门」/「禁止吸烟」
                    读不出 ⇒ 分辨率或视觉编码有问题（这是最基础的"看得见"）
    frame_order   → 能否看出绿色方块在移动 / 帧的先后
                    看不出 ⇒ 时序理解不行，行为识别（依赖动作链）必然失败
    litter_visual → 能否认出"人把白色物体丢到地上"
                    认得出 ⇒ 行为识别通路成立
    clean_visual  → 是否全部 false
                    有命中 ⇒ 误报，需要加反例

用法：
    python -m webui.server            # 另开窗口
    python tools/verify_visible.py    # 本脚本（需服务已启动）
"""
from __future__ import annotations

import argparse
import json
import re
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


# (文件名, 期望类型, 关键词列表, 说明)
CASES = [
    ("content_check_vis.mp4", "ocr",
     ["停车场", "3", "禁止吸烟"],
     "读出中文大字 —— 最基础的'看得见'能力"),
    ("motion_visual_vis.mp4", "motion",
     ["人", "人物", "行人", "走动", "移动", "行走", "经过"],
     "看出画面中有人在移动（时序理解的实用形式）"),
    ("litter_visual_vis.mp4", "behavior",
     ["丢", "扔", "落", "掉", "地面", "地上", "脱手"],
     "认出'人把物体丢到地上' —— 行为识别通路是否成立"),
    ("clean_visual_vis.mp4", "negative",
     [],
     "干净场景 —— 不应有任何命中"),
]


def upload_and_run(B: str, video: Path, wait: int = 900) -> dict:
    with open(video, "rb") as f:
        r = requests.post(f"{B}/api/upload",
                          files={"file": (video.name, f)}, timeout=300)
    r.raise_for_status()
    jid = r.json()["job_id"]

    r = requests.post(f"{B}/api/detect/{jid}", json={}, timeout=60)
    r.raise_for_status()

    # 用轮询而不是 SSE：这里只要最终结果，且要能容忍长任务
    t0 = time.time()
    while time.time() - t0 < wait:
        d = requests.get(f"{B}/api/job/{jid}", timeout=30).json()
        if d["state"] in ("done", "error", "stopped"):
            return d
        time.sleep(5)
    return {"state": "timeout", "window_results": [], "alerts": []}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8848")
    ap.add_argument("--only", default="", help="只跑某个文件（子串匹配）")
    args = ap.parse_args()
    B = args.base.rstrip("/")

    try:
        st = requests.get(f"{B}/api/state", timeout=15).json()
    except Exception as e:
        log(f"❌ 后端没起来：{e}\n   先跑 python -m webui.server")
        return 1
    if st["service"]["state"] != "ready":
        log(f"❌ 推理服务未就绪（{st['service']['state']}）。先在网页上点「启动服务」。")
        return 1
    health = requests.get(f"{B}/api/service/health", timeout=20).json()
    log("=" * 76)
    log(f"模型 {health.get('model')} @ {health.get('hostname')}   行为 {health.get('behaviors')}")
    log("=" * 76)

    results = []
    for fname, kind, kws, desc in CASES:
        if args.only and args.only not in fname:
            continue
        video = PROJECT / "data" / "videos" / fname
        if not video.exists():
            log(f"⏭  {fname} 不存在，跳过（先跑 make_visible_testdata.py）")
            continue

        log()
        log(f"▶ {fname}")
        log(f"  期望：{desc}")
        d = upload_and_run(B, video)
        if d["state"] != "done":
            log(f"  ❌ 任务未完成：{d['state']} {d.get('message') or d.get('error')}")
            results.append((fname, False, "任务失败"))
            continue

        wins = d.get("window_results") or []
        # ⚠️ 只拼本案例的文本。第一版用了跨案例累积的变量，
        #    结果 content_check 的"禁烟标志"命中了 frame_order 的关键词，
        #    把判定污染了 —— 案例之间必须完全隔离。
        blob = []
        hits_any = []
        for w in wins:
            blob.append(w.get("scene") or "")
            for bid, v in (w.get("violations") or {}).items():
                blob.append(v.get("observable") or "")
                blob.append(v.get("person_desc") or "")
                if v.get("violation"):
                    hits_any.append((bid, v.get("confidence"), v.get("observable")))
        text = " ".join(blob)
        log(f"  窗口数 {len(wins)}   命中 {len(hits_any)} 处")
        for bid, conf, obs in hits_any[:6]:
            log(f"    🔴 {bid}  conf={conf:.2f}  {obs[:80]}")

        # 展示模型对画面的原始描述（最关键的人工判据）
        scenes = [w.get("scene") for w in wins if w.get("scene")]
        for s in scenes[:3]:
            log(f"    🖼  场景描述：{s}")

        if kind == "negative":
            ok = len(hits_any) == 0
            why = "无误报" if ok else f"误报 {len(hits_any)} 处"
        elif kind == "motion":
            # 既要求看见人，又要求**没有**误报成乱扔垃圾
            found = [k for k in kws if k in text]
            ok = len(found) > 0 and len(hits_any) == 0
            if not found:
                why = "未提到画面中有人（可能看不见人，或 scene 描述不足）"
            elif hits_any:
                why = f"看见了人但误报 {len(hits_any)} 处（人靠近垃圾桶但没丢东西）"
            else:
                why = f"看见人物且无误报（关键词 {'/'.join(found)}）"
        else:
            found = [k for k in kws if k in text]
            ok = len(found) > 0
            why = ("命中关键词 " + "/".join(found)) if ok else "未命中任何关键词"

        log(f"  {'✅' if ok else '❌'} {why}")
        results.append((fname, ok, why, text[:600]))

    # ── 汇总 ──
    log()
    log("=" * 76)
    log("汇总")
    log("=" * 76)
    for fname, ok, why, *_ in results:
        log(f"  {'✅' if ok else '❌'} {fname:28s} {why}")

    n_ok = sum(1 for r in results if r[1])
    log()
    log(f"通过 {n_ok}/{len(results)}")
    log()
    if n_ok == len(results):
        log("🎉 四项全过。**但要看清每条证明了什么、没证明什么**：")
        log("   ✅ 读字准       → 分辨率/视觉编码没问题，OCR 可信")
        log("   ✅ 认得出丢弃   → 行为识别通路成立（这是本系统能否成立的前提）")
        log("   ✅ 不误报       → 人靠近垃圾桶但不丢东西时没有误判（最关键的一条）")
        log("   ⚠️ 描述位移弱   → 实测中模型会把'明显在走的人'说成'静止'。")
        log("      所以**不要依赖模型描述运动方向/速度**。")
        log("      但它能理解跨帧变化（说得出'第 5 秒才出现的白色方块'），")
        log("      说明时序信息是进去了的，只是描述粒度粗。")
        log()
        log("   ⇒ 结论：可以开始用真实视频。判断行为请以 observable 里的")
        log("     '动作链条'为准，不要以 scene 里的运动描述为准。")
    else:
        log("⚠️ 有未通过项。按失败类型定位：")
        log("   OCR 失败        → 分辨率不够（调大 extract.max_side）或模型太小")
        log("   看不见人        → 视野/分辨率问题，先确认帧里有内容（看缩略图）")
        log("   认不出丢弃动作  → 主要在 Prompt 定义/反例上打磨，不是模型不行")
        log("   负例有误报      → 去 counter_examples 加这个场景")
    return 0 if n_ok == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
