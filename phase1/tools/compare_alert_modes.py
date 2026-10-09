#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比三种告警策略 —— 同一批推理结果、同一份标注，只换判定方式。

为什么需要它：
    "该用哪个策略"不能拍脑袋。三种模式的差别只在**聚合**这一步，
    推理结果完全一样，所以可以拿同一份 windows.jsonl 直接对比，
    零额外推理成本、零额外机时。

    看完这张表再决定用哪个：精确率低说明误报多（该收紧），
    召回率低说明漏报（该放松或换 immediate）。

用法：
    python -m civility.cli all            # 先产生结果
    python tools/compare_alert_modes.py   # 再对比
    python tools/compare_alert_modes.py --input runs/default/windows-xxxx.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

from civility.aggregate import derive_events  # noqa: E402
from civility.config import load_behaviors, load_settings  # noqa: E402
from civility.runner import behavior_fingerprint, load_results  # noqa: E402
from civility.score import load_annotations, score_events, score_windows  # noqa: E402

MODES = ["immediate", "ai", "smoothed"]


def fmt(v) -> str:
    return "—" if v is None else f"{v * 100:5.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="", help="窗口结果 jsonl；留空自动找")
    ap.add_argument("--annotations", default="data/annotations.jsonl")
    args = ap.parse_args()

    cfg = load_settings()
    bs = load_behaviors(cfg["paths"]["behaviors"])

    if args.input:
        in_path = Path(args.input)
    else:
        fp = behavior_fingerprint(bs)
        in_path = Path(cfg["paths"]["results_dir"]) / "default" / f"windows-{fp}.jsonl"
    if not in_path.exists():
        print(f"❌ 找不到结果文件 {in_path}")
        print("   先跑：python -m civility.cli all")
        return 1

    results = load_results(in_path)
    ann_path = Path(args.annotations)
    if not ann_path.is_absolute():
        ann_path = PROJECT / ann_path
    labels = load_annotations(ann_path)

    print("=" * 78)
    print(f"结果 {in_path.name}   窗口 {len(results)}   标注 {len(labels)} 条")
    print(f"行为 {[b.id for b in bs.enabled]}")
    print("=" * 78)
    print()

    # ── 窗口级是三种模式共用的，先单独给出来 ──
    win = score_windows(results, labels, bs,
                        min_confidence=float((cfg.get("aggregate") or {})
                                             .get("min_confidence", 0.45)))
    o = win["overall"]
    print("窗口级（与策略无关，反映模型+Prompt 的真实能力）")
    print(f"  P {fmt(o['precision'])}   R {fmt(o['recall'])}   "
          f"TP {o['tp']}  FP {o['fp']}  FN {o['fn']}")
    for bid, m in win["per_behavior"].items():
        if m["tp"] or m["fp"] or m["fn"]:
            print(f"    {bid:16s} P {fmt(m['precision'])}  R {fmt(m['recall'])}  "
                  f"TP {m['tp']} FP {m['fp']} FN {m['fn']}")
    print()

    # ── 三种策略对比 ──
    print("告警级（同一批结果，只换判定策略）")
    print(f"  {'策略':<11} {'告警数':>6} {'精确率':>8} {'召回率':>8} {'F1':>8}  说明")
    notes = {
        "immediate": "命中即报，最快，误报最多",
        "ai": "模型自己判断该不该报",
        "smoothed": "多窗命中才报，误报最少",
    }
    rows = []
    for mode in MODES:
        agg = dict(cfg.get("aggregate") or {})
        agg["mode"] = mode
        alerts = derive_events(results, bs, agg)
        ev = score_events(alerts, labels, bs)["overall"]
        rows.append((mode, len(alerts), ev))
        print(f"  {mode:<11} {len(alerts):>6} {fmt(ev['precision']):>8} "
              f"{fmt(ev['recall']):>8} {fmt(ev['f1']):>8}  {notes[mode]}")
    print()

    # ── 给出可执行的建议，而不是只丢一张表 ──
    best_p = max(rows, key=lambda r: (r[2]["precision"] or -1))
    best_r = max(rows, key=lambda r: (r[2]["recall"] or -1))
    print("判读")
    print(f"  · 精确率最高：{best_p[0]}（{fmt(best_p[2]['precision'])}）"
          " —— 不想被打扰就用它")
    print(f"  · 召回率最高：{best_r[0]}（{fmt(best_r[2]['recall'])}）"
          " —— 宁可多报也别漏就用它")
    imm = next(r for r in rows if r[0] == "immediate")
    smo = next(r for r in rows if r[0] == "smoothed")
    if imm[2]["precision"] and smo[2]["precision"]:
        gap = (smo[2]["precision"] - imm[2]["precision"]) * 100
        if gap > 5:
            print(f"  · 时序平滑比立即告警精确率高 {gap:.1f} 个百分点 ——")
            print("    说明存在单窗口抖动造成的误报，smoothed/ai 更划算。")
        else:
            print("  · 平滑带来的精确率提升有限（<5pp）——")
            print("    说明模型在单窗口上已经比较稳，用 immediate/ai 换更低延迟更合适。")
    print()
    print("  改策略：config/settings.yaml 的 aggregate.mode，或在网页上直接切换")
    print("  ⚠️ 只改聚合方式，不用重新推理 —— 同一批结果就能对比出结论。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
