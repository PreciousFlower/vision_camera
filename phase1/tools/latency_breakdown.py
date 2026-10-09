#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把最近任务的"延迟分解"打出来：输入/输出 token、耗时、是否跳帧。

为什么要单独看这个：
    "慢"的根因是**输出 token**（实测 35.8 tok/s ≈ 每 token 28 毫秒），
    所以优化到底有没有生效，只能看 tokens_out 降没降 ——
    光看总延迟会被"读图/网络/排队"的抖动误导。

用法：
    python tools/latency_breakdown.py            # 读 last_port.txt 指向的实例
    python tools/latency_breakdown.py --port 8849
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()
    if args.port is None:
        try:
            args.port = int((ROOT / "webui" / "last_port.txt")
                            .read_text(encoding="utf-8").strip())
        except Exception:
            args.port = 8848
    base = f"http://127.0.0.1:{args.port}"

    with urllib.request.urlopen(base + "/api/state", timeout=20) as r:
        st = json.loads(r.read().decode("utf-8"))
    jobs = st.get("jobs") or []
    print(f"目标 {base}　任务数 {len(jobs)}")
    if not jobs:
        print("（还没有任务）")
        return 1

    for j in jobs[:4]:
        jid = j["job_id"]
        try:
            with urllib.request.urlopen(base + f"/api/job/{jid}", timeout=30) as r:
                d = json.loads(r.read().decode("utf-8"))
        except Exception as e:
            print(f"\n{jid}: 取不到详情（{type(e).__name__}）")
            continue
        wres = d.get("window_results") or []
        print(f"\n=== 任务 {jid}　源={d.get('filename')}　"
              f"mock={d.get('use_mock')}　窗口 {len(wres)} ===")
        if not wres:
            continue
        print(f"{'#':>3} {'耗时':>7} {'输入tok':>8} {'输出tok':>8} "
              f"{'tok/s':>6} {'跳帧':>5}  场景")
        outs, lats = [], []
        for i, w in enumerate(wres, 1):
            lat = w.get("latency") or 0.0
            ti = w.get("tokens_prompt") or 0
            to = w.get("tokens_out") or 0
            tps = round(to / lat, 1) if (lat and to) else 0
            if not w.get("skipped"):
                outs.append(to)
                lats.append(lat)
            print(f"{i:>3} {lat:>6.1f}s {ti:>8} {to:>8} {tps:>6} "
                  f"{'是' if w.get('skipped') else '否':>5}  "
                  f"{str(w.get('scene'))[:38]}")
        if outs:
            print(f"\n  实际调用模型 {len(outs)} 次："
                  f"平均耗时 {sum(lats)/len(lats):.1f}s，"
                  f"平均输出 {sum(outs)/len(outs):.0f} token")
            print(f"  按输出算的解码占比：{sum(outs)*0.028:.1f}s"
                  f"（{sum(outs)*0.028/sum(lats)*100:.0f}% 的时间花在逐字生成上）")
        skipped = sum(1 for w in wres if w.get("skipped"))
        if skipped:
            print(f"  跳过模型调用 {skipped}/{len(wres)} 窗（画面无变化）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
