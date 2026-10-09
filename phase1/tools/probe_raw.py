#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单窗口原始输出探测 —— 排查"为什么没报出来"。

验证脚本给的是汇总判断，但排查问题需要看**模型的原话**：
它可能认出了动作但没往 littering 上靠，也可能压根没看出来。
这两种情况的修法完全不同（改 Prompt vs 改视觉输入）。

用法：
    python tools/probe_raw.py litter_visual_vis       # 指定视频子串
    python tools/probe_raw.py litter_visual_vis --frame 2   # 换一帧
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help="视频名子串（对应 data/frames/<stem>/）")
    ap.add_argument("--base", default="http://127.0.0.1:8848")
    ap.add_argument("--n-frames", type=int, default=4)
    ap.add_argument("--start", type=int, default=0, help="从第几帧开始取")
    ap.add_argument("--few-shot", action="store_true")
    args = ap.parse_args()

    import requests

    B = args.base.rstrip("/")
    st = requests.get(f"{B}/api/state", timeout=15).json()
    if st["service"]["state"] != "ready":
        print(f"❌ 服务未就绪（{st['service']['state']}）")
        return 1
    url = st["service"]["local_url"]

    # 找帧目录
    root = PROJECT / "data" / "frames"
    cands = [d for d in root.glob("*") if args.pattern in d.name]
    if not cands:
        print(f"❌ data/frames 下没有匹配 {args.pattern!r} 的目录")
        print("   现有：", [d.name for d in root.glob('*')])
        return 1
    d = cands[0]
    files = sorted(d.glob("f_*.jpg"))
    sel = files[args.start: args.start + args.n_frames]
    if len(sel) < 2:
        print(f"❌ 帧不足（{len(files)} 个）")
        return 1

    print("=" * 78)
    print(f"帧目录 {d.name}   取 {len(sel)} 帧（第 {args.start}–{args.start + len(sel) - 1} 张）")
    print(f"时间标签 {[f't={args.start + i}.0s' for i in range(len(sel))]}")
    print(f"端点 {url}/infer   few_shot={args.few_shot}")
    print("=" * 78)

    # 顺便报一下这几帧的亮度，确认不是黑图
    from PIL import Image, ImageStat

    for p in sel:
        s = ImageStat.Stat(Image.open(p).convert("L"))
        print(f"  {p.name}  亮度均值 {s.mean[0]:6.1f}  标准差 {s.stddev[0]:6.1f}")
    print()

    payload = {
        "frames": [{"b64": base64.b64encode(p.read_bytes()).decode()} for p in sel],
        "labels": [f"t={args.start + i}.0s" for i in range(len(sel))],
        "few_shot": args.few_shot,
    }
    r = requests.post(f"{url}/infer", json=payload, timeout=300)
    if r.status_code != 200:
        print(f"❌ HTTP {r.status_code}: {r.text[:500]}")
        return 1
    res = r.json()

    print(f"耗时 {res.get('latency_sec')}s   prompt {res.get('tokens_prompt')} tok   "
          f"输出 {res.get('tokens_out')} tok   {res.get('tok_per_sec')} tok/s")
    if res.get("error"):
        print(f"⚠️ {res['error']}")
    print()
    print("── 模型的原始输出（未经解析）──")
    print(res.get("raw") or "(空)")
    print()
    parsed = res.get("parsed")
    if parsed:
        print("── 解析后的结构 ──")
        print(f"scene: {parsed.get('scene')}")
        for bid, v in (parsed.get("violations") or {}).items():
            mark = "🔴" if v.get("violation") else "⚪"
            print(f"  {mark} {bid:16s} conf={v.get('confidence', 0):.2f}")
            if v.get("observable"):
                print(f"       依据：{v['observable']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
