#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 A800 上的用量汇总成"GPU·小时 + 估算金额"，用于对账和控制成本。

为什么需要它：
    sacct 的输出是一行一个作业，几百行里看不出"到底花了多少"。
    而且**真正花钱的是 GPU·小时**（占用时长 × 卡数），
    不是作业个数 —— 一个跑了 12 小时的服务作业比一百个 26 秒的探测作业贵得多。
    这个脚本按作业名分组做这个乘法，直接给出金额量级。

用法：
    python tools/cluster_usage.py                 # 近 7 天
    python tools/cluster_usage.py --days 30
    python tools/cluster_usage.py --days 1 --detail
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from civility.config import load_settings          # noqa: E402
from webui.cluster_client import ClusterClient     # noqa: E402

YUAN_PER_GPU_HOUR = 4.0     # 平台报价（元/卡时），用于估算


def elapsed_hours(s: str) -> float:
    """Slurm 的 Elapsed：'1-02:03:04' / '02:03:04' / '3:04' → 小时。"""
    s = (s or "").strip()
    if not s or s in ("00:00:00", "INVALID"):
        return 0.0
    if not re.fullmatch(r"[\d\-:]+", s):
        return 0.0          # 表头或异常值（'Elapsed'）→ 当成 0，别让脚本崩
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        try:
            days = int(d)
        except ValueError:
            return 0.0
    parts = [int(x) for x in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return days * 24 + parts[-3] + parts[-2] / 60 + parts[-1] / 3600

def gpus_of(tres: str) -> int:
    """从 AllocTRES 里抠出 GPU 数（'...gres/gpu=1'）；抠不到就按 1 张算。"""
    m = re.search(r"gres/gpu(?:[:=])(\d+)", tres or "")
    if m:
        return int(m.group(1))
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="A800 用量与估算花费")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--detail", action="store_true", help="逐个作业列出")
    ap.add_argument("--user", default="u00009")
    args = ap.parse_args()

    cc = load_settings().get("cluster") or {}
    cl = ClusterClient(workdir=cc["remote_workdir"], python=cc["remote_python"],
                       model_root=cc["remote_model_root"])

    fmt = ("JobID%12,JobName%22,State%12,Elapsed%14,AllocTRES%28")
    cmd = (f"sacct -u {args.user} --starttime=now-{args.days}days "
           f"--format={fmt} -X -P 2>/dev/null")
    rc, out, err = cl.run(cmd, timeout=180)
    if rc != 0:
        print(f"❌ sacct 失败（rc={rc}）：{(err or '').strip()[:300]}")
        return 2

    rows = []
    for line in (out or "").splitlines():
        p = line.split("|")
        if len(p) < 5 or not p[0].strip():
            continue
        jid, name, state, el, tres = p[0], p[1], p[2], p[3], p[4]
        if jid.strip() in ("JobID", "") or name.strip() == "JobName":
            continue        # sacct -P 也会输出表头，不排掉会被当成一个作业
        if jid.endswith(".extern") or jid.endswith(".batch"):
            continue
        rows.append((jid, name, state, el, gpus_of(tres)))

    if not rows:
        print(f"近 {args.days} 天没有任何作业记录 —— 没有在计费。")
        return 0

    # 只有真正占用了卡的才计费：COMPLETED / TIMEOUT / FAILED / CANCELLED 都按实际时长算，
    # 但 PENDING / 0 秒的忽略。
    by_name: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "gpu_h": 0.0, "states": set()})
    total_gpu_h = 0.0
    for jid, name, state, el, ngpu in rows:
        h = elapsed_hours(el)
        if h <= 0:
            continue
        gh = h * max(1, ngpu)
        total_gpu_h += gh
        b = by_name[name]
        b["n"] += 1
        b["gpu_h"] += gh
        b["states"].add(state.split()[0])

    print("=" * 76)
    print(f"A800 用量汇总（账号 {args.user}，近 {args.days} 天）")
    print("=" * 76)
    print(f"{'作业名':<24}{'次数':>6}{'GPU·小时':>12}{'估算金额':>12}   状态")
    print("-" * 76)
    for name, b in sorted(by_name.items(), key=lambda kv: -kv[1]["gpu_h"]):
        yuan = b["gpu_h"] * YUAN_PER_GPU_HOUR
        print(f"{name:<24}{b['n']:>6}{b['gpu_h']:>12.2f}{yuan:>11.1f} 元   "
              f"{'/'.join(sorted(b['states']))}")
    print("-" * 76)
    print(f"{'合计':<24}{len(rows):>6}{total_gpu_h:>12.2f}"
          f"{total_gpu_h * YUAN_PER_GPU_HOUR:>11.1f} 元")
    print(f"\n（按 {YUAN_PER_GPU_HOUR:.0f} 元/卡时估算，仅供对账参考；"
          f"实际以平台账单为准）")

    if args.detail:
        print("\n--- 明细（按占用 GPU·小时降序）---")
        for jid, name, state, el, ngpu in sorted(
                rows, key=lambda r: -elapsed_hours(r[3]) * r[4])[:40]:
            h = elapsed_hours(el)
            print(f"  {jid:<12}{name:<22}{state:<12}{el:<14}"
                  f"{ngpu} 卡 × {h:6.3f}h = {h * ngpu:7.3f} GPU·h")

    print("\n提示：真正烧钱的是**常驻服务作业**（12 小时上限 × 卡数）。")
    print("     用完请点网页「停止并释放节点」，或跑 "
          "`python tools/cluster_cost.py --stop`。")
    print("     网页现在还有「空闲自动释放」：页面关掉/切到后台超过设定分钟数，"
          "会自动帮你把节点退掉。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
