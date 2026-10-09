#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""看一眼 A800 上还有没有在跑、在烧钱的作业，并可一键全部停掉。

为什么单独写这个工具（用户的原话："我不用的时候也在花钱"）：
    网页上的「停止并释放节点」只能停**当前这次**的服务作业。而网页一旦重启
    或换了端口，它内存里的作业号就丢了 —— 集群上的作业会继续跑到 12 小时
    上限，钱照扣，页面上却显示"服务未运行"。这种"孤儿作业"是这套架构里
    最烧钱的一种故障，所以需要一个不依赖网页状态的、直接问集群的入口。

用法：
    python tools/cluster_cost.py            # 只看（默认，安全）
    python tools/cluster_cost.py --stop     # 停掉本项目自己的作业
    python tools/cluster_cost.py --stop-all # 停掉 u00009 名下**所有**作业（含非本项目）
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from webui.cluster_client import ClusterClient, ClusterError  # noqa: E402
sys.path.insert(0, str(HERE.parent))
from civility.config import load_settings  # noqa: E402

# 本项目提交的作业名（cluster/serve-vlm.sbatch 等）
OUR_JOB_NAMES = ("civility-serve", "civility-vlm", "civility-batch")

# A800 计费单价（元/卡时）。取自 tju 文档里的平台报价，用于估算，不保证精确。
YUAN_PER_GPU_HOUR = 4.0


def parse_elapsed(s: str) -> float:
    """把 Slurm 的已运行时长（如 1-02:03:04 / 02:03:04 / 3:04）折成小时。"""
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [int(x) for x in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts[-3], parts[-2], parts[-1]
    return days * 24 + h + m / 60 + sec / 3600


def main() -> int:
    ap = argparse.ArgumentParser(description="查看/停止 A800 上的作业（控成本）")
    ap.add_argument("--stop", action="store_true",
                    help="停掉本项目自己的作业（名字以 civility- 开头）")
    ap.add_argument("--stop-all", action="store_true",
                    help="停掉你名下所有作业（含非本项目，谨慎）")
    ap.add_argument("--user", default=None, help="默认用配置里的用户名")
    args = ap.parse_args()

    try:
        cfg = load_settings()
        cc = cfg.get("cluster") or {}
        cl = ClusterClient(
            workdir=cc.get("remote_workdir", "/share/home/u00009/apps/civility-kit"),
            python=cc.get("remote_python",
                          "/share/home/u00009/apps/conda/envs/vlm/bin/python"),
            model_root=cc.get("remote_model_root", "/ssdfs/datahome/u00009/models"),
        )
    except ClusterError as e:
        print(f"❌ 连不上集群：{e}")
        print("   先连学校 VPN，并设置 $env:VLM_SSH_PASSWORD")
        return 2
    except Exception as e:
        print(f"❌ 初始化失败：{type(e).__name__}: {e}")
        return 2

    ok, info = cl.check_vpn()
    if not ok:
        print(f"❌ {info}")
        return 2

    user = args.user or os.environ.get("VLM_SSH_USER") or "u00009"
    # ⚠️ GPU 数必须从 %b（TRES/节点）里解析，**不能**用 %C（CPU 核数）。
    #    踩过的坑：一开始把 %C 当成 GPU 数，而本项目作业申请的是 7 核 1 卡，
    #    于是成本被放大 7 倍（实测把约 4 元报成 27 元）。
    #    成本工具报错数比不报还糟 —— 会让人据此做出错误的省钱决策。
    fmt = "%.10i %.22j %.2t %.12M %.5D %.30b %R"
    try:
        rc, out, err = cl.run(f'squeue -u {user} -h -o "{fmt}"', timeout=60)
    except Exception as e:
        print(f"❌ 查询失败：{type(e).__name__}: {e}")
        return 2
    if rc != 0:
        print(f"❌ squeue 返回 {rc}：{(err or out or '').strip()[:300]}")
        return 2

    rows = [r for r in (out or "").splitlines() if r.strip()]
    print("=" * 78)
    print(f"A800 作业清单（账号 {user}）")
    print("=" * 78)
    if not rows:
        print("✅ 当前没有任何作业在跑 —— 没有在计费。")
        return 0

    print(f"{'作业号':<11}{'名称':<24}{'状态':<5}{'已运行':<13}{'节点':<6}{'GPU':<4}节点/原因")
    jobs = []
    for r in rows:
        p = r.split()
        if len(p) < 6:
            continue
        jid, name, st, el, nodes, tres = p[0], p[1], p[2], p[3], p[4], p[5]
        rest = " ".join(p[6:]) if len(p) > 6 else ""
        # 从 TRES 里数 GPU：形如 gres/gpu:a800:2 或 gres/gpu=2
        n_gpu = 1
        m = re.search(r"gpu(?::[A-Za-z0-9]+)?[:=](\d+)", tres or "")
        if m:
            n_gpu = int(m.group(1))
        elif tres in ("N/A", "(null)", ""):
            n_gpu = 1
        else:
            n_gpu = 0          # TRES 里没有 gres/gpu → 这个作业其实没占卡
        jobs.append((jid, name, st, el, nodes, n_gpu))
        flag = "" if name.startswith(OUR_JOB_NAMES) else " ← 不是本项目"
        print(f"{jid:<11}{name:<24}{st:<5}{el:<13}{nodes:<6}{n_gpu:<4}{rest}{flag}")

    # 估算花费：按"每作业占用的 GPU 数 × 已运行小时 × 单价"
    total_gpu_h = 0.0
    running = [j for j in jobs if j[2] in ("R", "CG")]
    for jid, name, st, el, nodes, n_gpu in running:
        total_gpu_h += n_gpu * parse_elapsed(el)
    print("-" * 78)
    if running:
        print(f"⚠️ 正在运行 {len(running)} 个作业，累计已占用约 "
              f"{total_gpu_h:.1f} GPU·小时 ≈ {total_gpu_h * YUAN_PER_GPU_HOUR:.1f} 元"
              f"（按 {YUAN_PER_GPU_HOUR:.0f} 元/卡时估算）")
    else:
        print("（有作业在队列里，但当前没有处于运行态 —— 排队通常不计费）")

    ours = [j for j in jobs if j[1].startswith(OUR_JOB_NAMES)]
    others = [j for j in jobs if not j[1].startswith(OUR_JOB_NAMES)]
    if others:
        print(f"\n注意：有 {len(others)} 个作业不是本项目提交的"
              f"（{'、'.join(j[1] for j in others[:4])}…），"
              f"用 --stop 不会动它们，避免误杀你自己的训练/实验作业。")

    if not (args.stop or args.stop_all):
        print("\n（本次只查看，没有停任何作业。）")
        if ours:
            print(f"要停掉本项目这 {len(ours)} 个作业："
                  f"python tools/cluster_cost.py --stop")
        return 0

    targets = jobs if args.stop_all else ours
    if not targets:
        print("\n没有需要停的本项目作业。")
        return 0

    ids = " ".join(j[0] for j in targets)
    print(f"\n正在停止 {len(targets)} 个作业：{ids}")
    try:
        rc, res, err = cl.run(
            f"scancel {ids} && sleep 3 && squeue -u {user} -h -o '%i %T'", timeout=120)
    except Exception as e:
        print(f"❌ scancel 失败：{type(e).__name__}: {e}")
        return 2
    if rc != 0:
        print(f"❌ scancel 返回 {rc}：{(err or '').strip()[:300]}")
        return 2
    print(res or "(无输出)")
    left = [r for r in (res or "").splitlines() if r.strip()]
    if not left:
        print("✅ 已全部释放，集群侧不再计费。")
        print("   ⚠️ 网页上如果还显示'运行中'，那是本地状态没刷新 —— "
              "刷新页面或点一次「停止」即可。")
        return 0
    print(f"⚠️ 还剩 {len(left)} 个作业（可能正在取消中）：")
    for r in left:
        print("   ", r)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
