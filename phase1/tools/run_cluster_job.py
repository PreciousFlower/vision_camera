#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""提交一个"跑完就退"的集群作业并等它结束，把日志取回来。

为什么需要它：
    集群侧的小任务（微基准、显存探测、单次批量推理）用 sbatch 提交后，
    如果没有一个统一的"提交→等待→取日志"入口，就得手工 squeue 反复看、
    再手工 tail 日志 —— 而每次手工操作都可能忘记检查作业是否真的结束了
    （忘了就是白烧卡时，这是本项目最贵的一类错误）。

用法：
    python tools/run_cluster_job.py cluster/bench-vlm.sbatch --timeout 1800
    python tools/run_cluster_job.py cluster/bench-vlm.sbatch --export "MODEL=... ,CONFIGS=A"
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from civility.config import load_settings          # noqa: E402
from webui.cluster_client import ClusterClient     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sbatch_rel", help="相对工作目录的 sbatch 路径")
    ap.add_argument("--timeout", type=int, default=1800, help="最长等待秒数")
    ap.add_argument("--export", default="", help="额外 --export 变量，如 A=1,B=2")
    ap.add_argument("--tail", type=int, default=120, help="回读日志的行数")
    args = ap.parse_args()

    cc = load_settings().get("cluster") or {}
    wd = cc["remote_workdir"]
    cl = ClusterClient(workdir=wd, python=cc["remote_python"],
                       model_root=cc["remote_model_root"])
    ok, host = cl.check_vpn()
    if not ok:
        print(f"❌ 连不上集群（{host}）")
        return 2
    print(f"✅ {host}")

    # 提交前先确认没有别人在跑，避免"两个作业同时占卡、双倍计费"
    rc, out, _ = cl.run('squeue -u u00009 -h -o "%i %j %T"', timeout=60)
    running = [l for l in (out or "").splitlines() if l.strip()]
    if running:
        print("⚠️ 当前已有作业在跑：")
        for l in running:
            print("   ", l)
        print("   （继续提交会同时占两张卡，确认这是你要的吗？Ctrl+C 可中止）")
        time.sleep(5)

    extra = f"--export=ALL,{args.export}" if args.export else ""
    jid = cl.client.submit(args.sbatch_rel, extra=extra) if hasattr(cl, "client") \
        else cl.submit(args.sbatch_rel, extra=extra)
    print(f"已提交作业 {jid}，等待中（最长 {args.timeout}s）…")

    t0 = time.time()
    last = ""
    while time.time() - t0 < args.timeout:
        st, node = cl.job_state(jid)
        if st != last:
            print(f"   [{time.time() - t0:5.0f}s] 状态={st or '已结束'} 节点={node}")
            last = st
        if st in ("", "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT",
                  "NODE_FAIL", "OUT_OF_MEMORY"):
            break
        time.sleep(10)

    elapsed = time.time() - t0
    print(f"\n作业结束（等待 {elapsed:.0f}s，状态 {last or '未知'}）")
    log = f"{wd}/logs/bench-{jid}.out"
    rc, out, err = cl.run(f"tail -n {args.tail} {log} 2>/dev/null", timeout=120)
    print(f"\n───── {log} 末尾 ─────")
    print(out or err or "（没取到日志）")

    # 结果 JSON（如果有）
    rc, out2, _ = cl.run(f"cat {wd}/logs/bench_result.json 2>/dev/null | head -c 4000",
                         timeout=60)
    if out2 and out2.strip().startswith("{"):
        (ROOT / "runs" / "bench_result.json").write_text(out2, encoding="utf-8")
        print(f"\n（结果 JSON 已存到 runs/bench_result.json，{len(out2)} 字节）")

    print(f"\n💡 本次作业占用时间约 {elapsed / 60:.1f} 分钟"
          f"（结束即释放，不会继续计费）")
    return 0 if last in ("COMPLETED", "") else 1


if __name__ == "__main__":
    raise SystemExit(main())
