#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""盯住集群作业直到它**真正**消失（不是"状态变成 CG"就算完）。

为什么需要：
    常驻推理服务被 scancel 后会先进入 COMPLETING(CG) —— 这时候进程还在收尾，
    节点仍被占着。只看一眼 squeue 很容易误判成"已经释放了"，然后走人 ——
    而计费是按占用时间算的。实测踩过：CG 状态持续了一分多钟。

用法：
    python tools/watch_jobs.py                # 盯到没有作业为止（最多 3 分钟）
    python tools/watch_jobs.py --timeout 600
"""
from __future__ import annotations

import argparse
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
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()
    cc = load_settings().get("cluster") or {}
    cl = ClusterClient(workdir=cc["remote_workdir"], python=cc["remote_python"],
                       model_root=cc["remote_model_root"])
    ok, host = cl.check_vpn()
    if not ok:
        print(f"❌ 连不上集群（{host}）")
        return 2

    t0 = time.time()
    last = None
    while time.time() - t0 < args.timeout:
        rc, out, _ = cl.run("squeue -u u00009 -h -o '%i %T %M %N'", timeout=60)
        s = (out or "").strip()
        if not s:
            print(f"✅ [{time.time() - t0:5.0f}s] 已无作业 —— 节点释放完毕，停止计费")
            return 0
        if s != last:
            print(f"[{time.time() - t0:5.0f}s] {s}")
            last = s
        time.sleep(6)
    print(f"⚠️ {args.timeout}s 后仍有作业在队列里：\n{last}")
    print("   若一直是 CG（COMPLETING）超过几分钟，说明有进程没退出，")
    print("   可以再跑一次 python tools/cluster_cost.py --stop")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
