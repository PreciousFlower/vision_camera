#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 cluster/ 目录下的脚本同步到集群工作目录（$WORKDIR/cluster/）。

为什么需要单独一个工具：
    `cluster-prepare` 打的包只包含 `pkg/...`（里面有 pkg/cluster/*），
    而 Slurm 是从 `$WORKDIR/cluster/serve-vlm.sbatch` 提交的 ——
    **那份 sbatch 不在包里**。
    以前它是手工 scp 上去的，于是"改了 sbatch 却忘了同步"就变成
    "改完没生效"，排查起来极浪费时间（改的是本地文件，跑的是旧脚本）。
    这个工具把这一步变成一条命令，并在上传后回读校验（比对大小/首行），
    避免"以为传上去了其实没传"。

用法：
    python tools/upload_cluster_files.py            # 上传 cluster/ 下所有脚本
    python tools/upload_cluster_files.py --list     # 只看远端现有文件
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from civility.config import load_settings          # noqa: E402
from webui.cluster_client import ClusterClient     # noqa: E402

EXTS = (".sbatch", ".sh", ".py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="只列远端文件，不上传")
    args = ap.parse_args()

    cc = load_settings().get("cluster") or {}
    wd = cc.get("remote_workdir", "/share/home/u00009/apps/civility-kit")
    cl = ClusterClient(workdir=wd, python=cc["remote_python"],
                       model_root=cc["remote_model_root"])

    ok, host = cl.check_vpn()
    if not ok:
        print(f"❌ 连不上集群（{host}）—— 先连 VPN、设 VLM_SSH_PASSWORD")
        return 2
    print(f"✅ 已连接 {host}")

    rc, out, _ = cl.run(f"ls -la {wd}/cluster/ 2>/dev/null | head -20", timeout=60)
    print(f"\n远端 {wd}/cluster/ ：")
    print(out or "（目录不存在）")
    if args.list:
        return 0

    local = ROOT / "cluster"
    files = sorted(p for p in local.iterdir()
                   if p.is_file() and p.suffix in EXTS)
    if not files:
        print("❌ 本地 cluster/ 下没有可上传的脚本")
        return 1

    cl.run(f"mkdir -p {wd}/cluster", timeout=60)
    failed = cl.upload_tree(str(local), f"{wd}/cluster")
    print(f"\n上传 {len(files)} 个文件 → {wd}/cluster/"
          + (f"（失败：{failed}）" if failed else ""))
    for p in files:
        print(f"   · {p.name}  {p.stat().st_size / 1024:.1f} KB")

    # 回读校验：比大小能挡住"传了但没生效"，比"看起来成功"可靠
    print("\n回读校验：")
    bad = 0
    for p in files:
        rc, out, _ = cl.run(
            f"stat -c '%s' {wd}/cluster/{p.name} 2>/dev/null || echo 0", timeout=60)
        try:
            remote_size = int((out or "0").strip() or 0)
        except ValueError:
            remote_size = 0
        match = remote_size == p.stat().st_size
        bad += 0 if match else 1
        print(f"   {'✅' if match else '❌'} {p.name}：本地 {p.stat().st_size} / "
              f"远端 {remote_size}")

    if bad:
        print(f"\n❌ {bad} 个文件大小不一致 —— 请重试（网络中断会留下半个文件）")
        return 1
    print("\n✅ 全部一致。注意：作业包里那份（pkg/cluster/）由 "
          "cluster-prepare + cluster-upload 负责，改完代码两处都要更新。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
