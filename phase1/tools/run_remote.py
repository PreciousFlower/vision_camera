#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""上传任意脚本到集群并执行 —— 避免 PowerShell 的引号地狱。

PowerShell 的 here-string 会把 `$(...)` 当自己的子表达式求值，
写 bash 脚本进命令行几乎必炸（已经踩了两次）。所以：本地用 write 工具
写好 .sh 文件，用本脚本上传执行。

用法：
    set VLM_SSH_PASSWORD=...
    python tools/run_remote.py cluster/scan_all_nodes.sh --timeout 1800
"""
from __future__ import annotations

import argparse
import os
import sys
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

TOOLS_DIR = os.environ.get("VLM_TOOLS_DIR", r"E:\vibecoding\aipc\tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

REMOTE_DIR = "/share/home/u00009/apps/civility-kit"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("script", help="本地脚本路径（相对 phase1/）")
    ap.add_argument("--timeout", type=int, default=1800, help="远端执行超时（秒）")
    ap.add_argument("--no-upload", action="store_true", help="跳过上传，直接执行远端已有脚本")
    args = ap.parse_args()

    import cluster  # type: ignore

    pw = os.environ.get("VLM_SSH_PASSWORD")
    kw = {"password": pw, "allow_agent": False, "look_for_keys": False} if pw else {}

    here = os.path.dirname(os.path.abspath(__file__))
    project = os.path.dirname(here)
    local = args.script if os.path.isabs(args.script) else os.path.join(project, args.script)
    name = os.path.basename(local)
    remote = f"{REMOTE_DIR}/cluster/{name}"

    if not args.no_upload:
        if not os.path.exists(local):
            print(f"❌ 本地找不到 {local}")
            return 1
        try:
            cluster.run(f"mkdir -p {REMOTE_DIR}/cluster", timeout=60, **kw)
            c, sftp = cluster.sftp(timeout=25, **kw)
            try:
                sftp.put(local, remote)
            finally:
                sftp.close()
                c.close()
            print(f"✅ 已上传 → {remote}")
        except Exception as e:
            print(f"❌ 上传失败：{type(e).__name__}: {e}")
            return 1

    print(f"▶ 执行：bash {remote}")
    t0 = time.time()
    try:
        rc, out, err = cluster.run(f"cd {REMOTE_DIR} && bash {remote} 2>&1", timeout=args.timeout, **kw)
        print(out or "(无输出)")
        if err.strip():
            print("--- stderr ---")
            print(err[:2000])
        print(f"[rc={rc}  用时 {time.time() - t0:.0f}s]")
        return rc
    except Exception as e:
        print(f"❌ 执行异常：{type(e).__name__}: {e}")
        print(f"   （超时设为 {args.timeout}s；排队久时会被截断，可加大 --timeout）")
        return 1


if __name__ == "__main__":
    sys.exit(main())
