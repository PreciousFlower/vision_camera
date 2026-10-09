#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同济 A800 连通性与环境探针 —— 只读，不消耗机时。

在你开始第一阶段之前跑一次，确认这五件事：
    1. 学校 VPN 通不通（不通则一切免谈，集群没有公网入口）
    2. 登录节点能不能连上、落在哪台
    3. Python 环境与已下好的模型权重是否还在
    4. A800 分区现在有没有空卡（决定要不要排队）
    5. 集群上有没有 pyyaml / qwen-vl-utils（影响作业包怎么打）

用法：
    python tools/cluster_probe.py
先决条件：已连学校 VPN。

认证：优先免密（agent/私钥）；如果没有配免密，
     把密码放进环境变量 VLM_SSH_PASSWORD 再跑（不要写进代码，见 tju/11b-credentials.md）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

TOOLS_DIR = os.environ.get("VLM_TOOLS_DIR", r"E:\vibecoding\aipc\tools")
VLM_PY = "/share/home/u00009/apps/conda/envs/vlm/bin/python"
WORKDIR = "/share/home/u00009/apps/civility-kit"
MODEL_HUB = "/ssdfs/datahome/u00009/models/hf/hub"

# 远端只读检查项：(标题, shell 命令)
PROBES: list[tuple[str, str]] = [
    ("身份与时间", "hostname; whoami; date"),
    ("家目录与磁盘配额", "df -h /share /ssdfs/datahome/u00009 2>/dev/null | tail -3"),
    (
        "Python 环境",
        f"{VLM_PY} -c \"import sys;print('python',sys.version.split()[0])\" 2>&1 | head -3",
    ),
    (
        "已装关键包",
        f"{VLM_PY} -c \"import importlib;[print(m, getattr(importlib.import_module(m),'__version__','?')) "
        "for m in ['torch','transformers','PIL','numpy']]\" 2>&1 | head -8",
    ),
    (
        "作业包所需依赖（决定要不要额外装）",
        f"{VLM_PY} -c \"import importlib\nfor m in ['yaml','qwen_vl_utils']:\n"
        "    try:\n        importlib.import_module(m);print(m,'OK')\n"
        "    except Exception as e:\n        print(m,'MISSING')\" 2>&1 | head -5",
    ),
    ("已下载的模型权重", f"ls -1 {MODEL_HUB} 2>/dev/null"),
    (
        "A800 分区空闲情况",
        "sinfo -p A800 -o '%12N %8t %12C %26G' 2>/dev/null | head -22",
    ),
    ("我的作业队列", "squeue -u u00009 -o '%.10i %.20j %.9T %.11M %R' 2>/dev/null"),
    ("已有工作目录", f"ls -la {WORKDIR} 2>/dev/null || echo '（还没有，cluster-upload 会创建）'"),
]


def main() -> int:
    if TOOLS_DIR not in sys.path:
        sys.path.insert(0, TOOLS_DIR)
    try:
        import cluster  # type: ignore
    except ImportError:
        print(f"❌ 找不到 {TOOLS_DIR}\\cluster.py")
        print("   用环境变量 VLM_TOOLS_DIR 指定 aipc 仓库的 tools 目录")
        return 1

    pw = os.environ.get("VLM_SSH_PASSWORD")
    kw = {"password": pw, "allow_agent": False, "look_for_keys": False} if pw else {}
    print(f"认证方式：{'密码（来自 VLM_SSH_PASSWORD）' if pw else '免密（agent/私钥）'}")

    print("\n[0] 尝试连接登录节点…")
    print("    ⚠️ 前提：学校 VPN 已连。没连的话这里必然失败。")
    try:
        rc, out, err = cluster.run("hostname; whoami", timeout=30, **kw)
    except Exception as e:
        print(f"❌ 连接失败：{type(e).__name__}: {e}")
        print("\n排查顺序：")
        print("  1. ping 192.168.212.236    ← 100% 丢包 = VPN 没连")
        print("  2. 端口必须是 10022（不是 22）")
        print("  3. 换备用 IP：192.168.212.245")
        print("  4. 认证：配免密，或设置 VLM_SSH_PASSWORD 环境变量")
        return 1
    print(f"✅ 已连接：{out.strip().replace(chr(10), ' / ')}")

    for title, cmd in PROBES:
        print(f"\n[{title}]")
        try:
            rc, out, err = cluster.run(cmd, timeout=90, **kw)
            text = (out or err or "(无输出)").rstrip()
            for line in text.splitlines():
                print(f"    {line}")
            if rc != 0:
                print(f"    （退出码 {rc}）")
        except Exception as e:
            print(f"    ⚠️ 执行失败：{type(e).__name__}: {e}")

    print("\n" + "─" * 60)
    print("接下来（都要在连 VPN 的前提下）：")
    print("  1. 本地跑通 mock 链路：")
    print("       cd phase1 && python -m civility.cli all")
    print("  2. 换成云端 API 调 Prompt（主力路径）：")
    print("       $env:DASHSCOPE_API_KEY='...'")
    print("       python -m civility.cli run --backend cloud")
    print("  3. 走集群本地模型（免费，但要排队）：")
    print("       python -m civility.cli cluster-prepare")
    print("       python -m civility.cli cluster-upload")
    print("       ssh -p 10022 u00009@192.168.212.236")
    print(f"       cd {WORKDIR} && sbatch cluster/run-vlm.sbatch")
    print("       python -m civility.cli cluster-fetch")
    print("       python -m civility.cli run --backend cluster")
    return 0


if __name__ == "__main__":
    sys.exit(main())
