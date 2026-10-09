#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""集群快探 —— 为"网页 + 服务器后台"方案做可行性侦察。

比 cluster_probe.py 快得多：只建一次连接，把所有检查塞进一条远端命令。
默认带超时保护，不会像逐项重连那样卡住。

用法：
    set VLM_SSH_PASSWORD=...        # 或配好免密
    python tools/cluster_recon.py
"""
from __future__ import annotations

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

# 一条命令查完所有东西，只为少建几次连接（每次连接要几秒）
REMOTE = r"""
echo "### 身份"; hostname; whoami; date
echo "### 负载"; uptime; echo "核数 $(nproc)"; free -g | head -2
echo "### Python"
VLM_PY=/share/home/u00009/apps/conda/envs/vlm/bin/python
$VLM_PY -c "import sys;print('python',sys.version.split()[0])"
echo "### 关键包"
$VLM_PY - <<'PYEOF'
import importlib
for m in ["torch","transformers","yaml","qwen_vl_utils","fastapi","uvicorn","flask","PIL"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:16s} OK  {getattr(mod,'__version__','?')}")
    except Exception:
        print(f"  {m:16s} MISSING")
PYEOF
echo "### apps 目录"; ls -1 /share/home/u00009/apps/ 2>/dev/null
echo "### 权重"; ls -1 /ssdfs/datahome/u00009/models/hf/hub/ 2>/dev/null
echo "### A800 空闲卡"
sinfo -p A800 -o "%12N %8t %14C %26G" 2>/dev/null | head -22
echo "### 我在排的作业"; squeue -u u00009 -o "%.10i %.20j %.9T %.11M %R" 2>/dev/null
echo "### 已装的 tmux/screen（决定能否常驻进程）"
which tmux screen nohup 2>/dev/null || echo "  (none)"
echo "### 是否限制长时进程（登录节点策略线索）"
echo "  ulimit -u = $(ulimit -u)   max user processes"
"""


def main() -> int:
    try:
        import paramiko  # noqa: F401
    except ImportError:
        print("❌ 需要 paramiko：pip install paramiko")
        return 1

    import cluster  # type: ignore

    pw = os.environ.get("VLM_SSH_PASSWORD")
    kw = {"password": pw, "allow_agent": False, "look_for_keys": False} if pw else {}
    print(f"认证：{'密码' if pw else '免密'}    工具目录：{TOOLS_DIR}")

    t0 = time.time()
    try:
        c = cluster.connect(timeout=25, **kw)
    except Exception as e:
        print(f"❌ 连接失败：{type(e).__name__}: {e}")
        return 1
    print(f"✅ 已连接 {getattr(c, '_vlm_host', '?')}（{time.time() - t0:.1f}s）\n")

    try:
        _, out, err = c.exec_command(REMOTE, timeout=120)
        print(out.read().decode("utf-8", "replace"))
        e = err.read().decode("utf-8", "replace")
        if e.strip():
            print("--- stderr ---")
            print(e[:1500])
    finally:
        c.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
