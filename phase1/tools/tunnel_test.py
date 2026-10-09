#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""隧道可行性测试 —— 决定 Web 界面用哪种架构。

要回答的问题：
    1. 登录节点能不能 SSH 到计算节点？（决定能否做"常驻推理服务 + 隧道"）
    2. 计算节点能不能反向连回登录节点？
    3. 计算节点能不能访问外网？（预期不能 —— 影响模型下载和 API 调用）
    4. 能不能 pip 装包？（决定 Web 框架放哪）

为什么这件事是决定性的：
    用户想要"本地网页 + 服务器后台"。但计算节点没有公网出口，
    浏览器也连不到计算节点。所以只有两条路：
        A. 网页在本机，通过 SSH 把帧送上去、把结果拉回来（稳，但每次要等 SSH）
        B. 在计算节点常驻一个服务，用 SSH 隧道把它映射到本机端口
           （实时，但需要登录节点能跳到计算节点、且能建隧道）
    本脚本判定 B 是否可行。

用法：
    set VLM_SSH_PASSWORD=...
    python tools/tunnel_test.py
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

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
LOCAL_PROBE = os.path.join(PROJECT, "cluster", "probe_compute.sh")
REMOTE_DIR = "/share/home/u00009/apps/civility-kit"


def main() -> int:
    import cluster  # type: ignore

    pw = os.environ.get("VLM_SSH_PASSWORD")
    kw = {"password": pw, "allow_agent": False, "look_for_keys": False} if pw else {}

    if not os.path.exists(LOCAL_PROBE):
        print(f"❌ 找不到 {LOCAL_PROBE}")
        return 1

    print("=" * 62)
    print("步骤 0：上传探测脚本到登录节点")
    print("=" * 62)
    try:
        rc, out, err = cluster.run(f"mkdir -p {REMOTE_DIR}/cluster", timeout=60, **kw)
        c, sftp = cluster.sftp(timeout=25, **kw)
        try:
            sftp.put(LOCAL_PROBE, f"{REMOTE_DIR}/cluster/probe_compute.sh")
            print(f"✅ 已上传 → {REMOTE_DIR}/cluster/probe_compute.sh")
        finally:
            sftp.close()
            c.close()
    except Exception as e:
        print(f"❌ 上传失败：{type(e).__name__}: {e}")
        return 1

    print()
    print("=" * 62)
    print("步骤 1：在计算节点上执行探测（申请 1 卡 5 分钟）")
    print("=" * 62)
    print("正在申请 1 卡，可能要排队…")
    # 只申请 2 核 8G 5 分钟 —— 按需申请才排得上队（见 tju 文档 §2.3）
    srun_cmd = (
        "srun -p A800 --gres=gpu:a800:1 --cpus-per-task=2 --mem=8g "
        "--time=0-00:05:00 --exclude=gpu8002,gpu8003,gpu8012 "
        f"bash {REMOTE_DIR}/cluster/probe_compute.sh"
    )
    t0 = time.time()
    try:
        rc, out, err = cluster.run(srun_cmd, timeout=480, **kw)
        print(out or "")
        if err.strip():
            print("--- stderr ---")
            print(err[:2000])
        print(f"[退出码 {rc}  用时 {time.time() - t0:.0f}s]")
    except Exception as e:
        print(f"❌ 失败：{type(e).__name__}: {e}")
        print("   排队太久也会超时，这本身说明 A800 忙。")
        return 1

    print()
    print("=" * 62)
    print("步骤 2：登录节点能否做端口转发（ssh -L 所需）")
    print("=" * 62)
    try:
        rc, out, err = cluster.run(
            "which ssh; echo '---'; ssh -V 2>&1; echo '--- 转发相关配置 ---'; "
            "grep -i -E '^[[:space:]]*(AllowTcpForwarding|GatewayPorts|PermitTunnel)' "
            "/etc/ssh/sshd_config 2>/dev/null || echo '(读不到，用实测确认)'",
            timeout=60, **kw,
        )
        print(out or err)
    except Exception as e:
        print(f"⚠️ {e}")

    print()
    print("=" * 62)
    print("下一步")
    print("=" * 62)
    print("""
  看"计算节点 → 登录节点 可达性"这一节：
    ✅ 可达  → 走方案 B：计算节点常驻推理服务 + 本机 SSH 隧道，网页可实时交互
    ❌ 不可达 → 走方案 A：本机 web 后端通过 SSH 传帧/收结果（每次等 SSH 往返）

  看"外网"这一节：
    不通是预期的 —— 集群上只能用已下好的 Qwen3-VL 权重，不能调云端 API。

  看"pip 装包"这一节：
    失败是预期的 —— 所以 Web 框架要么放本机，要么用纯标准库写（无依赖）。
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
