#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端验证：本机 → 隧道 → 计算节点常驻模型 → 真实推理。

这条链路是 Web 界面的全部依赖。任何一环坏了，网页都是空壳。
所以这里不走网页，直接用 Python 跑一遍，把每一步的证据打出来。

用法：
    set VLM_SSH_PASSWORD=...
    python tools/e2e_verify.py            # 启动服务 + 推理验证
    python tools/e2e_verify.py --keep     # 验证后不停止服务（留着给网页用）
    python tools/e2e_verify.py --attach   # 复用已启动的服务，不重新提交
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

import yaml  # noqa: E402


def log(m: str = "") -> None:
    print(m, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="验证后保留服务")
    ap.add_argument("--attach", action="store_true", help="复用已在运行的服务")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    args = ap.parse_args()

    cfg = yaml.safe_load((PROJECT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    cl_cfg = cfg["cluster"]

    from webui.cluster_client import ClusterClient, ServiceSession, Tunnel

    cl = ClusterClient(
        cl_cfg["remote_workdir"], cl_cfg["remote_python"], cl_cfg["remote_model_root"]
    )

    # ── 1. 连通性 ──
    log("=" * 66)
    log("步骤 1  连通性")
    log("=" * 66)
    ok, info = cl.check_vpn()
    if not ok:
        log(f"❌ 连不上登录节点：{info}")
        log("   检查：① 学校 VPN ② VLM_SSH_PASSWORD 或免密是否配好")
        return 1
    log(f"✅ 登录节点 {info}")

    # ── 2. 远端语法检查 ──
    log()
    log("=" * 66)
    log("步骤 2  远端脚本语法检查（避免浪费一次排队）")
    log("=" * 66)
    for rel in ("pkg/cluster/vlm_service.py", "pkg/civility/prompts.py"):
        good, out = cl.remote_py_compile(rel)
        log(f"  {'✅' if good else '❌'} {rel}")
        if not good:
            log("    " + out.strip()[:400])
            return 1

    # ── 3. 启动服务 ──
    sess = ServiceSession(cl, model=args.model,
                          port=int(cl_cfg.get("service_port", 18080)))
    if args.attach:
        log()
        log("=" * 66)
        log("步骤 3  复用已运行的服务")
        log("=" * 66)
        rc, out, _ = cl.run(
            f"cat {cl.workdir}/logs/serving-node.txt 2>/dev/null", timeout=60
        )
        parts = (out or "").split()
        if not parts:
            log("❌ 找不到 serving-node.txt，说明没有在跑的服务。去掉 --attach 重试。")
            return 1
        sess.node, sess.port = parts[0], int(parts[1])
        log(f"  作业运行在 {sess.node}:{sess.port}")
        sess.tunnel = Tunnel(cl, sess.node, sess.port)
        lp = sess.tunnel.start()
        log(f"  ✅ 隧道 127.0.0.1:{lp} → {sess.node}:{sess.port}")
        sess.state = "running"
    else:
        log()
        log("=" * 66)
        log("步骤 3  提交计算节点常驻服务")
        log("=" * 66)
        try:
            sess.start(progress=lambda m: log(f"  · {m}"))
        except Exception as e:
            log(f"❌ 启动失败：{e}")
            log(cl.tail_log(sess.jobid or "", "err", 40) if sess.jobid else "")
            return 1

    # ── 4. 健康检查 ──
    log()
    log("=" * 66)
    log("步骤 4  健康检查（经隧道）")
    log("=" * 66)
    t0 = time.time()
    info = None
    while time.time() - t0 < 900:
        try:
            info = sess.health()
            if info.get("ready"):
                break
            if info.get("error"):
                log(f"❌ 模型加载失败：{info['error']}")
                return 1
        except Exception as e:
            pass
        time.sleep(8)
    else:
        log("❌ 等待模型就绪超时")
        return 1

    log(f"  节点      {info.get('hostname')}")
    log(f"  模型      {info.get('model')}   设备 {info.get('device')}")
    log(f"  加载耗时  {info.get('load_sec')}s")
    log(f"  行为      {info.get('behaviors')}")

    # ── 5. 真实推理 ──
    log()
    log("=" * 66)
    log("步骤 5  真实推理（用已有测试视频的帧）")
    log("=" * 66)
    frames_dir = PROJECT / "data" / "frames" / "litter_pos_synthetic"
    files = sorted(frames_dir.glob("f_*.jpg"))[:4]
    if not files:
        log(f"❌ 找不到帧：{frames_dir}。先跑 python -m civility.cli extract")
        return 1

    b64 = [base64.b64encode(p.read_bytes()).decode() for p in files]
    labels = [f"t={i}.0s" for i in range(len(files))]
    log(f"  发送 {len(files)} 帧（{sum(p.stat().st_size for p in files) / 1024:.0f} KB）…")

    t1 = time.time()
    try:
        res = sess.infer(b64, labels)
    except Exception as e:
        log(f"❌ 推理调用失败：{type(e).__name__}: {e}")
        return 1
    dt = time.time() - t1

    log(f"  往返耗时  {dt:.2f}s（模型侧 {res.get('latency_sec')}s）")
    log(f"  模型      {res.get('model')} @ {res.get('hostname')}")
    if res.get("error"):
        log(f"  ⚠️ {res['error']}")
        log(f"  原始输出：{(res.get('raw') or '')[:600]}")
    parsed = res.get("parsed")
    if parsed:
        log(f"  场景      {parsed.get('scene')}")
        for bid, v in parsed["violations"].items():
            mark = "🔴 有" if v["violation"] else "⚪ 无"
            log(f"    {mark}  {bid:16s} conf={v['confidence']:.2f}  {v['observable'][:90]}")

    # ── 6. 结论 ──
    log()
    log("=" * 66)
    log("结论")
    log("=" * 66)
    if parsed:
        log("🎉 全链路打通：本机 → 两级隧道 → 计算节点模型 → 结构化结果")
        log("   网页现在可以实时交互（每窗口只花推理时间本身）。")
    else:
        log("⚠️ 链路通但输出不合 Schema —— 需要看原始输出调 Prompt。")

    if args.keep:
        log()
        log(f"服务保留在 {sess.node}:{sess.port}，本机隧道 {sess.base_url()}")
        log("现在可以启动网页：python -m webui.server")
        log("⚠️ 本进程退出会让隧道消失。要长期用请在网页上点「启动服务」。")
    else:
        log()
        log("清理：停止服务并释放计算节点…")
        sess.stop()
        log("✅ 已释放")
    return 0 if parsed else 2


if __name__ == "__main__":
    sys.exit(main())
