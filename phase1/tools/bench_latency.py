#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""延迟基准 —— 找出"每窗口几秒"到底花在哪，以及哪个旋钮最有效。

为什么必须测而不是猜：
    "模型慢"有三种完全不同的原因，对应的解法互不相同：
      · prompt token 多  → 视觉编码是瓶颈 → 减帧数 / 降分辨率 / 拼图
      · 输出 token 多    → 自回归生成是瓶颈 → 精简输出格式 / 减行为数
      · 两者都不多但很慢 → 框架层问题（无 vLLM、bf16、设备映射）
    猜错方向会白折腾很久，所以这里把每个组合的 token 与耗时都打出来。

测的组合（对真实使用最有指导意义的四组）：
    A 帧数 4 / 行为 3   ← 当前默认，基线
    B 帧数 2 / 行为 3   ← 看视觉编码占比
    C 帧数 4 / 行为 1   ← 看输出 token 占比
    D 帧数 1 / 行为 3   ← 单帧（无法判时序，仅作下界参考）

用法（需要服务已在运行）：
    python tools/bench_latency.py
    python tools/bench_latency.py --repeat 2 --model Qwen/Qwen3-VL-2B-Instruct
"""
from __future__ import annotations

import argparse
import base64
import json
import statistics
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


def log(m: str = "") -> None:
    print(m, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8848")
    ap.add_argument("--frames-dir", default="data/frames/litter_pos_synthetic")
    ap.add_argument("--repeat", type=int, default=2, help="每组重复次数（取中位数）")
    ap.add_argument("--model", default="", help="保留：仅作记录，切换模型需重启服务")
    args = ap.parse_args()

    import requests

    B = args.base.rstrip("/")
    try:
        st = requests.get(f"{B}/api/state", timeout=15).json()
    except Exception as e:
        log(f"❌ 后端没起来：{e}\n   先跑 python -m webui.server")
        return 1
    if st["service"]["state"] != "ready":
        log(f"❌ 推理服务未就绪（{st['service']['state']}）。先在网页上点「启动服务」。")
        return 1

    fdir = PROJECT / args.frames_dir
    files = sorted(fdir.glob("f_*.jpg"))
    if len(files) < 4:
        log(f"❌ 帧不足：{fdir}（先跑 python -m civility.cli extract）")
        return 1
    b64_all = [base64.b64encode(p.read_bytes()).decode() for p in files]

    svc = requests.get(f"{B}/api/service/health", timeout=20).json()
    n_behaviors = len(svc.get("behaviors") or [])
    log("=" * 74)
    log(f"推理服务  {svc.get('model')} @ {svc.get('hostname')} ({svc.get('device')})")
    log(f"启用行为  {svc.get('behaviors')}（{n_behaviors} 项）")
    log("=" * 74)

    # 直连计算节点服务（绕过网页编排，纯粹测模型往返）
    base_url = st["service"]["local_url"]
    log(f"直接测模型端点 {base_url}（绕开网页层层封装）\n")

    def probe(n_frames: int, note: str) -> dict:
        frames = b64_all[:n_frames]
        labels = [f"t={i}.0s" for i in range(n_frames)]
        rows = []
        for i in range(args.repeat):
            t0 = time.time()
            r = requests.post(
                f"{base_url}/infer",
                json={"frames": [{"b64": b} for b in frames], "labels": labels},
                timeout=300,
            )
            dt = time.time() - t0
            if r.status_code != 200:
                log(f"  ⚠️ HTTP {r.status_code}: {r.text[:200]}")
                return {}
            d = r.json()
            rows.append({
                "latency": d.get("latency_sec") or dt,
                "roundtrip": dt,
                "tok_in": d.get("tokens_prompt"),
                "tok_out": d.get("tokens_out"),
                "tps": d.get("tok_per_sec"),
                "error": d.get("error"),
            })
            if i == 0:
                parsed = d.get("parsed") or {}
                first_viol = next(
                    (v for v in (parsed.get("violations") or {}).values()
                     if v.get("violation")), None
                )
                rows[-1]["sample"] = (
                    first_viol.get("observable", "")[:60] if first_viol
                    else (parsed.get("scene") or "")[:60]
                )
        lat = [r["latency"] for r in rows]
        out = {
            "frames": n_frames,
            "latency_med": statistics.median(lat),
            "latency_min": min(lat),
            "latency_max": max(lat),
            "tok_in": rows[0].get("tok_in"),
            "tok_out": rows[0].get("tok_out"),
            "tps": rows[0].get("tps"),
            "note": note,
            "sample": rows[0].get("sample", ""),
            "errors": sum(1 for r in rows if r.get("error")),
        }
        log(f"  {note}")
        log(f"    中位延迟 {out['latency_med']:.2f}s   "
            f"prompt {out['tok_in']} tok   输出 {out['tok_out']} tok   "
            f"{out['tps']} tok/s")
        if out["sample"]:
            log(f"    样例输出：{out['sample']}")
        if out["errors"]:
            log(f"    ⚠️ {out['errors']}/{args.repeat} 次有 schema 错误")
        log()
        return out

    log("── 基线：4 帧 / 全部行为 ──")
    base = probe(4, f"A  4帧/{n_behaviors}行为（当前默认）")

    log("── 降帧数：看视觉编码占多少 ──")
    two = probe(2, f"B  2帧/{n_behaviors}行为")
    one = probe(1, f"D  1帧/{n_behaviors}行为（下界参考，无法判时序）")

    # 减行为需要改配置，这里只估算：输出 token 与行为数近似成正比
    log("── 行为数的影响（按输出 token 线性外推，未实测）──")
    if base.get("tok_out"):
        per_behavior = base["tok_out"] / max(1, n_behaviors)
        log(f"  每个行为约 {per_behavior:.0f} 输出 token")
        for k in (1, 2, 3):
            est = base["latency_med"] * (1 - (1 - k / n_behaviors) * 0.6)
            log(f"  {k} 个行为 ≈ {est:.1f}s / 窗口（粗估）")
        log()

    # ── 结论 ──
    log("=" * 74)
    log("结论")
    log("=" * 74)
    if base and two:
        saved = base["latency_med"] - two["latency_med"]
        ratio = base["tok_in"] / two["tok_in"] if two.get("tok_in") else 0
        log(f"  4帧 → 2帧：延迟 {base['latency_med']:.1f}s → {two['latency_med']:.1f}s"
            f"（省 {saved:.1f}s，prompt token 比 {ratio:.2f}×）")
        if saved / max(0.01, base["latency_med"]) > 0.25:
            log("  ⇒ 视觉编码占了可观比例。想再快，优先减帧数或降分辨率。")
        else:
            log("  ⇒ 减帧数收益有限，瓶颈在自回归生成。")
    if base.get("tps"):
        log(f"  解码速度 {base['tps']} tok/s（A800 + 纯 transformers，无 vLLM）")
        log("  要提速只能：① 精简输出格式（少写文字）② 减少行为数 "
            "③ 换更小模型 ④ 上 vLLM（本集群 glibc 2.28 装不了）")
    log()
    log("  实时性判据：如果窗口步长是 4 秒、而单窗口耗时 > 4 秒，")
    log("  则系统跟不上实时流。此时应增大 window_stride_sec，")
    log("  或降低抽帧频率，让判断周期与推理速度匹配。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
