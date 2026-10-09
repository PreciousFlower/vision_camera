#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量大小 → (端到端延迟, GPU 成本) 曲线 —— 实时性的取舍用真实数据说话。

为什么要专门量这个：
    用户反馈"点击开始检测后 15 秒才出第一个结果，之后动作与检出也始终差 15 秒"。
    读代码能看出滞后来自本地"凑够 N 个窗口才发"（8 窗 × 2 秒步长 = 16 秒），
    但**该凑几个**才是"快得起来又不烧卡"的最优点，只能实测。

    关键洞察（本工具要验证的）：
        lag  ≈ 凑批时间 = 批内窗口数 × 窗口步长
        成本 ≈ 每窗均摊 GPU 秒 = 整批耗时 / 批内窗口数
    这两个量**由同一个旋钮（批量）决定，方向相反** ——
    所以不存在"又快又便宜"，只能找到拐点。

⚠️ 为什么不能直接 POST 到 webui 的端口（踩过）：
    webui 上**没有** /infer 路由（它只有 /api/*）。第一版这么写，
    十六次请求全是 `404 Not Found`，白跑一轮。
    真正的 /infer 在**计算节点**上，只能经两级隧道到达。
    所以本工具自己建一条隧道（和 webui 用的同一套 Tunnel 代码），
    节点名从 webui 的 /api/state 读 —— 复用已经起来的那份服务，
    不重新提交作业（重新起一次要多花约 2 分钟和一次模型加载）。

用法（需要推理服务已在运行）：
    python tools/bench_batch_lag.py                        # 用 8848 上那个服务
    python tools/bench_batch_lag.py --sizes 1,2,3,4,8 --rounds 4

⚠️ 会真的占用 GPU（每个配置 rounds 轮），跑完请记得停服务。
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
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


def collect_frames(frames_root: Path, need: int) -> list[Path]:
    """从真实任务目录里凑出 need 张帧（优先用最近跑过的任务）。"""
    dirs = sorted([p for p in frames_root.iterdir() if p.is_dir()],
                  key=lambda p: p.stat().st_mtime, reverse=True)
    picked: list[Path] = []
    for d in dirs:
        fs = sorted(d.glob("f_*.jpg"))
        if len(fs) >= 4:
            picked.extend(fs[:max(4, need)])
        if len(picked) >= need:
            break
    if not picked:
        raise SystemExit(f"❌ {frames_root} 里找不到任何 f_*.jpg 素材")
    # 帧不够就重复用（对延迟测量无影响：token 数一样）
    while len(picked) < need:
        picked.extend(picked[:max(1, need - len(picked))])
    return picked[:need]


def open_tunnel(args, log_fn=log):
    """建一条到计算节点的隧道，返回 (base_url, tunnel, svc)。"""
    import requests

    sys.path.insert(0, str(PROJECT))
    from webui.cluster_client import ClusterClient, Tunnel      # noqa: E402

    B = args.base.rstrip("/")
    node = args.node
    svc: dict = {}
    try:
        svc = (requests.get(f"{B}/api/state", timeout=20).json()
               .get("service") or {})
    except Exception as e:
        if not node:
            log_fn(f"❌ 读不到 webui 状态（{B}）：{type(e).__name__}: {e}")
            log_fn("   先起 webui：cd phase1 && python -m webui.server --port 8848")
            log_fn("   （或者用 --node 直接指定计算节点，跳过 webui）")
            return None, None, svc
        log_fn(f"⚠️ 读不到 webui（{type(e).__name__}），"
               f"按 --node {node} 直接建隧道")
    if not node:
        node = str(svc.get("node") or "")
    # ⚠️ 只在"依赖 webui 状态"时才检查 ready：用 --node 指定节点时，
    #    webui 可能根本没在跑，拿不到 state —— 那就别拦。
    if not args.node and svc.get("state") not in (None, "ready"):
        log_fn(f"❌ 服务未就绪（state={svc.get('state')}），"
               f"先在网页「服务」页启动")
        return None, None, svc
    if not node:
        log_fn("❌ 拿不到计算节点名，请用 --node 指定")
        return None, None, svc

    log_fn(f"建立隧道 → {node}:{args.service_port} …")
    cli = ClusterClient(workdir=args.workdir, python=args.remote_python,
                        model_root=args.model_root)
    tun = Tunnel(cli, node, args.service_port)
    try:
        tun.start(timeout=40)
    except Exception as e:
        log_fn(f"❌ 隧道建立失败：{type(e).__name__}: {e}")
        return None, None, svc
    base = tun.base_url()
    log_fn(f"隧道就绪 {base} → {node}:{args.service_port}")
    try:
        h = requests.get(f"{base}/health", timeout=20).json()
        log_fn(f"/health: ready={h.get('ready')} model={h.get('model')} "
               f"device={h.get('device')}")
    except Exception as e:
        log_fn(f"⚠️ /health 失败：{type(e).__name__}: {e}")
        tun.stop()
        return None, None, svc
    return base, tun, svc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8848",
                    help="webui 地址（只用来读状态、拿计算节点名）")
    ap.add_argument("--node", default="", help="直接指定计算节点，跳过 /api/state")
    ap.add_argument("--service-port", type=int, default=18080)
    ap.add_argument("--sizes", default="1,2,4,8", help="要测的批量（逗号分隔）")
    ap.add_argument("--rounds", type=int, default=3, help="每个配置跑几轮（取中位数）")
    ap.add_argument("--delay", type=float, default=0.5, help="两轮之间的间隔秒数")
    ap.add_argument("--frames-per-window", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--stride", type=float, default=2.0,
                    help="窗口步长（只用于把结果换算成'用户看到的滞后'）")
    ap.add_argument("--frames-root", default=str(PROJECT / "data" / "frames"))
    ap.add_argument("--workdir", default="/share/home/u00009/apps/civility-kit")
    ap.add_argument("--remote-python",
                    default="/share/home/u00009/apps/conda/envs/vlm/bin/python")
    ap.add_argument("--model-root", default="/ssdfs/datahome/u00009/models")
    ap.add_argument("--with-actions", action="store_true", default=True)
    ap.add_argument("--no-actions", dest="with_actions", action="store_false")
    args = ap.parse_args()

    import requests

    base, tun, svc = open_tunnel(args)
    if base is None:
        return 1
    log("=" * 78)
    log(f"推理服务：{svc.get('state')} · {svc.get('message')}")
    log(f"模型 {svc.get('model')} · 节点 {svc.get('node')} · 作业 {svc.get('jobid')}")
    log("=" * 78)

    try:
        sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
        nfpw = args.frames_per_window
        frames = collect_frames(Path(args.frames_root), max(sizes) * nfpw)
        log(f"素材：{len(frames)} 张真实帧（如 {frames[0].parent.name}）")
        b64s = [base64.b64encode(p.read_bytes()).decode("ascii") for p in frames]

        def one_window(k: int) -> dict:
            """第 k 个窗口：4 张帧 + 时间标签（与 webui 真跑时一致）。"""
            fb = [b64s[(k * nfpw + i) % len(b64s)] for i in range(nfpw)]
            payload = {
                "frames": [{"b64": b} for b in fb],
                "labels": [f"t={i * args.stride:.1f}s" for i in range(nfpw)],
                "max_new_tokens": args.max_new_tokens,
                "few_shot": False,
                "with_alert": True,
                "with_actions": args.with_actions,
            }
            t_sub = time.time()
            try:
                r = requests.post(f"{base}/infer", json=payload, timeout=600)
                if r.status_code != 200:
                    return {"ok": False, "t_sub": t_sub, "t_done": time.time(),
                            "err": f"HTTP {r.status_code}: {r.text[:200]}"}
                d = r.json()
            except Exception as e:                          # noqa: BLE001
                return {"ok": False, "t_sub": t_sub, "t_done": time.time(),
                        "err": f"{type(e).__name__}: {e}"}
            return {"ok": True, "t_sub": t_sub, "t_done": time.time(),
                    "latency": d.get("latency_sec"), "batch": d.get("batch"),
                    "tokens_out": d.get("tokens_out"),
                    "tokens_prompt": d.get("tokens_prompt"),
                    "tok_per_sec": d.get("tok_per_sec")}

        results: list[dict] = []
        for n in sizes:
            log("")
            log(f"── 批量 {n} ─────────────────────────────────────────────")
            rounds: list[dict] = []
            for rd in range(args.rounds):
                t0 = time.time()
                with cf.ThreadPoolExecutor(max_workers=n) as ex:
                    rs = [f.result() for f in
                          [ex.submit(one_window, k) for k in range(n)]]
                t1 = time.time()
                good = [r for r in rs if r["ok"]]
                if not good:
                    log(f"  第 {rd + 1} 轮 全部失败：{rs[0].get('err')}")
                    continue
                round_sec = t1 - t0
                # 一轮耗时 = 从"并发发出"到"最后一个返回"。
                # 页面是逐条铺开的，所以这既是末条的等待，也是整批的节拍。
                rounds.append({
                    "round_sec": round_sec,
                    "per_win": round_sec / len(good),
                    "n_ok": len(good),
                    "server_batch": statistics.median(
                        [r["batch"] for r in good if r.get("batch")] or [0]),
                    "tokens_out": statistics.median(
                        [r["tokens_out"] for r in good if r.get("tokens_out")]
                        or [0]),
                    "tok_per_sec": statistics.median(
                        [r["tok_per_sec"] for r in good if r.get("tok_per_sec")]
                        or [0]),
                })
                last = rounds[-1]
                log(f"  第 {rd + 1} 轮：整批 {round_sec:5.2f}s · "
                    f"均摊 {last['per_win']:4.2f}s/窗 · 服务端批 "
                    f"{last['server_batch']:.0f} · 输出 {last['tokens_out']:.0f} tok"
                    f" · {last['tok_per_sec']:.0f} tok/s")
                time.sleep(args.delay)
            if not rounds:
                log("  （全部失败，跳过）")
                continue
            med_round = statistics.median([r["round_sec"] for r in rounds])
            med_per = statistics.median([r["per_win"] for r in rounds])
            results.append({
                "n": n,
                "round_sec": med_round,
                "per_win": med_per,
                "obs_batch": statistics.median([r["server_batch"] for r in rounds]),
                "tokens_out": statistics.median([r["tokens_out"] for r in rounds]),
                "tok_per_sec": statistics.median([r["tok_per_sec"] for r in rounds]),
                # 用户看到的滞后：凑批期间，最新那窗已经过去了 (n-1)×步长，
                # 再等推理跑完（med_round）才出结论。
                "lag_newest": (n - 1) * args.stride + med_round,
                "lag_oldest": n * args.stride + med_round,
            })

        log("")
        log("=" * 78)
        log("结果（批量 → 延迟 & 成本）")
        log("=" * 78)
        if not results:
            log("没有任何成功的数据")
            return 1
        ref = next((r for r in results if r["n"] == max(sizes)), results[-1])
        log(f"{'批量':>4} {'整批耗时':>9} {'均摊/窗':>8} {'服务端批':>8} "
            f"{'聚合tok/s':>10} {'新窗滞后':>9} {'旧窗滞后':>9} {'相对成本':>8}")
        log("-" * 78)
        for r in results:
            cost = r["per_win"] / ref["per_win"] if ref["per_win"] else 0
            log(f"{r['n']:>4} {r['round_sec']:>8.2f}s {r['per_win']:>7.2f}s "
                f"{r['obs_batch']:>8.0f} {r['tok_per_sec']:>10.0f} "
                f"{r['lag_newest']:>8.1f}s {r['lag_oldest']:>8.1f}s {cost:>7.1f}×")
        log("-" * 78)
        log(f"（步长按 {args.stride:.1f}s 折算；'新窗滞后'=同批最新那窗从采集到出结论，")
        log("  '旧窗滞后'=同批最旧那窗；页面逐条铺开时你看到的就是这两者之间）")
        log(f"相对成本以批量 {ref['n']} 为 1.0"
            f"（每窗均摊 GPU 秒的比值 ≈ 卡时费用比值）")

        out = PROJECT / "runs" / "bench_batch_lag.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "when": time.strftime("%Y-%m-%d %H:%M:%S"),
            "service": {k: svc.get(k) for k in ("model", "node", "jobid")},
            "stride": args.stride, "rounds": args.rounds, "rows": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"\n已存：{out}")
        return 0
    finally:
        # 隧道是自己建的，必须自己收掉（与 webui 那条独立，互不影响）
        try:
            tun.stop()
            log("（隧道已关闭）")
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
