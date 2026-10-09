#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""花最少的钱，看**真模型**在真实（RTSP）流上的识别效果。

为什么需要它：
    演示模式（本地假模型）永远看不出真实能力，而"点一下启动服务、跑一会儿、
    再记得关掉"这套流程又容易忘 —— 忘了就是在烧钱（实测一个没人管的作业
    跑满 12 小时约 48 元）。所以这个脚本把整件事做成**一次原子操作**：
        启动服务 → 等就绪 → 跑 N 个窗口 → 停止监测 → **释放节点** → 报花费
    中途出错也会走 finally 释放节点，不会留下孤儿作业。

用法：
    python tools/real_verify.py                       # 3 个窗口，步长 8 秒
    python tools/real_verify.py --windows 5 --stride 10
    python tools/real_verify.py --source data/videos/littering_pos_vis.mp4

注意：需要 VPN。跑完会自动释放，花费会打印出来（按 settings 里的单价估算）。
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def post(base: str, path: str, body: dict | None = None, timeout: int = 120) -> dict:
    r = urllib.request.Request(base + path, data=json.dumps(body or {}).encode(),
                               headers={"Content-Type": "application/json"},
                               method="POST")
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get(base: str, path: str, timeout: int = 60):
    with urllib.request.urlopen(base + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def default_base() -> str:
    """默认连哪个端口：读 webui/last_port.txt，而不是写死 8848。

    ⚠️ 写死 8848 踩过大坑：8848 常常是**用户自己启的那个实例**
    （非受限令牌、能用摄像头），而带沙箱启的实例会自动顺延到 8849。
    写死端口会让这个脚本连到用户的实例上，甚至在人家的服务上做
    start/stop —— 实测就这么误停过一次用户的服务。
    """
    lp = HERE.parent / "webui" / "last_port.txt"
    try:
        return f"http://127.0.0.1:{int(lp.read_text(encoding='utf-8').strip())}"
    except Exception:
        return "http://127.0.0.1:8848"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None,
                    help="网页服务地址，默认读 webui/last_port.txt")
    ap.add_argument("--source", default="rtsp://127.0.0.1:8554/live")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--stride", type=float, default=8.0)
    ap.add_argument("--ready-timeout", type=int, default=600)
    args = ap.parse_args()
    if not args.base:
        args.base = default_base()
    print(f"（目标服务：{args.base}）")

    st0 = get(args.base, "/api/state")
    # ⚠️ 窗口步长在 live 段，不在 extract 段（一开始读错了地方，
    #    导致"跑完还原步长"这步静默失效 —— 用户原来的设置被改掉却没人知道）
    old_stride = ((st0.get("live") or {}).get("window_stride_sec")
                  or (st0.get("service_cfg") or {}).get("window_stride_sec"))
    print("=" * 74)
    print("真模型验证（跑完自动释放节点，不留孤儿作业）")
    print(f"源：{args.source}　模型：{args.model}　"
          f"窗口：{args.windows} 个 × {args.stride:.0f} 秒步长")
    print("=" * 74)

    released = False
    try:
        # ── 0. 清掉任何遗留作业，避免"两个服务同时在跑、双倍计费" ──
        try:
            swept = post(args.base, "/api/service/sweep")
            if swept.get("cancelled"):
                print(f"[0] 先清掉遗留作业：{swept['cancelled']}")
        except Exception:
            pass

        # ── 1. 调整步长（跑得越快越省钱）──
        post(args.base, "/api/live/config", {"window_stride_sec": args.stride})
        print(f"[1] 窗口步长设为 {args.stride:.0f} 秒")

        # ── 2. 启动推理服务 ──
        r = post(args.base, "/api/service/start", {"model": args.model})
        print(f"[2] 启动服务：{r.get('message')}")
        t0 = time.time()
        node = None
        while time.time() - t0 < args.ready_timeout:
            time.sleep(5)
            s = (get(args.base, "/api/state").get("service") or {})
            if s.get("state") == "ready":
                node = s.get("node")
                print(f"    ✅ 就绪（{time.time() - t0:.0f}s）节点 {node} "
                      f"作业 {s.get('jobid')}")
                break
            if s.get("state") == "error":
                print(f"    ❌ 启动失败：{s.get('message')}")
                return 1
            if int(time.time() - t0) % 30 < 5:
                print(f"    …等待中 {time.time() - t0:.0f}s（{s.get('state')}）")
        else:
            print("    ❌ 等就绪超时")
            return 1

        # ── 3. 跑 N 个窗口（真模型）──
        print(f"\n[3] 开始真模型监测")
        j = post(args.base, "/api/live/start", {
            "source": args.source, "continuous": True,
            "with_actions": True, "use_mock": False})
        jid = j["job_id"]
        seen: list[dict] = []

        def reader() -> None:
            try:
                with urllib.request.urlopen(
                        f"{args.base}/api/job/{jid}/stream", timeout=900) as resp:
                    for raw in resp:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            continue
                        try:
                            ev = json.loads(line[5:].strip())
                        except json.JSONDecodeError:
                            continue
                        if ev.get("type") == "window":
                            seen.append(ev)
                            hits = ev.get("hits") or []
                            print(f"    · 窗口 {ev.get('cumulative')} "
                                  f"{ev.get('t_start')}-{ev.get('t_end')}s "
                                  f"耗时 {ev.get('latency')}s")
                            print(f"      场景：{str(ev.get('scene'))[:70]}")
                            for p in (ev.get("people") or [])[:2]:
                                print(f"      人：{p.get('where')} | "
                                      f"{str(p.get('action'))[:60]}")
                            for h in hits:
                                print(f"      ⚠️ 命中 {h.get('name')} "
                                      f"{float(h.get('confidence') or 0) * 100:.0f}%"
                                      f" | {str(h.get('observable'))[:60]}")
                            if not hits:
                                print("      （本窗口无命中）")
            except Exception:
                pass

        th = threading.Thread(target=reader, daemon=True)
        th.start()
        deadline = time.time() + args.windows * (args.stride + 60) + 120
        while len(seen) < args.windows and time.time() < deadline:
            time.sleep(1.0)
        print(f"\n    收到 {len(seen)} 个窗口结果")

        # ── 4. 收尾：停监测（会自动释放节点）──
        print("\n[4] 停止监测并释放节点")
        post(args.base, f"/api/stop/{jid}")
        time.sleep(2)
        stopr = post(args.base, "/api/service/stop")
        released = True
        print(f"    {stopr.get('message')}")

        d = get(args.base, f"/api/job/{jid}")
        alerts = d.get("alerts") or []
        cost = get(args.base, "/api/service/cost")
        print("\n" + "=" * 74)
        print(f"结果：{len(d.get('window_results') or [])} 个窗口、{len(alerts)} 条告警")
        for a in alerts[:5]:
            print(f"  ⚠️ {a.get('behavior_id')} t={a.get('t_start')}s "
                  f"| {(a.get('reason') or '')[:60]}")
        print(f"本次服务运行 {cost.get('uptime_text')}，"
              f"约 {cost.get('est_cost')} 元（{cost.get('price_per_hour')} 元/卡时）")
        return 0

    except Exception as e:
        print(f"\n❌ 出错：{type(e).__name__}: {e}")
        return 1
    finally:
        # 无论如何都要把节点退掉 —— 这是本脚本存在的意义
        if not released:
            try:
                post(args.base, "/api/live/stop", {})
            except Exception:
                pass
            try:
                r = post(args.base, "/api/service/stop")
                print(f"\n[收尾] 已释放节点：{r.get('message')}")
            except Exception as e:
                print(f"\n[收尾] ⚠️ 释放节点失败，请手工检查："
                      f"python tools/cluster_cost.py --stop（{e}）")
        if old_stride is not None:
            try:
                post(args.base, "/api/live/config",
                     {"window_stride_sec": old_stride})
                print(f"[收尾] 窗口步长已还原为 {old_stride}")
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
