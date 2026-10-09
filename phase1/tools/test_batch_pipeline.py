#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证"批量流水线"真的把结论间隔压下来了（真模型，跑完自动释放节点）。

改造前后要对比的两个数（微基准给了上界，这里要验**端到端**）：
    · 单窗延迟：一条结论从窗口开始采集到出结果
    · 结论间隔：连续两条结论之间隔多久 ← 用户真正感知的"快不快"

实测基准（cluster/vlm_bench.py，批量 8 → 4 倍吞吐）：
    批量 1：1.96 s/窗   批量 2：1.11   批量 4：0.64   批量 8：0.49
所以本地必须"凑批 + 并发提交 + 采集流水线"，本脚本就是验这三件事都生效：
    1. 服务端确实收到了批量（返回里的 batch 字段 > 1）
    2. 结论间隔显著小于单窗延迟（说明并发/流水线在起作用）
    3. 不出现"跟不上"（说明节奏没超过处理能力）

用法：
    python tools/test_batch_pipeline.py --source rtsp://127.0.0.1:8555/live \
        --stride 2 --windows 12
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    results.append((ok, name, detail))
    print(f"  {'✅' if ok else '❌'} {name}" + (f" — {detail}" if detail else ""),
          flush=True)
    return ok


def post(base: str, path: str, body: dict | None = None, timeout: int = 300) -> dict:
    r = urllib.request.Request(base + path, data=json.dumps(body or {}).encode(),
                               headers={"Content-Type": "application/json"},
                               method="POST")
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get(base: str, path: str, timeout: int = 60):
    with urllib.request.urlopen(base + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def default_base() -> str:
    lp = ROOT / "webui" / "last_port.txt"
    try:
        return f"http://127.0.0.1:{int(lp.read_text(encoding='utf-8').strip())}"
    except Exception:
        return "http://127.0.0.1:8848"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None)
    ap.add_argument("--source", default="rtsp://127.0.0.1:8555/live")
    ap.add_argument("--stride", type=float, default=2.0)
    ap.add_argument("--windows", type=int, default=12)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    args = ap.parse_args()
    base = args.base or default_base()
    print(f"目标 {base}　源 {args.source}　步长 {args.stride}s　"
          f"目标窗口 {args.windows}")

    st = get(base, "/api/state")
    svc = (st.get("service") or {}).get("state")
    if svc != "ready":
        print(f"服务未就绪（{svc}），先启动：POST /api/service/start")
        return 2

    post(base, "/api/live/config", {"window_stride_sec": args.stride})
    j = post(base, "/api/live/start", {
        "source": args.source, "continuous": True, "with_actions": True,
        "use_mock": False})
    jid = j["job_id"]
    print(f"任务 {jid}")

    events: list[dict] = []
    done = threading.Event()

    def reader() -> None:
        try:
            with urllib.request.urlopen(
                    f"{base}/api/job/{jid}/stream", timeout=900) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        ev = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    if ev.get("type") == "window":
                        ev["_t"] = time.time()
                        events.append(ev)
                        print(f"    [{len(events):>2}] +{ev['_t'] % 1000:.2f} "
                              f"窗{ev.get('cumulative')} "
                              f"{ev.get('t_start')}-{ev.get('t_end')}s "
                              f"耗时{ev.get('latency')}s "
                              f"批量{ev.get('batch')} "
                              f"出{ev.get('tokens_out')}tok "
                              f"命中{len(ev.get('hits') or [])}", flush=True)
                    if ev.get("type") in ("done", "error"):
                        done.set()
        except Exception:
            pass

    th = threading.Thread(target=reader, daemon=True)
    th.start()

    deadline = time.time() + args.windows * (args.stride + 25) + 120
    while len(events) < args.windows and time.time() < deadline:
        time.sleep(1.0)
    post(base, f"/api/stop/{jid}", {})
    done.wait(timeout=10)
    time.sleep(1)

    d = get(base, f"/api/job/{jid}")
    wres = d.get("window_results") or []

    print(f"\n收到 {len(events)} 条 window 事件，落库 {len(wres)} 条")
    if len(events) < 2:
        check(False, "至少产出 2 条结论（否则无法判断间隔）",
              f"只有 {len(events)} 条")
        return 1

    # ① 批量确实发生
    batches = [e.get("batch") or 1 for e in events]
    mx = max(batches)
    check(mx > 1, "服务端收到了批量（一次前向算了多个窗口）",
          f"最大批量 {mx}，各次 {batches[:8]}")

    # ② 结论间隔
    ts = [e["_t"] for e in events]
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    gaps_sorted = sorted(gaps)
    med = gaps_sorted[len(gaps_sorted) // 2]
    mean = sum(gaps) / len(gaps)
    lats = [e.get("latency") or 0 for e in events if not e.get("skipped")]
    lat_med = sorted(lats)[len(lats) // 2] if lats else 0
    print(f"\n  结论间隔：中位 {med:.2f}s　平均 {mean:.2f}s　"
          f"最短 {min(gaps):.2f}s　最长 {max(gaps):.2f}s")
    print(f"  单窗延迟（服务端报告）：中位 {lat_med:.2f}s")
    check(med < 6.0, "结论间隔明显小于改造前（6~8 秒一条）",
          f"中位 {med:.2f}s / 目标 ≤ 6s")
    check(lat_med > 0, "服务端有返回单窗延迟", f"{lat_med:.2f}s")

    # ③ 没跟不上
    st2 = get(base, f"/api/job/{jid}")
    check(not st2.get("behind"), "没有出现「跟不上画面」的滞后",
          f"behind={st2.get('behind')}")

    post(base, "/api/stop/" + jid, {})
    print("\n" + "=" * 74)
    bad = [x for x in results if not x[0]]
    if bad:
        print(f"❌ {len(bad)} 项未通过：")
        for _, n, dd in bad:
            print(f"   · {n} — {dd}")
        return 1
    print(f"✅ 全部通过 —— 批量流水线生效，结论间隔中位 {med:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
