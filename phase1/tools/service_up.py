#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""启动集群推理服务并等它就绪（不跑任何任务）。

为什么要单独一个：
    "启动服务 → 等就绪 → 干活 → 释放节点"这四步里，中间两步之间的等待
    最容易出错 —— 手工等容易忘了它还在烧钱，脚本等又不方便中途切活。
    把它拆出来，别的脚本（real_verify / test_batch_pipeline）就能各管一段。

用法：
    python tools/service_up.py                 # 用 last_port.txt 的端口
    python tools/service_up.py --base http://127.0.0.1:8849 --wait 600
    python tools/service_up.py --stop          # 只做"停并释放"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def default_base() -> str:
    try:
        p = int((ROOT / "webui" / "last_port.txt").read_text(encoding="utf-8").strip())
        return f"http://127.0.0.1:{p}"
    except Exception:
        return "http://127.0.0.1:8848"


def post(base: str, path: str, body: dict | None = None, timeout: int = 300) -> dict:
    r = urllib.request.Request(base + path, data=json.dumps(body or {}).encode(),
                               headers={"Content-Type": "application/json"},
                               method="POST")
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get(base: str, path: str, timeout: int = 60):
    with urllib.request.urlopen(base + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--wait", type=int, default=600)
    ap.add_argument("--stop", action="store_true", help="只停服务并释放节点")
    args = ap.parse_args()
    base = args.base or default_base()
    print(f"目标 {base}")

    if args.stop:
        r = post(base, "/api/service/stop", {})
        print(f"已停止：{r.get('message')}")
        return 0

    st = (get(base, "/api/state").get("service") or {}).get("state")
    if st == "ready":
        print("服务已经就绪，无需启动")
        return 0

    # 先清掉遗留作业，避免"两个作业同时占卡、双倍计费"
    try:
        swept = post(base, "/api/service/sweep")
        if swept.get("cancelled"):
            print(f"先清掉遗留作业：{swept['cancelled']}")
    except Exception:
        pass

    r = post(base, "/api/service/start", {"model": args.model})
    print(f"启动：{r.get('message')}")
    t0 = time.time()
    while time.time() - t0 < args.wait:
        time.sleep(5)
        s = (get(base, "/api/state").get("service") or {})
        state = s.get("state")
        if state == "ready":
            print(f"✅ 就绪（{time.time() - t0:.0f}s）节点 {s.get('node')} "
                  f"作业 {s.get('jobid')}")
            print("⚠️ 用完记得释放：python tools/service_up.py --stop")
            return 0
        if state == "error":
            print(f"❌ 启动失败：{s.get('message')}")
            return 1
        if int(time.time() - t0) % 30 < 6:
            print(f"   …等待 {time.time() - t0:.0f}s（{state}）")
    print(f"❌ 等就绪超时（{args.wait}s），当前 {st}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
