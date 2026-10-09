#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""看一眼本机有哪些网页服务在跑、各是什么状态（多实例排查用）。

为什么要这个工具：
    被"旧进程占着 8848、新服务自动换到 8849"坑过不止一次 ——
    改了代码、页面却没变，因为浏览器连的是另一个实例。
    而且有的实例是用户自己启的（非受限令牌、能用摄像头），
    绝不能随手杀掉。所以先把"谁在跑、什么状态"列清楚再决定。
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PORTS = [8848, 8849, 8850, 8851]


def probe(port: int) -> None:
    base = f"http://127.0.0.1:{port}"
    try:
        with urllib.request.urlopen(base + "/api/state", timeout=8) as r:
            d = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        print(f"[{port}] 不通（{type(e).__name__}）")
        return
    svc = (d.get("service") or {})
    live = (d.get("live") or {})
    print(f"[{port}] ✅ 在跑")
    print(f"       推理服务：{svc.get('state')}  作业={svc.get('jobid')} "
          f"节点={svc.get('node')}")
    print(f"       任务数：{len(d.get('jobs') or [])}   "
          f"实时步长：{live.get('window_stride_sec')}   门控：{live.get('motion_gate')}")
    print(f"       行为：{[b.get('id') for b in (d.get('behaviors') or {}).get('enabled') or []]}")
    try:
        with urllib.request.urlopen(base + "/api/live/cameras", timeout=90) as r:
            c = json.loads(r.read().decode("utf-8"))
        print(f"       摄像头：{c.get('cameras')}  受限令牌={c.get('restricted')}")
    except Exception as e:
        print(f"       摄像头接口异常：{type(e).__name__}: {e}")


def main() -> int:
    lp = ROOT / "webui" / "last_port.txt"
    if lp.exists():
        print(f"（last_port.txt 记录的是 {lp.read_text(encoding='utf-8').strip()}）")
    for p in PORTS:
        probe(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
