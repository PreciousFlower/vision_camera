#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定位：预览帧在磁盘上变了，但 /preview 接口返回的还是同一张？

分开验证两段链路：
    磁盘：任务帧目录里 preview_*.jpg 是否在持续新增、内容是否在变
    HTTP：连续请求 /api/job/{id}/preview，返回内容是否跟着变
只要两边结论不一致，问题就锁定在中间那一段（路由/缓存/取最新文件的逻辑）。
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://127.0.0.1:8848"
FRAMES = ROOT / "data" / "frames"


def post(p: str, b: dict | None = None) -> dict:
    r = urllib.request.Request(BASE + p, data=json.dumps(b or {}).encode(),
                               headers={"Content-Type": "application/json"},
                               method="POST")
    return json.loads(urllib.request.urlopen(r, timeout=60).read())


def blob(p: str) -> bytes:
    return urllib.request.urlopen(BASE + p, timeout=15).read()


def main() -> int:
    j = post("/api/live/start", {"source": "rtsp://127.0.0.1:8554/live",
                                 "continuous": True, "with_actions": True,
                                 "use_mock": True, "mock_scene": "littering"})
    jid = j["job_id"]
    d = FRAMES / jid
    print(f"任务 {jid}　帧目录 {d}")
    time.sleep(4)                      # 等采集线程写出预览

    print("\n--- 同时采样：磁盘最新预览 + HTTP 返回 + 任务是否还在产出窗口 ---")
    disk: list[str] = []
    http: list[str] = []
    for i in range(10):
        pv = sorted(d.glob("preview_*.jpg"))
        if pv:
            h = hashlib.md5(pv[-1].read_bytes()).hexdigest()[:8]
            disk.append(h)
            name = pv[-1].name
        else:
            h, name = "-", "（无）"
        raw = blob(f"/api/job/{jid}/preview")
        http.append(hashlib.md5(raw).hexdigest()[:8])
        st = json.loads(urllib.request.urlopen(
            f"{BASE}/api/job/{jid}", timeout=15).read())
        print(f"   [{i}] 磁盘最新 {name} {h} | HTTP {http[-1]} | "
              f"状态={st.get('state')} 窗口={st.get('windows_seen')} "
              f"当前消息={str(st.get('message'))[:26]}")
        time.sleep(0.5)

    print("\n--- 对照 ---")
    print(f"磁盘不同内容 {len(set(disk))}/{len(disk)}　"
          f"HTTP 不同内容 {len(set(http))}/{len(http)}")
    if len(set(disk)) > 1 and len(set(http)) == 1:
        print("❌ 磁盘在变、接口不变 → 问题在路由（取错文件 / 被缓存）")
    elif len(set(disk)) == 1:
        print("❌ 磁盘就没变 → 采集线程没写预览（回到 capture.py 查）")
    else:
        print("✅ 两边都在变 —— 接口正常")

    # 直接看接口到底取了哪个文件：对比它和磁盘最新文件的哈希
    raw = blob(f"/api/job/{jid}/preview")
    pv = sorted(d.glob("preview_*.jpg"))
    if pv:
        newest = hashlib.md5(pv[-1].read_bytes()).hexdigest()[:8]
        print(f"\n接口返回 md5={hashlib.md5(raw).hexdigest()[:8]} / "
              f"磁盘最新 {pv[-1].name} md5={newest}")
        alt = [p for p in pv
               if hashlib.md5(p.read_bytes()).hexdigest()[:8]
               == hashlib.md5(raw).hexdigest()[:8]]
        print(f"接口返回的是：{alt[0].name if alt else '（不是目录里任何一个文件！）'}")
    post(f"/api/stop/{jid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
