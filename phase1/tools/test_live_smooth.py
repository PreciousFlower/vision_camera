#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证"实时画面卡顿"真的被治好了 —— 而且不花钱（用人为延迟模拟慢推理）。

用户反馈"摄像头实时的很卡顿"。根因是：采集与推理在**同一个线程**里顺序执行，
推理那 10~12 秒没人读摄像头 → 页面上的实时画面整段冻住
（一个周期 17 秒里冻 11 秒，占 65%）。

这个脚本做一次 **A/B 对照**，两边跑完全一样的负载：
    A（修复后）：有专职读帧线程 —— 推理期间画面照走
    B（修复前）：CIVILITY_NO_READER=1 关掉读帧线程，回到老路径
判据用**硬指标**：连续采样预览帧，量"相邻两张不同画面之间的最大间隔"
（= 最长冻结时长）。而不是看代码觉得对。

用法：
    python tools/test_live_smooth.py                  # 自动起两个临时服务做对比
    python tools/test_live_smooth.py --seconds 25
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RTSP = "rtsp://127.0.0.1:8554/live"
results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    results.append((ok, name, detail))
    print(f"  {'✅' if ok else '❌'} {name}" + (f" — {detail}" if detail else ""),
          flush=True)
    return ok


def api(base: str, path: str, body: dict | None = None, timeout: int = 60):
    if body is None:
        with urllib.request.urlopen(base + path, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def measure(port: int, *, mock_latency: float, no_reader: bool,
            seconds: float, stride: float) -> dict:
    """起一个临时服务，跑一段监测，返回冻结时长的统计。"""
    base = f"http://127.0.0.1:{port}"
    env = dict(os.environ)
    env["CIVILITY_MOCK_LATENCY"] = str(mock_latency)
    env.pop("CIVILITY_NO_READER", None)
    if no_reader:
        env["CIVILITY_NO_READER"] = "1"
    log = open(ROOT / "runs" / f"_smooth_{port}.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "webui.server", "--port", str(port),
         "--no-auto-port"],
        cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        # 等端口起来
        for _ in range(40):
            time.sleep(0.5)
            try:
                api(base, "/api/state", timeout=5)
                break
            except Exception:
                pass
        else:
            raise RuntimeError("临时服务起不来，看 runs/_smooth_%d.log" % port)

        api(base, "/api/live/config", {"window_stride_sec": stride})
        j = api(base, "/api/live/start", {
            "source": RTSP, "continuous": True, "with_actions": True,
            "use_mock": True, "mock_scene": "littering"})
        jid = j["job_id"]

        # 采样预览帧，记录"内容变化"的时刻
        changes: list[float] = []
        hashes: list[str] = []
        t_end = time.time() + seconds
        while time.time() < t_end:
            try:
                with urllib.request.urlopen(
                        f"{base}/api/job/{jid}/preview", timeout=8) as r:
                    blob = r.read()
                if blob[:2] == b"\xff\xd8":
                    h = hashlib.md5(blob).hexdigest()[:8]
                    if not hashes or h != hashes[-1]:
                        hashes.append(h)
                        changes.append(time.time())
            except Exception:
                pass
            time.sleep(0.25)

        st = api(base, f"/api/job/{jid}")
        api(base, f"/api/stop/{jid}", {})     # 注意要传 body，否则是 GET → 405
        gaps = [b - a for a, b in zip(changes, changes[1:])]
        return {
            "frames_seen": len(hashes),
            "changes": len(changes),
            "max_gap": max(gaps) if gaps else seconds,
            "median_gap": sorted(gaps)[len(gaps) // 2] if gaps else seconds,
            "windows": len(st.get("window_results") or []),
            "state": st.get("state"),
        }
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        log.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=26.0)
    ap.add_argument("--stride", type=float, default=6.0)
    ap.add_argument("--mock-latency", type=float, default=6.0,
                    help="人为的推理延迟（秒），模拟真模型 10~12 秒的节奏")
    args = ap.parse_args()

    print("=" * 76)
    print("实时画面流畅度 A/B 对照（人为延迟模拟慢推理，不花钱）")
    print(f"负载：窗口步长 {args.stride:.0f}s，每次推理人为延迟 "
          f"{args.mock_latency:.0f}s，采样 {args.seconds:.0f}s")
    print("=" * 76)

    print("\n[B] 修复前：没有读帧线程（CIVILITY_NO_READER=1）")
    b = measure(8861, mock_latency=args.mock_latency, no_reader=True,
                seconds=args.seconds, stride=args.stride)
    print(f"    画面变化 {b['changes']} 次；最长冻结 {b['max_gap']:.2f}s；"
          f"中位间隔 {b['median_gap']:.2f}s；产出窗口 {b['windows']} 个")

    print("\n[A] 修复后：有专职读帧线程")
    a = measure(8862, mock_latency=args.mock_latency, no_reader=False,
                seconds=args.seconds, stride=args.stride)
    print(f"    画面变化 {a['changes']} 次；最长冻结 {a['max_gap']:.2f}s；"
          f"中位间隔 {a['median_gap']:.2f}s；产出窗口 {a['windows']} 个")

    print("\n" + "=" * 76)
    check(a["changes"] > 0, "修复后实时画面确实在更新",
          f"{a['changes']} 次内容变化")
    check(a["max_gap"] < 1.2,
          "推理期间画面**不再冻住**（最长冻结 < 1.2 秒）",
          f"实测最长冻结 {a['max_gap']:.2f}s"
          f"（推理延迟 {args.mock_latency:.0f}s，修复前至少冻这么久）")
    check(a["max_gap"] < b["max_gap"] * 0.5,
          "相比修复前有量级改善",
          f"{b['max_gap']:.2f}s → {a['max_gap']:.2f}s"
          f"（{b['max_gap'] / max(0.01, a['max_gap']):.1f}×）")
    check(a["windows"] >= 1, "推理链路照常产出窗口",
          f"{a['windows']} 个窗口")

    bad = [x for x in results if not x[0]]
    if bad:
        print(f"\n❌ {len(bad)} 项未通过")
        return 1
    print("\n✅ 全部通过 —— 实时画面在推理期间持续更新，卡顿已治")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
