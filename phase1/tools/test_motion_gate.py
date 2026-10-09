#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证"动作量门控"：画面没动就不调模型（省钱也省等待），有动就必须调。

背景：实测每个窗口推理 10~12 秒，而空走廊/无人经过的窗口往往占大多数。
      这些窗口送模型纯属浪费。门控用"相邻帧灰度差的最大值"当判据。

标定（本机实测，见日志）：
    移动素材（有人物移动）动作量 ≈ 4.6 ~ 5.1
    静止素材（同一张图循环）动作量 ≈ 0.00 ~ 0.13
    阈值 1.5 落在 30 倍的安全间隔里。

必须**两个方向**都验，否则等于没验：
    A. 静止源 → 应当跳过（模型不被调用）
    B. 移动源 → 不许跳过（否则就是漏报的来源）
    C. 上一窗刚命中违规时 → 即使画面静止也不跳过（保险丝）

用法：
    python tools/test_motion_gate.py            # 需要 8554(移动) 与 8556(静止) 两个演示流
"""
from __future__ import annotations

import argparse
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

MOVING = "rtsp://127.0.0.1:8554/live"
STATIC = "rtsp://127.0.0.1:8556/live"
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


def run_case(port: int, source: str, scene: str, seconds: float) -> dict:
    base = f"http://127.0.0.1:{port}"
    env = dict(os.environ)
    env["CIVILITY_MOCK_LATENCY"] = "1.5"     # 让"没被跳过"这件事看得出来
    log = open(ROOT / "runs" / f"_gate_{port}.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "webui.server", "--port", str(port),
         "--no-auto-port"],
        cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(40):
            time.sleep(0.5)
            try:
                api(base, "/api/state", timeout=5)
                break
            except Exception:
                pass
        # ⚠️ 步长要配合"每批几窗"：攒批大小默认已从 8 改成 2
        #    （见 settings.yaml 的 continuous_batch_windows —— 它同时是实时性的
        #    主旋钮，滞后 ≈ 攒批窗口数 × 窗口步长）。步长 6 秒时首批要采 12 秒
        #    才出结果，测试等太短就只看到"0 个窗口"，看起来像门控坏了，
        #    其实是批次没凑齐（实测踩到过：注释里还写着"48 秒"，是 8 窗时代的
        #    数字，改默认值后没跟着改 —— 注释与实现不一致比没注释更误导人）。
        #    这里用 1 秒步长，一批只覆盖 2 秒，30 秒能覆盖很多批次。
        api(base, "/api/live/config", {"window_stride_sec": 1.0,
                                       "continuous_batch_windows": 2})
        j = api(base, "/api/live/start", {
            "source": source, "continuous": True, "with_actions": True,
            "use_mock": True, "mock_scene": scene})
        jid = j["job_id"]
        time.sleep(seconds)
        d = api(base, f"/api/job/{jid}")
        api(base, f"/api/stop/{jid}", {})
        wres = d.get("window_results") or []
        return {
            "windows": len(wres),
            "skipped": sum(1 for w in wres if w.get("skipped")),
            "called": sum(1 for w in wres if not w.get("skipped")),
            "motions": [w.get("motion") for w in wres if w.get("motion") is not None],
            "latencies": [w.get("latency") for w in wres],
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
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()

    print("=" * 76)
    print("动作量门控验证（静止源应跳过、移动源不许跳过）")
    print("=" * 76)

    print("\n[A] 静止源 + 无违规场景（应当跳过模型调用）")
    a = run_case(8863, STATIC, "normal", args.seconds)
    print(f"    窗口 {a['windows']} 个：跳过 {a['skipped']}，实际调模型 "
          f"{a['called']} 次；动作量样本 {a['motions'][:4]}；"
          f"耗时样本 {a['latencies'][:4]}")
    check(a["windows"] >= 2, "静止源也照常产出窗口（页面不会空白）",
          f"{a['windows']} 个")
    check(a["skipped"] >= 1, "画面没动时跳过模型调用",
          f"{a['skipped']}/{a['windows']} 个窗口被跳过")
    check(a["called"] < a["windows"], "确实省下了调用",
          f"只调了 {a['called']} 次，省 {a['windows'] - a['called']} 次")

    print("\n[B] 移动源（不该被跳过 —— 跳了就是漏报来源）")
    b = run_case(8864, MOVING, "normal", args.seconds)
    print(f"    窗口 {b['windows']} 个：跳过 {b['skipped']}，实际调模型 "
          f"{b['called']} 次；动作量样本 {b['motions'][:4]}")
    check(b["windows"] >= 1, "移动源正常产出窗口", f"{b['windows']} 个")
    check(b["skipped"] == 0, "画面在动时**不跳过**（否则会漏报）",
          f"跳过 {b['skipped']} 个；动作量 {b['motions'][:3]}")

    print("\n[C] 上一窗命中违规时 → 即使画面静止也不跳过（保险丝）")
    c = run_case(8865, STATIC, "littering", args.seconds)
    print(f"    窗口 {c['windows']} 个：跳过 {c['skipped']}，实际调模型 "
          f"{c['called']} 次")
    check(c["called"] >= 2, "刚报过违规就继续看，不省这次调用",
          f"调模型 {c['called']} 次（场景恒为违规，保险丝应一直生效）")

    print("\n" + "=" * 76)
    bad = [x for x in results if not x[0]]
    if bad:
        print(f"❌ {len(bad)} 项未通过：")
        for _, n, d in bad:
            print(f"   · {n} — {d}")
        return 1
    print("✅ 全部通过 —— 静止时省调用、移动时照常送模型、命中后有保险丝")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
