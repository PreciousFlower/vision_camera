#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断：持续监测跑起来之后，webui 的 HTTP 端点还**响应得过来**吗？

为什么单独查这个：
    tools/test_frame_cache.py 里 GET /api/job/{id} 在监测启动后**超时 30 秒**。
    而那个端点只是读内存里的一个 list，正常应该在毫秒级返回 ——
    超时说明服务端有东西被占住了，这属于"页面会卡住"级别的问题，
    不能当成测试工具的毛病绕过去。

本脚本按顺序问这几个问题，把"哪一步开始不响应"定位出来：
    1. 空载时 /api/state 快不快？
    2. 起一个持续监测（文件源 + 假模型，不碰集群）之后，快不快？
    3. /api/job/{id}（读内存）快不快？它和 /api/state 的差别在哪？
    4. 端点多的时候（模拟页面轮询）会不会雪崩？

不写结论，只打实测数字 —— 这类问题猜是猜不出来的。
"""
from __future__ import annotations

import http.client
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PORT = 8877
TMP = ROOT / "runs" / "_tmp_http_resp_test"


def timed(path: str, method: str = "GET", body: dict | None = None,
          timeout: float = 12.0):
    """返回 (耗时秒, status 或 None, 错误)。"""
    t0 = time.time()
    try:
        c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        hdrs = {"Content-Type": "application/json"} if payload else {}
        c.request(method, path, payload, hdrs)
        r = c.getresponse()
        r.read()
        st = r.status
        c.close()
        return time.time() - t0, st, ""
    except Exception as e:                                  # noqa: BLE001
        return time.time() - t0, None, f"{type(e).__name__}: {e}"


def main() -> int:
    if TMP.exists():
        shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    import yaml

    cfg = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text("utf-8"))
    cfg.setdefault("paths", {})["frames_dir"] = str(TMP / "frames")
    (TMP / "frames").mkdir(parents=True, exist_ok=True)
    sp = TMP / "settings.yaml"
    sp.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                  encoding="utf-8")

    env = dict(os.environ)
    env["CIVILITY_SETTINGS"] = str(sp)
    env["CIVILITY_MOCK_LATENCY"] = "0.05"
    logf = TMP / "server.log"
    fh = open(logf, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen([sys.executable, "-m", "webui.server",
                             "--port", str(PORT), "--no-auto-port"],
                            cwd=str(ROOT), env=env, stdout=fh, stderr=fh)
    try:
        for _ in range(80):
            d, st, _ = timed("/api/state", timeout=2)
            if st == 200:
                break
            time.sleep(0.4)
        print(f"服务已起（{PORT}）\n")

        print("[1] 空载 /api/state")
        for i in range(3):
            d, st, err = timed("/api/state")
            print(f"    #{i+1} {d*1000:7.1f} ms  status={st} {err}")

        src = str(ROOT / "runs" / "demo_cam.h264")
        print(f"\n[2] 启动持续监测（文件源 {Path(src).name} + 假模型）")
        d, st, err = timed("/api/live/config", "POST",
                           {"window_stride_sec": 2.0,
                            "continuous_batch_windows": 2})
        print(f"    config: {d*1000:.1f} ms status={st} {err}")
        d, st, err = timed("/api/live/start", "POST",
                           {"source": src, "continuous": True, "use_mock": True,
                            "with_actions": True})
        print(f"    start : {d*1000:.1f} ms status={st} {err}")
        jid = None
        if st == 200:
            try:
                c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=8)
                c.request("POST", "/api/live/start",
                          json.dumps({"source": src, "continuous": True,
                                      "use_mock": True}).encode(),
                          {"Content-Type": "application/json"})
                jid = json.loads(c.getresponse().read().decode())["job_id"]
                c.close()
            except Exception as e:                          # noqa: BLE001
                print(f"    （拿 job_id 失败：{e}）")
        print(f"    job_id={jid}")

        print("\n[3] 监测运行中：连续探测各端点（每 3 秒一轮，共 5 轮）")
        paths = ["/api/state", "/api/log"]
        if jid:
            paths.append(f"/api/job/{jid}")
            paths.append(f"/api/job/{jid}/preview")
        print(f"    {'t':>5} " + " ".join(f"{p.split('/')[-1][:14]:>16}"
                                         for p in paths))
        for k in range(5):
            row = []
            for p in paths:
                d, st, err = timed(p, timeout=12)
                row.append(f"{d*1000:9.0f}ms/{str(st):>3}")
            print(f"    {k*3:>4}s " + " ".join(f"{x:>16}" for x in row))
            time.sleep(3)

        if jid:
            print("\n[4] 窗口结果长什么样（前 2 条的关键字段）")
            _, _, _ = timed(f"/api/stop/{jid}", "POST", {})
            time.sleep(1)
            try:
                c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=20)
                c.request("GET", f"/api/job/{jid}")
                j = json.loads(c.getresponse().read().decode())
                c.close()
                for w in (j.get("window_results") or [])[:2]:
                    print("    " + json.dumps(
                        {k: w.get(k) for k in
                         ("window_id", "t_start", "t_end", "frames", "frame0",
                          "latency", "skipped")}, ensure_ascii=False))
            except Exception as e:                          # noqa: BLE001
                print(f"    （读结果失败：{type(e).__name__}: {e}）")

        print("\n[5] 服务端日志尾部")
        for line in logf.read_text(encoding="utf-8",
                                   errors="replace").splitlines()[-25:]:
            print("    " + line)
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        fh.close()
        shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
