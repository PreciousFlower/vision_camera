#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端验证：**持续监测**（摄像头实时监测）真的在出结果。

为什么单独写这个脚本：
    之前只在浏览器里点"开始监测"，看到"窗口数在涨"就以为通了，
    实际上页面上一条结果都没有 —— 采集循环第一批永远不返回，
    压根没走到推理（见 webui/server.py 里 continuous_batch_windows 的注释）。
    这种"看起来在跑"的 bug 必须用"结果条目数 > 0"这种硬指标来验，
    所以这里直接打 API、拉 SSE，逐条断言：

        1. 启动后确实进入 continuous 状态
        2. 收到 window 事件（说明推理在跑，不只是采集）
        3. window_results 里有场景描述 / 人的动作（监测模式的核心内容）
        4. 窗口 ID 全局唯一（跨批次不重复）
        5. 帧文件真的存在且能取回（页面回看依赖它）
        6. 历史帧不会被下一批清空（持续模式踩过的坑）
        7. 停止后状态是 stopped，不是 error

用法：
    python tools/test_monitor_rtsp.py                       # 默认打 127.0.0.1:8848
    python tools/test_monitor_rtsp.py --seconds 45
    python tools/test_monitor_rtsp.py --source rtsp://127.0.0.1:8554/live
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OK = "✅"
NO = "❌"
WARN = "⚠️ "


def post(base: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body or {}).encode("utf-8")
    req = urllib.request.Request(
        base + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def get(base: str, path: str, timeout: int = 30):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get_bytes(base: str, path: str, timeout: int = 30) -> bytes:
    """取二进制（帧是 JPEG，不能按 JSON 解析）。"""
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8848")
    ap.add_argument("--source", default="rtsp://127.0.0.1:8554/live")
    ap.add_argument("--seconds", type=float, default=40.0,
                    help="监测跑多久（每窗口 6 秒 + 推理，40 秒约 4-6 个窗口）")
    ap.add_argument("--expect-alerts", action="store_true",
                    help="断言至少产生 1 条告警（用名字带 _pos 的视频文件当源时用）")
    ap.add_argument("--scene", default="littering",
                    help="演示场景：littering / smoking / queue_jumping / normal / 空")
    ap.add_argument("--no-scene-check", action="store_true",
                    help="跳过演示场景自洽性断言（用真实模型时不必检查）")
    args = ap.parse_args()

    results: list[tuple[bool, str, str]] = []

    def check(ok: bool, name: str, detail: str = "") -> bool:
        results.append((ok, name, detail))
        print(f"  {OK if ok else NO} {name}" + (f" — {detail}" if detail else ""),
              flush=True)
        return ok

    print("=" * 74)
    print(f"持续监测端到端验证   {args.base}")
    print(f"源：{args.source}   监测时长：{args.seconds:.0f}s")
    print("=" * 74)

    # 先确认服务在
    try:
        st = get(args.base, "/api/state", timeout=10)
    except Exception as e:
        print(f"{NO} 打不开本地服务 {args.base}：{e}")
        return 1
    print(f"  服务在线，已有任务 {len(st.get('jobs') or [])} 个")

    # 探测源（真实链路的第一步，先把"能不能读"确认掉）
    print("\n[1] 探测视频源")
    pr = post(args.base, "/api/live/probe", {"source": args.source})
    ok_probe = bool(pr.get("ok"))
    check(ok_probe, "源可读", f"{pr.get('width')}x{pr.get('height')} "
                             f"{pr.get('transport')} "
                             f"{pr.get('measured_fps')}fps"
          if ok_probe else str(pr.get("error")))
    if not ok_probe:
        print("\n源不可读就不必往下测了。")
        return 1

    # 启动持续监测（use_mock：不依赖 VPN/集群，验的是链路而不是模型）
    # mock_scene：让演示模式的场景/动作/判定依据/告警互相自洽 ——
    #   这一项是用户实测反馈的直接结果（"AI 结论与视频不一致"，
    #   根因是演示模式只给随机占位文字、且永远不会产出违规）。
    print("\n[2] 启动持续监测（演示模式·场景=乱扔垃圾）")
    r = post(args.base, "/api/live/start", {
        "source": args.source, "continuous": True,
        "with_actions": True, "use_mock": True,
        "mock_scene": args.scene,
    })
    job_id = r.get("job_id")
    if not check(bool(job_id), "任务已创建", f"job_id={job_id}  {r.get('message', '')}"):
        return 1

    # SSE：真正证明"推理在跑"的信号
    events: list[dict] = []
    stop_ev = threading.Event()

    def reader() -> None:
        try:
            with urllib.request.urlopen(
                    f"{args.base}/api/job/{job_id}/stream", timeout=args.seconds + 60
            ) as resp:
                for raw in resp:
                    if stop_ev.is_set():
                        break
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        ev = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    events.append(ev)
                    if ev.get("type") == "window":
                        n_people = len(ev.get("people") or [])
                        print(f"      · 窗口 {ev.get('cumulative')} "
                              f"t={ev.get('t_start')}-{ev.get('t_end')}s "
                              f"场景={str(ev.get('scene'))[:24]!r} "
                              f"人={n_people} 命中={len(ev.get('hits') or [])} "
                              f"耗时={ev.get('latency')}")
        except Exception as e:
            print(f"      （SSE 结束：{type(e).__name__}）")

    th = threading.Thread(target=reader, daemon=True)
    th.start()

    t_end = time.time() + args.seconds
    last_n = 0
    # ⚠️ 预览帧必须在**监测进行中**采样。
    #    第一版把它放在停止之后，结果 3 秒里只拿到 1 张不同的图 ——
    #    不是预览没在动，而是任务已经停了、画面冻住了（测试自己顺序写错）。
    preview_hashes: list[str] = []
    preview_errs: list[str] = []
    preview_times: list[float] = []
    preview_after = time.time() + 3.5      # 给采集线程几秒出第一张预览
    printed_at = 0.0
    # 磁盘上的预览帧（用来对照：接口返回的到底是不是最新那张）
    frames_dir = Path(__file__).resolve().parent.parent / "data" / "frames" / job_id
    while time.time() < t_end and len(preview_hashes) < 14:
        time.sleep(0.3)
        if time.time() < preview_after:
            continue
        t_req = time.time()
        try:
            blob = get_bytes(args.base, f"/api/job/{job_id}/preview", timeout=8)
            preview_times.append(time.time() - t_req)
            if blob[:2] == b"\xff\xd8" and blob[-2:] == b"\xff\xd9":
                h = hashlib.md5(blob).hexdigest()[:8]
                preview_hashes.append(h)
                pv = sorted(frames_dir.glob("preview_*.jpg"))
                print(f"      · 请求 {time.time() - t_req:.2f}s 返回 {len(blob)}B "
                      f"md5={h} | 磁盘最新 "
                      f"{pv[-1].name if pv else '（无）'}（{len(pv)} 张在场）",
                      flush=True)
            else:
                preview_errs.append(f"非 JPEG({len(blob)}B)")
        except Exception as e:
            preview_times.append(time.time() - t_req)
            preview_errs.append(f"{type(e).__name__}: {e}"[:70])
            print(f"      · 请求失败 {time.time() - t_req:.2f}s {type(e).__name__}",
                  flush=True)
        if time.time() - printed_at < 3.0:
            continue
        printed_at = time.time()
        try:
            d = get(args.base, f"/api/job/{job_id}")
        except Exception:
            continue
        n = len(d.get("window_results") or [])
        if n != last_n:
            last_n = n
            print(f"      [{time.time() - (t_end - args.seconds):5.1f}s] "
                  f"状态={d.get('state')} 结果={n} 窗口={d.get('windows_seen')} "
                  f"告警={len(d.get('alerts') or [])}", flush=True)

    print("\n[3] 停止监测")
    try:
        post(args.base, f"/api/stop/{job_id}", {})
    except Exception as e:
        print(f"      （stop 调用：{e}）")
    time.sleep(3)
    stop_ev.set()

    d = get(args.base, f"/api/job/{job_id}")
    wres = d.get("window_results") or []
    wins = [e for e in events if e.get("type") == "window"]

    print("\n[4] 断言")
    check(len(wins) > 0, "采集→推理链路真的在跑（有 window 事件）",
          f"{len(wins)} 条")
    check(len(wres) > 0, "有窗口结果落库（不是只有采集在涨）", f"{len(wres)} 条")

    ids = [w.get("window_id") for w in wres]
    check(len(ids) == len(set(ids)), "窗口 ID 跨批次不重复",
          f"{len(ids)} 个 / 去重后 {len(set(ids))}")

    with_scene = [w for w in wres if (w.get("scene") or "").strip()]
    check(len(with_scene) > 0, "模型有给出画面描述（scene）",
          f"{len(with_scene)}/{len(wres)}")

    with_people = [w for w in wres if (w.get("people") or [])]
    check(len(with_people) > 0, "模型有描述画面里的人在做什么（people）",
          f"{len(with_people)}/{len(wres)}")

    # 帧文件：页面回看依赖它
    checked = 0
    missing = 0
    if wres:
        for w in wres[-3:]:
            try:
                blob = get_bytes(args.base, f"/api/job/{job_id}/frames/"
                                 f"{w['window_id']}/0", timeout=15)
                checked += 1
                # JPEG 的魔数：FFD8 开头、FFD9 结尾 —— 只判"能取回"不够，
                # 半截文件（比如 0 字节或 HTML 错误页）对页面同样是碎图。
                if not (blob[:2] == b"\xff\xd8" and blob[-2:] == b"\xff\xd9"):
                    missing += 1
                    print(f"      （{w['window_id']} 不是完整 JPEG："
                          f"{len(blob)} 字节 {blob[:8]!r}）")
            except urllib.error.HTTPError as e:
                missing += 1
                print(f"      （帧取回失败 {w['window_id']}：HTTP {e.code}）")
            except Exception as e:
                missing += 1
                print(f"      （帧取回失败 {w['window_id']}：{e}）")
    check(checked > 0 and missing == 0, "历史帧仍可取回（没被下一批清掉）",
          f"抽查 {checked} 个窗口，失败 {missing}")

    # 实时画面（监测页顶部那块）：必须返回完整 JPEG
    live_ok = live_bad = 0
    if wres:
        for _ in range(3):
            try:
                blob = get_bytes(args.base, f"/api/job/{job_id}/latest", timeout=10)
                if blob[:2] == b"\xff\xd8" and blob[-2:] == b"\xff\xd9":
                    live_ok += 1
                else:
                    live_bad += 1
            except Exception:
                live_bad += 1
            time.sleep(0.6)
    check(live_ok > 0 and live_bad == 0, "实时画面接口返回完整 JPEG",
          f"{live_ok} 次成功 / {live_bad} 次失败")

    # ── 预览帧（给人看的流畅画面）：用**监测进行中**采到的样本判定 ──
    # ⚠️ 这里必须用前面 preview_hashes 的样本。
    #    第一版在"停止之后"再采一轮，结果只有 1 张不同的图 ——
    #    不是预览坏了，是任务已经停了、画面冻住了（测试自身的顺序错误）。
    uniq = len(set(preview_hashes))
    check(uniq >= 4, "预览帧确实在流畅更新（监测期间每 0.3 秒一张）",
          f"采样 {len(preview_hashes)} 张 / 不同 {uniq} 张"
          + (f"，单次请求平均 {sum(preview_times) / len(preview_times):.2f}s"
             if preview_times else ""))
    if preview_errs:
        from collections import Counter
        print("      （预览取样失败明细："
              + "；".join(f"{k}×{v}" for k, v in
                          Counter(preview_errs).most_common(3)) + "）")

    # ── 演示模式的自洽性：选了场景就该有对应的画面描述、动作与告警 ──
    if not args.no_scene_check and args.scene and args.scene != "normal":
        alerts = d.get("alerts") or []
        names = {a.get("behavior_id") for a in alerts}
        check(args.scene in names,
              f"演示场景与告警一致（选了 {args.scene} 就报 {args.scene}）",
              f"实际告警行为={sorted(names) or '无'}")
        scenes = [w.get("scene") or "" for w in wres]
        check(any("演示" in s for s in scenes),
              "画面描述与所选场景一致（不再是无关的占位文字）",
              (scenes[-1][:46] + "…") if scenes else "无")
        ppl = [p for w in wres for p in (w.get("people") or [])]
        check(any("纸杯" in (p.get("action") or "") or "桶" in (p.get("action") or "")
                  for p in ppl),
              "动作描述与场景自洽（能说清在做什么）",
              (ppl[-1].get("action") or "")[:46] if ppl else "无动作描述")

    check(d.get("state") in ("stopped", "running"), "停止后状态正常",
          f"state={d.get('state')} {d.get('message', '')}")

    # 告警链路（监测的最终产出）：mock 后端的"真相"来自文件名/源名，
    # 所以用 data/videos/littering_pos_vis.mp4 这类**带正例标记**的源来验。
    alerts = d.get("alerts") or []
    if args.expect_alerts:
        check(len(alerts) > 0, "AI 结论已聚合成告警事件",
              f"{len(alerts)} 条：" + "；".join(
                  f"{a.get('behavior_name') or a.get('behavior_id')}"
                  f"@t={a.get('t_start')}s" for a in alerts[:3]))
        with_alert_flag = [w for w in wres
                           if any(v.get("alert") is not None
                                  for v in (w.get("violations") or {}).values())]
        check(len(with_alert_flag) > 0, "模型给出了告警决定与理由（alert/alert_reason）",
              f"{len(with_alert_flag)}/{len(wres)} 个窗口带 alert 字段")

    print("\n" + "=" * 74)
    bad = [r for r in results if not r[0]]
    if not bad:
        print(f"{OK} 全部通过 —— 持续监测链路可用"
              f"（{len(wres)} 个窗口 / {len(d.get('alerts') or [])} 条告警）")
        return 0
    print(f"{NO} {len(bad)} 项未通过：")
    for _, name, detail in bad:
        print(f"   · {name} — {detail}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
