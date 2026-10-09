#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证「本地视频分析」的两个真实故障已修好（用户实测反馈）。

故障 1：一个视频窗口被反复分析
    根因：run_job 的外层 `while True` 对非持续任务**没有出口**，
    每轮都重新抽帧、重新切窗，于是同一个视频被完整分析 N 遍。
    实测证据：某次任务 windows_total=3 而 windows_seen=12（跑了 4 遍）。

故障 2：告警要等点了「中止」才出现
    根因：告警只在"一批窗口全部推理完"之后聚合一次（文件任务只有一批），
    而中止触发的提前 return 会把上一轮算出的 alerts 一起推给页面。
    所以告警恰好在中止那一刻冒出来。

本测试走**真实 HTTP 接口**（上传 → 开始分析 → 拉 SSE），用演示模式
（本地假模型 + 指定场景）跑一个真实视频文件，逐条断言：
    1. 每个窗口 ID 只出现一次（无重复分析）
    2. windows_seen == windows_total（跑且只跑一遍）
    3. 告警在最后一个窗口**之前**就已经推送（不再等结束/中止）
    4. 任务自己走到 done，不需要点中止
    5. 演示模式的场景、判定依据、告警三者自洽（能看到理由）
    6. 勾了「循环播放」时轮次递增、窗口 ID 带轮次后缀（不会与上一轮撞车）

用法：
    python tools/test_file_pass.py
    python tools/test_file_pass.py --video data/videos/littering_pos_vis.mp4
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OK, NO = "✅", "❌"
results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    results.append((ok, name, detail))
    print(f"  {OK if ok else NO} {name}" + (f" — {detail}" if detail else ""),
          flush=True)
    return ok


def post_json(base: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def get_json(base: str, path: str, timeout: int = 60):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def upload(base: str, video: Path) -> str:
    """multipart 上传（不引第三方库，手写边界）。"""
    boundary = "----civility" + uuid.uuid4().hex
    ctype = mimetypes.guess_type(video.name)[0] or "video/mp4"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{video.name}"\r\n'.encode(),
        f"Content-Type: {ctype}\r\n\r\n".encode(),
        video.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        base + "/api/upload", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))["job_id"]


def run_pass(base: str, video: Path, *, use_mock: bool = True,
             scene: str = "littering", loop: bool = False,
             max_sec: float = 240.0) -> dict:
    """跑一遍分析，返回收集到的 SSE 事件与被分析的窗口序列。"""
    jid = upload(base, video)
    events: list[dict] = []
    win_ids: list[str] = []
    alert_marks: list[tuple[int, int]] = []   # (第几个 window 之后出现, 告警条数)
    lock = threading.Lock()
    stop_ev = threading.Event()
    done_ev = threading.Event()

    def reader() -> None:
        try:
            with urllib.request.urlopen(
                    f"{base}/api/job/{jid}/stream", timeout=max_sec + 60) as resp:
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
                    with lock:
                        events.append(ev)
                        if ev.get("type") == "window":
                            win_ids.append(ev.get("window_id", ""))
                        if ev.get("type") == "alerts":
                            alert_marks.append((len(win_ids),
                                                len(ev.get("alerts") or [])))
                        if ev.get("type") in ("done", "error", "final"):
                            done_ev.set()
        except Exception:
            pass

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    time.sleep(0.5)
    post_json(base, f"/api/detect/{jid}",
              {"use_mock": use_mock, "mock_scene": scene, "loop": loop})

    t_end = time.time() + max_sec
    while time.time() < t_end:
        try:
            d = get_json(base, f"/api/job/{jid}", timeout=30)
        except Exception:
            time.sleep(1.0)
            continue
        last = d.get("state")
        if last in ("done", "stopped", "error") and len(win_ids) > 0:
            if last != "running":
                break
        time.sleep(0.7)
    # 让 SSE 把最后的 done/final 事件读完
    done_ev.wait(timeout=8)
    stop_ev.set()
    time.sleep(0.4)
    final = get_json(base, f"/api/job/{jid}", timeout=30)
    return {"job_id": jid, "events": events, "win_ids": win_ids,
            "alert_marks": alert_marks, "final": final}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8848")
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    video = Path(args.video) if args.video else (
        HERE.parent / "data" / "videos" / "littering_pos_vis.mp4")
    if not video.exists():
        print(f"{NO} 找不到测试视频 {video}")
        return 2

    print("=" * 74)
    print(f"本地视频分析 · 回归测试   {args.base}")
    print(f"视频：{video.name}（{video.stat().st_size / 1024:.0f} KB）")
    print("=" * 74)

    # ── 第一组：默认（跑一遍就结束）──
    print("\n[1] 默认模式：应当跑一遍就结束，且告警实时出现")
    r = run_pass(args.base, video, scene="littering", loop=False)
    final = r["final"]
    wins = r["win_ids"]
    total = final.get("windows_total") or 0
    seen = final.get("windows_seen") or 0

    print(f"    任务 {r['job_id']}  状态={final.get('state')}  "
          f"窗口 total={total} seen={seen}  收到的 window 事件={len(wins)}")

    check(final.get("state") == "done",
          "任务自己跑完（不需要点中止）", f"state={final.get('state')}")
    check(len(wins) == len(set(wins)) and len(wins) > 0,
          "每个窗口只被分析一次（无重复）",
          f"{len(wins)} 个窗口事件 / 去重后 {len(set(wins))}")
    check(seen == total and total > 0, "windows_seen 与 windows_total 一致",
          f"{seen} / {total}")

    # 告警必须出现在"最后一个窗口之前"（这是本故障的核心判据）
    first_alert_at = next((i for i, n in r["alert_marks"] if n > 0), None)
    check(first_alert_at is not None,
          "分析过程中就出现了告警（不是等结束/中止）",
          f"第 {first_alert_at} 个窗口之后出现" if first_alert_at else "整程没有告警")
    if first_alert_at:
        check(first_alert_at < max(1, len(wins)),
              "告警早于最后一个窗口（说明是增量聚合，不是收尾才算）",
              f"告警在第 {first_alert_at} 个窗口后 / 共 {len(wins)} 个窗口")

    # 演示场景自洽：选了乱扔垃圾，就该有 littering 的告警 + 判定依据
    alerts = final.get("alerts") or []
    names = {a.get("behavior_id") for a in alerts}
    check("littering" in names, "演示场景与告警一致（选了乱扔垃圾就报乱扔垃圾）",
          f"告警行为={sorted(names)}")
    reasons = [a.get("reason") or a.get("alert_reason") or "" for a in alerts]
    check(any("纸杯" in x or "桶" in x or "演示" in x for x in reasons),
          "告警带可读的理由（不是空话）",
          (reasons[0][:46] + "…") if reasons and reasons[0] else "无理由")

    # ── 第二组：循环播放开关 ──
    print("\n[2] 勾选「循环播放」：应当按轮次继续、ID 不撞车、且按视频时长配速")
    r2 = run_pass(args.base, video, scene="littering", loop=True, max_sec=75)
    wins2 = r2["win_ids"]
    ids = [w for w in wins2]
    rounds = {w.split("__r")[-1] for w in ids if "__r" in w}
    print(f"    收到 {len(ids)} 个 window 事件，轮次数={len(rounds) + 1}")
    check(len(ids) > len(wins), "循环模式下确实跑了不止一遍",
          f"循环 {len(ids)} 个窗口 vs 单遍 {len(wins)} 个")
    check(len(ids) == len(set(ids)), "循环模式下窗口 ID 仍不重复",
          f"去重后 {len(set(ids))}")
    check(bool(rounds), "第 2 轮起的窗口 ID 带轮次后缀（区分不同轮）",
          f"轮次标记={sorted(rounds)[:3]}")
    # 配速：视频文件"能读多快就读多快"，不配速的话 14 秒视频一秒能跑几十轮
    # （实测 75 秒产出 3447 个窗口，页面会被刷爆）
    video_sec = max(1.0, total * 4.0)          # 窗口数 × 步长 ≈ 视频时长
    want_rounds = 75.0 / video_sec
    got_rounds = len(rounds) + 1
    check(got_rounds <= want_rounds + 3, "循环按视频时长配速（不会疯狂刷屏）",
          f"{got_rounds} 轮 / 75 秒内预期约 {want_rounds:.0f} 轮"
          f"（视频约 {video_sec:.0f} 秒）")
    try:
        post_json(args.base, f"/api/stop/{r2['job_id']}", {})
    except Exception:
        pass

    print("\n" + "=" * 74)
    bad = [x for x in results if not x[0]]
    if bad:
        print(f"{NO} {len(bad)} 项未通过：")
        for _, n, d in bad:
            print(f"   · {n} — {d}")
        return 1
    print(f"{OK} 全部通过 —— 视频只跑一遍、告警实时出现、循环开关可用")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
