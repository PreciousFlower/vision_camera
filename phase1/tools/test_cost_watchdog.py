#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证"成本守卫"真的会把空转的 A800 退掉（用户被孤儿作业坑了 48 元）。

为什么必须单独测：
    成本守卫的逻辑是"到点自动停服务"。这种代码的失败方式很隐蔽：
    · 写错了 → 永远不触发 → 用户以为有保护，其实还在烧钱（最坏情况）
    · 判据太激进 → 用户正在用的时候被退掉 → 分析中断
    两种都不能靠"看代码觉得对"来确认，只能把时间条件**压缩到秒级**实测一遍。

这里用 monkeypatch 把配置改成"空闲 3 秒就释放"，并**替换掉真正的 stop()**，
所以这个测试**不会真的去杀集群作业**，却完整走过了判定逻辑。

用法：
    python tools/test_cost_watchdog.py
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from webui import server as S  # noqa: E402

OK, NO = "✅", "❌"
results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    results.append((ok, name, detail))
    print(f"  {OK if ok else NO} {name}" + (f" — {detail}" if detail else ""),
          flush=True)
    return ok


class FakeSession:
    """假的推理服务会话：只记录 stop() 有没有被调用，不碰真集群。"""

    def __init__(self, state: str = "ready", started_ago: float = 0.0):
        self.state = state
        self.jobid = "9999999"
        self.node = "gpu-fake"
        self.model = "fake"
        self.started_at = time.time() - started_ago
        self.stopped = False
        self.swept = False

    def stop(self, sweep_orphans: bool = True) -> list[str]:
        self.stopped = True
        self.swept = sweep_orphans
        self.state = "stopped"
        return [self.jobid]


class FakeJob:
    def __init__(self, state: str = "running"):
        self.state = state
        self.stop_flag = False


def main() -> int:
    print("=" * 74)
    print("成本守卫（空闲自动释放）验证 —— 不会真的杀集群作业")
    print("=" * 74)

    # 把"空闲 15 分钟"压缩成"空闲 3 秒"，其余逻辑完全不动
    S.STATE.cfg["service"] = {
        "idle_release_min": 3 / 60.0,     # 3 秒
        "max_run_hours": 0,               # 本组不测上限，单独测
        "gpu_price_per_hour": 4.0,
    }

    # ── 1. 页面还开着 → 不许释放 ──
    print("\n[1] 页面刚刚有心跳（人在用）")
    sess = FakeSession(state="ready")
    S.STATE.session = sess
    S.STATE.page_visible = True
    S.STATE.last_visible_at = time.time()
    S.STATE.last_interact = time.time()
    S.STATE.jobs.clear()
    S.STATE.auto_released = ""
    why = S.watchdog_tick()
    check(why == "" and not sess.stopped, "有人在看时不会释放节点",
          f"动作={why or '无'}")

    # ── 2. 页面心跳停了 → 应该释放 ──
    print("\n[2] 页面心跳停了 4 秒（模拟关了浏览器）")
    # ⚠️ 服务要"已经起来一会儿"才算数：看门狗对新启动的服务有宽限期
    #    （防止"刚点启动、页面还没打开"就被退掉）。这里 started_ago 必须
    #    大于压缩后的空闲阈值，否则测的是宽限期、不是释放逻辑。
    sess = FakeSession(state="ready", started_ago=6.0)
    S.STATE.session = sess
    S.STATE.page_visible = False          # 页面已经不可见（关了/切后台）
    S.STATE.last_visible_at = time.time() - 4
    why = S.watchdog_tick()
    check(sess.stopped, "空闲超时后自动释放节点",
          f"原因：{why[:60]}…" if why else "没有动作（❌ 等于保护失效）")
    check(sess.swept, "释放时顺手扫了遗留作业（sweep_orphans=True）")
    check("省下" in (S.STATE.auto_released or ""),
          "给出了「省了多少钱」的说明", (S.STATE.auto_released or "")[-28:])

    # ── 2b. 关键回归：切到后台时心跳仍在发（定时器不会停）──
    #      只按"有没有请求"判空闲的实现会在这里永远不释放 —— 必须复现住。
    print("\n[2b] 页面在后台，但心跳每 30 秒照发（真实浏览器的行为）")
    sess = FakeSession(state="ready", started_ago=6.0)
    S.STATE.session = sess
    S.STATE.auto_released = ""
    S.STATE.page_visible = False
    S.STATE.last_visible_at = time.time() - 4      # 4 秒前切到后台
    S.STATE.last_ui_seen = time.time()             # 心跳刚刚才来过！
    why = S.watchdog_tick()
    check(sess.stopped, "后台页面照发心跳时仍能释放（不被心跳骗过去）",
          "按「最后一次可见」而不是「最后一次心跳」判定")

    # ── 2c. 页面开着但人没碰键盘鼠标 ──
    print("\n[2c] 页面开着、可见，但 4 秒没动过鼠标键盘")
    sess = FakeSession(state="ready", started_ago=6.0)
    S.STATE.session = sess
    S.STATE.auto_released = ""
    S.STATE.page_visible = True
    S.STATE.last_visible_at = time.time()
    S.STATE.last_interact = time.time() - 4
    why = S.watchdog_tick()
    check(sess.stopped, "开着页面但长时间无操作也会释放",
          f"原因：{why.split('。')[0]}" if why else "没有动作")

    # ── 3. 正在分析画面 → 即使空闲也不许打断 ──
    print("\n[3] 空闲超时，但此刻正在分析画面")
    sess = FakeSession(state="ready", started_ago=99)
    S.STATE.session = sess
    S.STATE.auto_released = ""
    S.STATE.page_visible = False
    S.STATE.last_visible_at = time.time() - 999
    S.STATE.jobs["j1"] = FakeJob("running")
    why = S.watchdog_tick()
    check(not sess.stopped and why == "", "正在干活时不会因为空闲被退掉",
          "分析任务优先，不会中断用户")
    S.STATE.jobs.clear()

    # ── 4. 未就绪的服务不该被处理 ──
    print("\n[4] 服务本来就没启动")
    sess = FakeSession(state="idle")
    S.STATE.session = sess
    S.STATE.page_visible = False
    S.STATE.last_visible_at = time.time() - 999
    why = S.watchdog_tick()
    check(not sess.stopped and why == "", "没启动的服务不会误触发释放")

    # ── 5. 最长运行上限 ──
    print("\n[5] 服务连续运行超过自设上限")
    S.STATE.cfg["service"] = {"idle_release_min": 0, "max_run_hours": 1 / 3600.0,
                              "gpu_price_per_hour": 4.0}
    sess = FakeSession(state="ready", started_ago=5.0)   # 已跑 5 秒 > 1 秒上限
    S.STATE.session = sess
    S.STATE.auto_released = ""
    S.STATE.cap_extended_at = 0.0
    why = S.watchdog_tick()
    check(sess.stopped, "超过最长运行上限后自动释放",
          f"原因：{why[:44]}…" if why else "没有动作")

    print("\n[6] 超上限但正在分析 → 延长而不是硬切")
    sess = FakeSession(state="ready", started_ago=5.0)
    S.STATE.session = sess
    S.STATE.auto_released = ""
    S.STATE.cap_extended_at = 0.0
    S.STATE.jobs["j2"] = FakeJob("capturing")
    why = S.watchdog_tick()
    check(not sess.stopped and why == "", "干活中到上限只延长、不打断")
    S.STATE.jobs.clear()

    # ── 7. 花费估算 ──
    print("\n[7] 花费估算（页面上那个「已花多少」的数字）")
    S.STATE.session = FakeSession(state="ready", started_ago=3600)
    c = S.service_cost()
    check(abs(c["est_cost"] - 4.0) < 0.2 and c["uptime_text"].startswith("1 小时"),
          "1 小时 × 4 元/卡时 ≈ 4 元", f"{c['uptime_text']} / {c['est_cost']} 元")

    # ── 8. 孤儿识别 ──
    print("\n[8] 孤儿作业识别（网页不知道、却还在计费的作业）")
    S.STATE.session = FakeSession(state="ready")
    S.STATE.session.jobid = "111"
    orph = S.describe_orphans([
        {"jobid": "111", "name": "civility-serve"},     # 是我管的
        {"jobid": "222", "name": "civility-serve"},     # 孤儿
        {"jobid": "333", "name": "civility-vlm"},       # 孤儿
    ])
    check([j["jobid"] for j in orph] == ["222", "333"],
          "把自己正在管的那个排除掉", f"识别出 {len(orph)} 个")

    # 收尾：别把测试状态留在 STATE 里影响后面的真实使用
    S.STATE.session = None
    S.STATE.jobs.clear()
    S.STATE.auto_released = ""
    S.STATE.cfg = S.load_settings()

    print("\n" + "=" * 74)
    bad = [r for r in results if not r[0]]
    if bad:
        print(f"{NO} {len(bad)} 项未通过：")
        for _, n, d in bad:
            print(f"   · {n} — {d}")
        return 1
    print(f"{OK} 全部通过 —— 成本守卫可用：久不操作会自动退节点，"
          f"干活时不会被打断")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
