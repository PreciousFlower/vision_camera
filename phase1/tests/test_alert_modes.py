#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证三种告警策略 —— 不需要真实模型，用构造数据把逻辑跑穿。

为什么值得单独测：
    告警策略是"用户唯一会盯着的输出"。它一旦静默失效（比如字段缺失导致
    恒为 0），界面上看不出来 —— 之前就踩过一次（sanitize 契约不一致，
    逐窗口显示"吸烟 95%"、告警却永远是 0 条）。
    所以三种模式的判定边界必须逐条断言。

覆盖：
    · immediate —— 首个命中窗口就报
    · ai        —— 模型说报才报；模型没表态时回落到 violation（关键回退）
    · smoothed  —— M/N 才报；单个窗口命中不报
    · 三模式共用的去抖
    · Prompt 在 ai 模式下确实多要求了 alert / alert_reason

用法：
    python tests/test_alert_modes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

from civility.aggregate import derive_events  # noqa: E402
from civility.config import load_behaviors  # noqa: E402
from civility.prompts import build_system_prompt  # noqa: E402
from civility.vlm import WindowResult  # noqa: E402

FAILS: list[str] = []
BS = load_behaviors(PROJECT / "config" / "behaviors.yaml")
BID = "smoking"


def check(name: str, fn) -> None:
    try:
        detail = fn()
        print(f"  ✅ {name}" + (f" — {detail}" if detail else ""))
    except Exception as e:
        print(f"  ❌ {name} — {type(e).__name__}: {e}")
        FAILS.append(name)


def W(i: int, t0: float, hit: bool, conf: float = 0.95,
      alert=None, reason: str = "", video: str = "v.mp4") -> WindowResult:
    """造一个窗口结果。alert=None 表示模型没给该字段。"""
    viol = {}
    for b in BS.enabled:
        d = {"violation": False, "confidence": 0.0, "observable": "无",
             "person_desc": "", "bbox": None, "alert": None, "alert_reason": ""}
        if b.id == BID:
            d.update(violation=hit, confidence=conf if hit else 0.0,
                     observable="口鼻有烟雾" if hit else "无人吸烟")
            if alert is not None:
                d["alert"] = alert
                d["alert_reason"] = reason
        viol[b.id] = d
    return WindowResult(window_id=f"{video}__w{i:04d}", video=video, t_start=t0,
                        t_end=t0 + 4, parsed={"scene": "工地", "violations": viol},
                        backend="cluster")


def cfg(mode: str, **kw) -> dict:
    base = {"mode": mode, "min_confidence": 0.45, "window_k": 4,
            "min_hits": 2, "dedup_sec": 30}
    base.update(kw)
    return base


def main() -> int:
    print("=" * 72)
    print("告警策略验证（immediate / ai / smoothed）")
    print("=" * 72)

    # ── immediate ──
    def t_immediate():
        # 三个窗口都命中，去抖 30s，窗口间隔 4s → 只有第一条能出
        r = [W(0, 0.0, True), W(1, 4.0, True), W(2, 8.0, True)]
        a = derive_events(r, BS, cfg("immediate"))
        assert len(a) == 1, f"去抖后应只有 1 条，实际 {len(a)}"
        assert a[0].t_start == 0.0, "应立即报第一个窗口（t=0）"
        assert a[0].hits == 1, "immediate 模式单窗口即告警，hits 应为 1"

        # 去抖设为 0 → 三个窗口各报一条
        a2 = derive_events(r, BS, cfg("immediate", dedup_sec=0))
        assert len(a2) == 3, f"去抖 0 时应报 3 条，实际 {len(a2)}"

        # 只有第一个窗口命中（后面没有）→ 仍然报
        a3 = derive_events([W(0, 0.0, True), W(1, 4.0, False)], BS, cfg("immediate"))
        assert len(a3) == 1, "immediate 不应要求多窗口命中"
        return "首窗即报；去抖 0 时逐窗各报"

    check("immediate（AI 一命中就报）", t_immediate)

    # ── ai ──
    def t_ai():
        # 模型明确 alert=true → 报
        a = derive_events([W(0, 0.0, True, alert=True, reason="动作清楚，现在就该提醒")],
                          BS, cfg("ai", dedup_sec=0))
        assert len(a) == 1, "模型说报却没报"
        assert a[0].reason, "告警里应带上模型的理由"
        assert "现在就该提醒" in a[0].reason

        # 模型明确 alert=false → 不报（即使 violation=true）
        b = derive_events([W(0, 0.0, True, alert=False, reason="只有一帧疑似，证据不足")],
                          BS, cfg("ai", dedup_sec=0))
        assert len(b) == 0, "模型说不用报却报了（没有尊重模型的判断）"

        # ★ 关键回退：模型没给 alert 字段（换了模型或 Prompt 没带指令）→ 回落 violation
        c = derive_events([W(0, 0.0, True, alert=None)], BS, cfg("ai", dedup_sec=0))
        assert len(c) == 1, "模型没表态时应回落到 violation，否则告警会静默变 0"

        # 未命中 → 都不报
        d = derive_events([W(0, 0.0, False, alert=True)], BS, cfg("ai", dedup_sec=0))
        assert len(d) == 0, "没命中却报了"
        return "尊重 alert=false；缺字段回落 violation；理由随告警返回"

    check("ai（模型自己判断，含缺失回退）", t_ai)

    # ── smoothed ──
    def t_smoothed():
        # 单窗口命中 → 不报（这正是用户之前困惑的地方）
        a = derive_events([W(0, 0.0, True)], BS, cfg("smoothed", dedup_sec=0))
        assert len(a) == 0, "smoothed 下单窗口不应报警"

        # 两窗口命中 → 报，且 span/hits 反映聚合范围
        b = derive_events([W(0, 0.0, True), W(1, 4.0, True)], BS,
                          cfg("smoothed", dedup_sec=0))
        assert len(b) == 1, f"两窗口命中应报 1 条，实际 {len(b)}"
        assert b[0].hits == 2 and b[0].span == 2, f"hits/span 不对：{b[0].hits}/{b[0].span}"
        assert b[0].t_start == 0.0 and b[0].t_end == 8.0, "聚合区间不对"
        return "单窗不报、双窗报，hits/span 正确"

    check("smoothed（M/N 时序平滑）", t_smoothed)

    # ── 策略名容错 ──
    def t_bad_mode():
        # 写错策略名不应崩，也不应静默不报警 —— 应回落到 smoothed
        a = derive_events([W(0, 0.0, True), W(1, 4.0, True)], BS,
                          cfg("typo-mode", dedup_sec=0))
        assert len(a) == 1, "未知策略名应回落到 smoothed 并仍能报警"
        return "未知策略名回落 smoothed"

    check("策略名容错", t_bad_mode)

    # ── Prompt ──
    def t_prompt():
        off = build_system_prompt(BS, with_alert=False)
        on = build_system_prompt(BS, with_alert=True)
        assert len(on) > len(off), "ai 模式的 Prompt 应更长（多一段告警判定）"
        assert "alert_reason" in on, "ai 模式应要求 alert_reason 字段"
        assert '"alert"' in on, "ai 模式应要求 alert 字段"
        assert "alert_reason" not in off, "非 ai 模式不该出现 alert_reason"
        # 必须明确区分 violation 与 alert，否则模型会当成同一个字段照抄
        assert "violation 与 alert 不是同一件事" in on, "缺少二者的区分说明"
        return f"{len(off)} → {len(on)} 字符（+{len(on) - len(off)}）"

    check("Prompt（ai 模式追加告警判定段）", t_prompt)

    print()
    if FAILS:
        print(f"❌ {len(FAILS)} 项失败：{FAILS}")
        return 1
    print("✅ 全部通过 —— 三种告警策略行为正确")
    return 0


if __name__ == "__main__":
    sys.exit(main())
