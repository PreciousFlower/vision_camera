#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""摄像头监测（动作识别）链路自检 —— 不需要 GPU、不需要 VPN。

这条链路的核心是「让 AI 说明画面里每个人在做什么」，涉及三处改动：
    1. Prompt 里追加动作描述段与 people schema
    2. sanitize 把 people 归一化成统一结构
    3. server 把 people 透到 SSE，前端渲染成动作卡片

任何一环断了，页面上就是"画面里未识别到人"，而**看不出是哪一环出的问题**。
所以这里逐环断言。

用法：
    python tests/test_actions.py
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

from civility.config import load_behaviors  # noqa: E402
from civility.prompts import build_messages, build_system_prompt, sanitize  # noqa: E402

FAILS: list[str] = []
BS = load_behaviors(PROJECT / "config" / "behaviors.yaml")


def check(name, fn):
    try:
        d = fn()
        print(f"  ✅ {name}" + (f" — {d}" if d else ""))
    except Exception as e:
        print(f"  ❌ {name} — {type(e).__name__}: {e}")
        FAILS.append(name)


def main() -> int:
    print("=" * 74)
    print("摄像头监测（动作识别）链路自检")
    print("=" * 74)

    def t_prompt_off():
        p = build_system_prompt(BS, with_actions=False)
        assert '"people"' not in p, "未开启动作描述时不该要求 people 字段"
        assert "动作描述" not in p, "未开启动作描述时不该出现该段落"
        return f"{len(p)} 字符，无 people 要求"

    check("Prompt：默认不含动作描述", t_prompt_off)

    def t_prompt_on():
        p = build_system_prompt(BS, with_actions=True)
        assert "# 动作描述" in p, "缺少动作描述段落"
        assert '"people"' in p, "缺少 people 字段定义"
        # ⚠️ 字段名是**短键** at/act/chg（不是 where/action/changed）：
        #    实测解码 35.8 token/s ≈ 每 token 28 毫秒，字段名和值都要短，
        #    这一项直接决定监测能不能"实时"。长键仍被 sanitize 接受（见下面用例）。
        for field in ('"at"', '"act"', '"chg"'):
            assert field in p, f"people schema 缺少短键 {field}"
        assert "where" not in p and '"action"' not in p, \
            "不该再出现长键（那会让输出变长、延迟上升）"
        # 必须明确要求"具体到动作"而不是空话，否则模型会写"有人在活动"
        assert "空话" in p or "不要" in p, "缺少'避免空话'的约束"
        # 必须要求没人时填空数组，否则模型可能编造
        assert "空数组" in p or "[]" in p, "缺少'没人时填空数组'的约束"
        # 长度纪律：未命中不必写依据（这是压延迟的关键），命中必须写
        assert "只写 `behavior_id` 和 `violation`" in p, \
            "缺少'未命中只写两个键'的长度纪律（这是延迟优化的核心）"
        assert "单行 JSON" in p, "缺少'紧凑单行 JSON'要求（美化格式会多花 30~50% token）"
        # 已废弃字段不该再出现在 schema 里（模型会照写，白花 token）
        assert "person_desc" not in p and "bbox" not in p, \
            "person_desc/bbox 已从 schema 移除，不该再出现"
        return f"{len(p)} 字符（比默认多 {len(p) - len(build_system_prompt(BS))}）"

    check("Prompt：开启后含动作描述段与 people schema", t_prompt_on)

    def t_messages():
        # 真的走到 build_messages 这一层（server 调用的就是它）
        from civility.vlm import WindowRequest
        m = build_messages(BS, ["data:image/jpeg;base64,AAAA"],
                           ["t=0.0s"], with_alert=True, with_actions=True)
        sys_txt = m[0]["content"]
        assert "动作描述" in sys_txt and "告警判定" in sys_txt, \
            "两个模式应能同时开启（监测 + AI 告警）"
        req = WindowRequest(window_id="w0", video="v", t_start=0, t_end=6,
                            frames=[], labels=[], with_alert=True, with_actions=True)
        assert req.with_actions is True, "WindowRequest 应带 with_actions 字段"
        return "with_alert 与 with_actions 可同时开启"

    check("消息构造：两种模式可叠加", t_messages)

    def t_sanitize_people():
        obj = {
            "scene": "门口有一人",
            "people": [
                {"where": "画面左侧近处", "appearance": "深色外套男性",
                 "action": "蹲下捡起地上的纸片", "changed": "由站立变为蹲下"},
                {"where": "画面右侧远处", "action": "向右走出画面", "changed": "无"},
            ],
            "violations": {b.id: {"violation": False, "confidence": 0.0,
                                  "observable": "无"} for b in BS.enabled},
        }
        s = sanitize(obj, BS)
        assert len(s["people"]) == 2, f"people 条数不对：{len(s['people'])}"
        assert s["people"][0]["action"] == "蹲下捡起地上的纸片"
        assert s["people"][0]["changed"] == "由站立变为蹲下"
        assert s["people"][1]["appearance"] == "", "缺失字段应补空串而不是崩"
        return "2 人、字段补全、动作与变化都保留"

    check("sanitize：people 归一化", t_sanitize_people)

    def t_sanitize_compact():
        """紧凑键（at/act/chg）必须被归一成长键，前端与聚合逻辑才不用改。"""
        obj = {
            "scene": "门口有一人",
            "people": [{"at": "左侧近处", "act": "把纸杯扔向桶", "chg": "由站立变抛掷"}],
            "violations": [{"behavior_id": "littering", "violation": True,
                            "confidence": 0.9, "observable": "第3帧手变空"}],
        }
        s = sanitize(obj, BS)
        assert len(s["people"]) == 1, f"紧凑键没被识别：{s['people']}"
        p0 = s["people"][0]
        assert p0["where"] == "左侧近处", f"at 未映射到 where：{p0}"
        assert p0["action"] == "把纸杯扔向桶", f"act 未映射到 action：{p0}"
        assert p0["changed"] == "由站立变抛掷", f"chg 未映射到 changed：{p0}"
        assert p0["appearance"] == "", "紧凑格式没有 appearance，应补空串"
        return "at/act/chg → where/action/changed 映射正确"

    check("sanitize：紧凑键归一化（延迟优化后的新格式）", t_sanitize_compact)

    def t_sanitize_drop_empty():
        # 没写 action 的条目没有价值，应被丢掉（避免前端显示空行）
        obj = {"scene": "x", "people": [{"where": "左"}, {"action": "行走"}],
               "violations": {}}
        s = sanitize(obj, BS)
        assert len(s["people"]) == 1, "无 action 的条目应被丢弃"
        assert s["people"][0]["action"] == "行走"
        return "丢弃无动作条目，保留有效项"

    check("sanitize：丢弃无动作的空条目", t_sanitize_drop_empty)

    def t_sanitize_no_people():
        # 模型没给 people（例如非监测模式）时不能崩
        s = sanitize({"scene": "x", "violations": {}}, BS)
        assert s["people"] == [], "没有 people 时应为空列表"
        # people 传了垃圾类型也不能崩
        s2 = sanitize({"scene": "x", "people": "不是数组", "violations": {}}, BS)
        assert s2["people"] == [], "people 类型不对时应安全返回空列表"
        return "缺字段/类型错误都安全降级为空列表"

    check("sanitize：无 people 或类型错误不崩", t_sanitize_no_people)

    def t_people_does_not_break_alert():
        # people 的存在不能影响 violations 的解析（告警链路必须照常）
        from civility.aggregate import derive_events
        from civility.vlm import WindowResult
        objs = []
        for i in range(2):
            objs.append(WindowResult(
                window_id=f"w{i}", video="v.mp4", t_start=i * 6, t_end=i * 6 + 6,
                parsed={"scene": "工地", "people": [{"where": "中", "action": "吸烟"}],
                        "violations": {b.id: {"violation": b.id == "smoking",
                                              "confidence": 0.95 if b.id == "smoking" else 0.0,
                                              "observable": "口鼻有烟", "alert": True,
                                              "alert_reason": "证据充分"}
                                       for b in BS.enabled}},
                backend="cluster"))
        alerts = derive_events(objs, BS, {"mode": "ai", "min_confidence": 0.45,
                                         "dedup_sec": 0})
        # ⚠️ 期望 2 条而不是 1 条：dedup_sec=0 时**每个命中窗口各出一条**，
        #    这正是 ai 模式的定义（模型说报就报）。第一版写成了 1 条，
        #    是把"有去抖"和"无去抖"搞混了 —— 测试预期写错比代码写错更隐蔽。
        assert len(alerts) == 2, f"两个窗口都应产出告警，实际 {len(alerts)}"
        assert all(a.behavior_id == "smoking" for a in alerts)
        # 理由应透传（前端要显示"为什么现在就要提醒"）
        assert all(a.reason for a in alerts), "告警应带上模型的理由"
        return "两条窗口告警均产出，且都带模型理由"

    check("回归：people 不影响告警聚合", t_people_does_not_break_alert)

    print()
    if FAILS:
        print(f"❌ {len(FAILS)} 项失败：{FAILS}")
        return 1
    print("✅ 全部通过 —— 动作识别链路可用")
    return 0


if __name__ == "__main__":
    sys.exit(main())
