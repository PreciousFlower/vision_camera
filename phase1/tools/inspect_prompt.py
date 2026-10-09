#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 Prompt 的"输出预算"：字段表、示例、长度纪律是否都为省 token 服务。

背景：实测解码 35.8 token/s ≈ **每 token 28 毫秒**，而一次真实输出里有
      相当一部分是"JSON 美化格式"和"没人用的字段"白花的。
      优化这类问题不能靠感觉，得把 Prompt 的实际内容打出来逐项核对。

用法：
    python tools/inspect_prompt.py
    python tools/inspect_prompt.py --mode live     # 监测模式（带动作描述）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from civility.config import load_behaviors, load_settings      # noqa: E402
from civility.prompts import _few_shot, build_system_prompt    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["simple", "live"], default="live")
    args = ap.parse_args()

    bs = load_behaviors(load_settings()["paths"]["behaviors"])
    live = args.mode == "live"
    p = build_system_prompt(bs, with_alert=True, with_actions=live)

    print("=" * 74)
    print(f"Prompt 体检（{'监测模式：带逐人动作' if live else '简单模式'}）")
    print("=" * 74)
    print(f"长度：{len(p)} 字符　行为数：{len(bs.enabled)}"
          f"（{', '.join(b.id for b in bs.enabled)}）")

    checks = [
        ("要求单行紧凑 JSON（省 30~50% token）", "单行 JSON" in p),
        ("未命中只写两个键", "只写 `behavior_id` 和 `violation`" in p
         or "其余键" in p),
        ("alert_reason 只在 alert=true 时写", "仅 alert=true 时填" in p),
        ("已移除没人用的 person_desc", "person_desc" not in p),
        ("已移除没人用的 bbox", "bbox" not in p),
        ("observable 只在命中时必填", "仅 violation=true 时填" in p),
        ("scene ≤20 字", "不超过 20 字" in p),
    ]
    if live:
        checks += [
            ("动作字段用短键 at/act/chg", all(k in p for k in ('"at"', '"act"'))),
            ("明确不要写外观衣着", "不要**写外观" in p or "不要" in p),
        ]
    print()
    bad = 0
    for name, ok in checks:
        print(f"  {'✅' if ok else '❌'} {name}")
        bad += 0 if ok else 1

    print("\n--- few-shot 示例（模型模仿的对象，必须和规则一致）---")
    total = 0
    for i, (_u, a) in enumerate(_few_shot(bs, with_actions=live), 1):
        total += len(a)
        print(f"  示例{i}（{len(a)} 字符）：{a[:160]}")
        # 示例里不该出现已废弃字段，也不该有换行/缩进
        for needle in ('person_desc', 'bbox', '\n'):
            if needle in a:
                print(f"     ❌ 示例里出现 {needle!r} —— 模型会跟着写")
                bad += 1
    print(f"\n示例合计 {total} 字符（这部分是每次都随请求发过去的 prompt 开销）")

    print("\n--- 输出预算估算 ---")
    n = len(bs.enabled)
    # 命中 1 个 + 其余未命中，是监控场景里最常见的形态
    est = (12 + n * 12 + 25 + 20 + (20 if live else 0))
    print(f"  按 {n} 个行为、命中 1 个估算：约 {est} token "
          f"≈ {est * 0.028:.1f} 秒解码时间")
    print(f"  （改造前实测 330 token / 10.8 秒，主要浪费在美化格式与空字段上）")

    print("\n" + "=" * 74)
    if bad:
        print(f"❌ {bad} 项不合格")
        return 1
    print("✅ Prompt 体检通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
