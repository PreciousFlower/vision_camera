#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""集群侧导入路径的冒烟测试 —— 不需要 GPU，几秒钟跑完。

为什么需要它：集群作业的成本是"排队时间 + 机时费 + 一次失败的往返"。
本地的 mock 自检覆盖不到集群独有的失败模式，比如：

    1. 集群侧 import civility.prompts 时连带 import yaml，而 vlm 环境没装 pyyaml
       → 作业 30 秒内崩，白白排队一次（本文件用 `sys.modules['yaml'] = None`
         把 yaml 伪装成不可用来复现这个场景）
    2. behaviors.json 的字段和 Behavior 数据类对不上
    3. Prompt 里用 PIL 图片对象构造消息时结构不对

用法（在作业包目录里跑，或指向作业包）：
    python tests/smoke_cluster_import.py
    python tests/smoke_cluster_import.py runs/cluster/pkg
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Windows 控制台默认 GBK，输出 emoji 会 UnicodeEncodeError。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def main() -> int:
    pkg = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT / "runs" / "cluster" / "pkg"
    pkg = pkg.resolve()
    if not (pkg / "behaviors.json").exists():
        print(f"❌ 找不到 {pkg / 'behaviors.json'}")
        print("   先跑：python -m civility.cli cluster-prepare")
        return 1

    sys.path.insert(0, str(pkg))
    # ⚠️ 关键：把 yaml 伪装成不可用，模拟集群 vlm 环境（它的包清单里没有 pyyaml）。
    #    如果 civility 的任何模块在顶层 import yaml，这里就会 ImportError。
    sys.modules["yaml"] = None  # type: ignore[assignment]

    failures = []

    def check(name: str, fn):
        try:
            detail = fn()
            print(f"  ✅ {name}" + (f" — {detail}" if detail else ""))
        except Exception as e:
            print(f"  ❌ {name} — {type(e).__name__}: {e}")
            failures.append(name)

    print(f"作业包: {pkg}")
    print("模拟集群环境（yaml 不可用）：")

    state: dict = {}

    def _load():
        from civility.config import Behavior, BehaviorSet

        raw = json.loads((pkg / "behaviors.json").read_text(encoding="utf-8"))
        items = [
            Behavior(
                id=b["id"],
                name=b.get("name", b["id"]),
                definition=b.get("definition", ""),
                evidence_required=b.get("evidence_required", ""),
                counter_examples=b.get("counter_examples") or [],
                observable_cues=b.get("observable_cues") or [],
                severity=b.get("severity", "medium"),
            )
            for b in raw.get("behaviors", [])
        ]
        state["bs"] = BehaviorSet(behaviors=items, defaults=raw.get("defaults") or {})
        return f"{len(items)} 个行为"

    check("import civility.config（不得依赖 yaml）", _load)
    check("import civility.prompts", lambda: __import__("civility.prompts") and "ok")

    if "bs" in state:
        bs = state["bs"]

        def _prompt():
            from civility.prompts import build_system_prompt

            sp = build_system_prompt(bs)
            assert len(sp) > 500, "system prompt 太短，可能没渲染出行为定义"
            assert "JSON" in sp, "system prompt 里没有输出格式说明"
            return f"{len(sp)} 字符"

        check("生成 system prompt", _prompt)

        def _messages_str():
            from civility.prompts import build_messages

            m = build_messages(bs, ["data:image/jpeg;base64,AAAA", "data:image/jpeg;base64,BBBB"],
                               ["t=0.0s", "t=1.0s"])
            parts = [p["type"] for p in m[-1]["content"]]
            assert parts.count("image_url") == 2, f"图片部件数不对: {parts}"
            assert m[0]["role"] == "system"
            assert any(x["role"] == "assistant" for x in m), "few-shot 少了 assistant 轮"
            return f"{len(m)} 条消息"

        check("构造消息（云端：data URL）", _messages_str)

        def _messages_pil():
            from PIL import Image

            from civility.prompts import build_messages

            imgs = [Image.new("RGB", (32, 32), "red") for _ in range(2)]
            m = build_messages(bs, imgs, ["t=0.0s", "t=1.0s"])
            parts = [p["type"] for p in m[-1]["content"]]
            assert parts.count("image") == 2, f"PIL 图片部件数不对: {parts}"
            return f"{len(m)} 条消息"

        check("构造消息（集群：PIL 对象）", _messages_pil)

        def _validate():
            from civility.prompts import sanitize, validate

            good = {
                "scene": "测试",
                "violations": [
                    {"behavior_id": b.id, "violation": False, "confidence": 0.1}
                    for b in bs.enabled
                ],
            }

            # ① 结构合法：在线路径用 strict=False，绝不能被 lint 判成失败。
            #    这条很关键 —— 若 lint 触发重试，"模型判定全部无违规、
            #    因而不写描述"这种完全合法的输出会被无限重试、白烧机时。
            ok, probs = validate(good, bs, strict=False)
            assert ok, f"结构合法的输出被判非法（会导致线上无谓重试）: {probs}"

            # ② 全部未命中、且都没写 observable —— 在**现在的**长度纪律下
            #    这是**正常**输出（判 false 时要求只写两个键来省 token，
            #    实测每 token 约 28 毫秒）。所以即使 strict=True 也不该报它。
            #    ⚠️ 旧断言在这里要求"必须被报出"，那是旧纪律（判 false 也要写依据），
            #    改 Prompt 时同步改测试，否则测试会挡住正确的优化。
            ok2, probs2 = validate(good, bs, strict=True)
            assert ok2, f"全未命中且无 observable 属于正常输出，不该被判异常: {probs2}"

            # ③ 真正该报的是**命中却没给依据**：那种结论无法复核。
            hit_no_evidence = {
                "scene": "测试",
                "violations": [
                    {"behavior_id": bs.enabled[0].id, "violation": True,
                     "confidence": 0.9, "observable": ""},
                ],
            }
            ok3, probs3 = validate(hit_no_evidence, bs, strict=True)
            assert not ok3, "命中的行为没写 observable，strict 模式应当报出"
            assert any("observable" in p for p in probs3), f"lint 内容不对: {probs3}"

            # ③ 结构错误必须挡（无论 strict 与否）
            bad = {"scene": "x", "violations": [{"behavior_id": "不存在的行为",
                                                 "violation": True, "confidence": 1}]}
            ok3, probs3 = validate(bad, bs, strict=False)
            assert not ok3, "非法 behavior_id 竟然通过了校验"

            # ④ dict 形式与 list 形式都要支持
            #    （踩过的严重 bug：只认 list，导致 aggregate 拿 dict 回来时
            #      所有行为被判 false、告警永远为 0）
            as_dict = {"scene": "测试",
                       "violations": {b.id: {"violation": True, "confidence": 0.9,
                                             "observable": "有依据"}
                                      for b in bs.enabled}}
            ok4, probs4 = validate(as_dict, bs, strict=True)
            assert ok4, f"dict 形式的 violations 不被接受: {probs4}"

            s = sanitize(good, bs)
            assert set(s["violations"]) == {b.id for b in bs.enabled}
            return f"结构校验 {len(probs3)} 项拦截，lint 与结构分离，list/dict 双形状支持"

        check("Schema 校验与净化（严格/宽松、list/dict）", _validate)

        def _extract():
            from civility.prompts import extract_json

            fence = "```json\n" + json.dumps({"a": 1}) + "\n```"
            assert extract_json(fence) == {"a": 1}, "代码块包裹解析失败"
            assert extract_json('前言 {"b": {"c": 2}} 后语') == {"b": {"c": 2}}, "前后缀解析失败"
            assert extract_json("完全不是 JSON") is None
            return "代码块 / 前后缀 / 无 JSON 三种情况均正确"

        check("输出 JSON 解析容错", _extract)

    print()
    if failures:
        print(f"❌ {len(failures)} 项失败：{failures}")
        print("   在提交集群作业前必须修掉，否则会浪费一次排队 + 机时。")
        return 1
    print("✅ 全部通过 —— 集群侧的 import 与 Prompt 逻辑可在本地安全验证")
    print("   下一步：cluster-upload → sbatch → cluster-fetch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
