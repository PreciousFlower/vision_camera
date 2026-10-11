#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""开放风险协议与告警边界回归；完全离线，不需要 GPU、摄像头或 API Key。"""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from civility.aggregate import Alert, derive_events
from civility.config import Behavior, BehaviorSet
from civility.prompts import _schema_for, build_messages, build_system_prompt, sanitize, validate
from civility.vlm import CloudBackend, MockBackend, WindowRequest, WindowResult


BS = BehaviorSet([Behavior("smoking", "吸烟", "可见持烟和吸烟动作", severity="high")], {})


def risk(**changes) -> dict:
    item = {"risk_id": "fall", "name": "人员摔倒", "severity": "high", "confidence": 0.93,
            "observable": "中央人物由站立倒地，后续帧未起身", "alert": True,
            "alert_reason": "持续倒地，需要现场确认", "person_desc": "画面中央"}
    item.update(changes)
    return item


def window(index: int, risks=None, **changes) -> WindowResult:
    item = WindowResult(window_id=f"w{index}", video="camera", t_start=index * 2,
                        t_end=index * 2 + 2,
                        parsed={"scene": "门口", "violations": [{"behavior_id": "smoking",
                                                               "violation": False}],
                                "risks": risks or []})
    for key, value in changes.items():
        setattr(item, key, value)
    return item


def config(mode="ai", **changes) -> dict:
    item = {"mode": mode, "min_confidence": 0.45, "window_k": 3, "min_hits": 2,
            "dedup_sec": 30, "open_risks": True}
    item.update(changes)
    return item


class OpenRiskTests(unittest.TestCase):
    def test_risk_independent_of_configured_violations(self):
        events = derive_events([window(0, [risk()])], BS, config())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].behavior_id, "ai_risk:fall")
        self.assertEqual(events[0].source, "ai_risk")
        self.assertEqual(events[0].reason, risk()["alert_reason"])
        self.assertEqual(events[0].evidences, [risk()["observable"]])
        self.assertEqual(events[0].to_dict()["source"], "ai_risk")

    def test_all_modes_respect_false_and_min_confidence(self):
        for mode in ("immediate", "ai", "smoothed"):
            with self.subTest(mode=mode):
                for value in (risk(alert=False), risk(confidence=0.44)):
                    self.assertEqual(derive_events([window(0, [value]), window(1, [value])],
                                                   BS, config(mode)), [])
                self.assertEqual(derive_events([window(0), window(1)], BS, config(mode)), [])

    def test_risk_switch_and_custom_class(self):
        item = risk(risk_id="dangerous_machinery", name="人员靠近运转机器")
        self.assertEqual(derive_events([window(0, [item])], BS, config(open_risks=False)), [])
        events = derive_events([window(0, [item])], BS, config())
        self.assertEqual(events[0].behavior_id, "ai_risk:dangerous_machinery")

    def test_invalid_risk_fields_cannot_alert_or_validate(self):
        bad_changes = [
            {"risk_id": "<script>alert(1)</script>"}, {"risk_id": "fall:person1"},
            {"name": ""}, {"name": []}, {"severity": "critical"}, {"severity": None},
            {"confidence": float("nan")}, {"confidence": float("inf")},
            {"confidence": -0.1}, {"confidence": 1.1}, {"confidence": "0.99"},
            {"confidence": True}, {"confidence": 10 ** 1000},
            {"observable": ""}, {"observable": "无"}, {"observable": ["倒地"]},
            {"alert": "false"}, {"alert": 1}, {"alert": None},
            {"alert_reason": ""}, {"alert_reason": []},
        ]
        for changes in bad_changes:
            with self.subTest(changes=str(changes)[:80]):
                result = window(0, [risk(**changes)])
                self.assertEqual(sanitize(result.parsed, BS)["risks"], [])
                self.assertEqual(derive_events([result], BS, config()), [])
                self.assertFalse(validate(result.parsed, BS, strict=False)[0])
        for item in (None, "fall", 1, []):
            self.assertEqual(sanitize(window(0, [item]).parsed, BS)["risks"], [])

    def test_missing_required_risk_field_is_safe(self):
        for field in ("risk_id", "name", "severity", "confidence", "observable", "alert"):
            value = risk()
            value.pop(field)
            self.assertEqual(derive_events([window(0, [value])], BS, config()), [])

    def test_risk_requested_requires_explicit_array_but_legacy_still_valid(self):
        obj = window(0).parsed
        obj.pop("risks")
        self.assertTrue(validate(obj, BS, strict=False)[0])
        ok, problems = validate(obj, BS, strict=False, with_risks=True)
        self.assertFalse(ok)
        self.assertTrue(any("缺少 risks" in message for message in problems))
        obj["risks"] = []
        self.assertTrue(validate(obj, BS, strict=False, with_risks=True)[0])
        self.assertEqual(sanitize({"scene": "门口", "violations": []}, BS)["risks"], [])

    def test_bad_top_level_and_repeated_sanitize(self):
        for bad in (None, [], "bad", 123):
            self.assertEqual(sanitize(bad, BS)["risks"], [])
        for bad in ("fall", {"fall": risk()}, 1):
            self.assertEqual(sanitize({"scene": "x", "violations": [], "risks": bad}, BS)["risks"], [])
        original = window(0, [risk()]).parsed
        snapshot = copy.deepcopy(original)
        once = sanitize(original, BS)
        self.assertEqual(sanitize(once, BS), once)
        self.assertEqual(original, snapshot)
        self.assertTrue(validate(once, BS, strict=False)[0])

    def test_dedup_and_severity_upgrade(self):
        results = [window(0, [risk(severity="low")]), window(1, [risk(severity="low")]),
                   window(2, [risk(severity="medium")]), window(3, [risk(severity="high")]),
                   window(4, [risk(severity="medium")]), window(20, [risk(severity="high")])]
        for mode in ("ai", "immediate"):
            with self.subTest(mode=mode):
                events = derive_events(results, BS, config(mode))
                self.assertEqual([x.severity for x in events], ["low", "medium", "high", "high"])
        self.assertEqual(len(derive_events(results, BS, config(dedup_sec=0))), 6)

    def test_duplicate_risk_and_configured_id_cannot_collide(self):
        result = window(0, [risk(severity="low"), risk(severity="high"), risk(risk_id="smoking")])
        result.parsed["violations"][0].update(violation=True, confidence=0.9,
                                              observable="可见持烟并将烟举到嘴边", alert=True)
        events = derive_events([result], BS, config())
        self.assertEqual({x.behavior_id for x in events}, {"smoking", "ai_risk:fall"})
        self.assertEqual(next(x.severity for x in events if x.behavior_id == "ai_risk:fall"), "high")

    def test_smoothed_uses_all_windows_and_requires_current_hit(self):
        hit = risk()
        self.assertEqual(derive_events([window(0, [hit])], BS, config("smoothed")), [])
        self.assertEqual(derive_events([window(0, [hit]), window(1), window(2), window(3, [hit])],
                                       BS, config("smoothed")), [])
        results = [window(0, [hit]), window(1), window(2, [hit]), window(3)]
        events = derive_events(results, BS, config("smoothed", dedup_sec=0))
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0].hits, events[0].span), (2, 3))
        self.assertEqual(events[0].window_ids, ["w0", "w2"])

    def test_configured_smoothed_counts_missing_and_failed_windows(self):
        def smoking(index):
            result = window(index)
            result.parsed["violations"][0].update(violation=True, confidence=0.9,
                                                  observable="可见举烟至嘴边", alert=True)
            return result

        results = [smoking(0), window(1, parsed={"scene": "门口", "violations": [], "skipped": True}),
                   window(2, parsed=None, error="推理失败"), smoking(3)]
        self.assertEqual(derive_events(results, BS, config("smoothed")), [])
        compact = [smoking(0), window(1, parsed=None, error="推理失败"), smoking(2), window(3)]
        events = derive_events(compact, BS, config("smoothed", dedup_sec=0))
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0].hits, events[0].span), (2, 3))

    def test_smoothed_dedup_preserves_risk_upgrade(self):
        results = [window(0, [risk(severity="low")]), window(1, [risk(severity="low")]),
                   window(2, [risk(severity="high")]), window(3, [risk(severity="medium")])]
        events = derive_events(results, BS, config("smoothed"))
        self.assertEqual([x.severity for x in events], ["low", "high"])

    def test_risk_category_and_camera_dedup_are_independent(self):
        second = window(1, [risk()], video="second camera")
        events = derive_events([window(0, [risk(), risk(risk_id="fire", name="火情")]), second],
                               BS, config())
        self.assertEqual(len(events), 3)

    def test_error_window_cannot_supply_hit(self):
        results = [window(0, [risk()], error="推理失败"), window(1, [risk()])]
        self.assertEqual(len(derive_events(results, BS, config())), 1)
        self.assertEqual(derive_events(results, BS, config("smoothed")), [])

    def test_legacy_and_configured_malformed_values(self):
        obj = {"scene": "门口", "violations": [{"behavior_id": "smoking", "violation": True,
                                                "confidence": 0.9, "observable": "举烟至嘴边"}]}
        self.assertTrue(validate(obj, BS, strict=False)[0])
        self.assertEqual(sanitize(obj, BS)["risks"], [])
        item = window(0, parsed=obj)
        events = derive_events([item], BS, config())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].source, "configured")
        for changes in ({"violation": "false"}, {"confidence": float("nan")},
                        {"confidence": float("inf")}, {"alert": "false"}):
            changed = copy.deepcopy(obj)
            changed["violations"][0].update(changes)
            self.assertEqual(derive_events([window(0, parsed=changed)], BS, config()), [])

    def test_empty_behavior_set_and_prompt_opt_in(self):
        empty = BehaviorSet([], {})
        with self.assertRaises(ValueError):
            build_system_prompt(empty)
        prompt = build_system_prompt(empty, with_risks=True)
        self.assertIn("开放风险识别", prompt)
        self.assertIn('"risks"', prompt)
        self.assertIn('"violations":[]', prompt)
        self.assertNotIn('"risk_id"', build_system_prompt(BS))
        schema = _schema_for([], with_risks=True)
        self.assertEqual(schema["properties"]["violations"]["maxItems"], 0)
        self.assertIn("risks", schema["required"])
        events = derive_events([window(0, [risk()], parsed={"scene": "空走廊起火", "violations": [],
                                                         "risks": [risk(risk_id="fire", name="火情",
                                                                        observable="门口地面有明火并持续冒烟",
                                                                        alert_reason="明火持续，需要现场处理")]} )],
                               empty, config())
        self.assertEqual(events[0].behavior_id, "ai_risk:fire")
        messages = build_messages(empty, ["data:image/jpeg;base64,AAAA"], with_risks=True,
                                  with_alert=True, with_actions=True)
        for message in messages:
            if message["role"] == "assistant":
                doc = json.loads(message["content"])
                self.assertEqual(doc["violations"], [])
                self.assertIn("risks", doc)
        self.assertEqual(BS.defaults, {})

    def test_mock_and_cloud_propagate_risk_request(self):
        req = WindowRequest("w0", "fall_pos.mp4", 0, 2, [], with_risks=True)
        mock = MockBackend(latency=0).infer(req, BS)
        self.assertEqual(len(derive_events([mock], BS, config())), 1)
        self.assertFalse(mock.parsed["violations"][0]["violation"])
        captured = []
        backend = CloudBackend("https://example.invalid", "mock", "unused", max_retries=1)

        def fake_post(payload):
            captured.append(payload)
            return {"choices": [{"message": {"content": json.dumps(window(0, [risk()]).parsed)}}]}

        backend._post = fake_post
        result = backend.infer(req, BS)
        self.assertIsNone(result.error)
        self.assertIn("开放风险识别", captured[0]["messages"][0]["content"])
        self.assertEqual(captured[0]["max_tokens"], 512 + 384)
        req.with_alert = req.with_actions = True
        backend.infer(req, BS)
        self.assertEqual(captured[1]["max_tokens"], 512 + 180 + 90 + 384)

    def test_cloud_retries_when_risk_requested_but_missing(self):
        req = WindowRequest("w0", "camera", 0, 2, [], with_risks=True)
        backend = CloudBackend("https://example.invalid", "mock", "unused", max_retries=2)
        captured = []
        responses = [window(0).parsed, window(0, [risk()]).parsed]
        responses[0].pop("risks")

        def fake_post(payload):
            captured.append(payload)
            doc = responses[len(captured) - 1]
            return {"choices": [{"message": {"content": json.dumps(doc)}}]}

        backend._post = fake_post
        result = backend.infer(req, BS)
        self.assertIsNone(result.error)
        self.assertEqual(result.retries, 1)
        self.assertEqual(len(captured), 2)
        self.assertIn("缺少 risks", captured[1]["messages"][-1]["content"])
        self.assertEqual(result.parsed["risks"][0]["risk_id"], "fall")


if __name__ == "__main__":
    unittest.main(verbosity=2)
