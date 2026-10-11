"""Evidence regressions: replay unsupported model claims through the real alert path.

These fixtures test acceptance of model output, not visual accuracy of the VLM.
"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from civility.aggregate import derive_events
from civility.config import Behavior, BehaviorSet, load_behaviors
from civility.prompts import build_messages, build_system_prompt, sanitize, validate
from civility.vlm import WindowResult


class VisualEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.bs = load_behaviors()

    def claim(self, bid, observable, **extra):
        return {"scene": "一人手持物品", "people": [], "risks": [], "violations": [
            {"behavior_id": b.id, "violation": b.id == bid,
             **({"confidence": 0.99, "observable": observable, "alert": True,
                 "alert_reason": "动作清楚", **extra} if b.id == bid else {})}
            for b in self.bs.enabled
        ]}

    def test_unsupported_smoke_and_landing_claims_do_not_become_alerts(self):
        for bid, observable in (("smoking", "手靠近嘴巴，口鼻有烟雾"),
                                ("littering", "手持垃圾袋，垃圾落地")):
            for mode in ("ai", "immediate", "smoothed"):
                with self.subTest(behavior=bid, mode=mode):
                    obj = self.claim(bid, observable)
                    rows = [WindowResult(f"w{i}", "camera", i * 2, i * 2 + 2,
                                         parsed=obj) for i in range(2)]
                    self.assertEqual(derive_events(rows, self.bs, {
                        "mode": mode, "min_hits": 2, "window_k": 2,
                        "min_confidence": 0.45}), [])

    def test_unsupported_claim_is_not_displayed_as_confirmed(self):
        obj = self.claim("smoking", "手靠近嘴巴，口鼻有烟雾")
        self.assertFalse(sanitize(obj, self.bs)["violations"]["smoking"]["violation"])

    def test_valid_evidence_still_allows_both_behaviors_in_all_modes(self):
        for bid, evidence in (("smoking", {"tobacco": 1, "use": 2}),
                              ("littering", {"held": 1, "released": 2,
                                            "landed": 2, "abandoned": 3})):
            obj = self.claim(bid, "各必要动作在对应帧清晰可见", evidence=evidence)
            clean = sanitize(obj, self.bs, frame_count=4)
            self.assertEqual(clean["violations"][bid]["evidence_status"], "provided")
            self.assertEqual(sanitize(clean, self.bs, frame_count=4), clean)
            rows = [WindowResult(f"w{i}", "camera", i * 2, i * 2 + 2,
                                 parsed=clean) for i in range(2)]
            for mode in ("ai", "immediate", "smoothed"):
                with self.subTest(behavior=bid, mode=mode):
                    self.assertEqual(len(derive_events(rows, self.bs, {
                        "mode": mode, "min_hits": 2, "window_k": 2})), 1)

    def test_missing_and_invalid_frame_references_are_suppressed(self):
        for evidence in (None, {}, {"tobacco": 1}, {"tobacco": 1, "use": 0},
                         {"tobacco": True, "use": 2}, {"tobacco": "1", "use": 2},
                         {"tobacco": 1, "use": 5}, {"tobacco": 1.0, "use": 2}):
            with self.subTest(evidence=evidence):
                obj = self.claim("smoking", "手遮嘴，声称在吸烟", evidence=evidence)
                clean = sanitize(obj, self.bs, frame_count=4)
                hit = clean["violations"]["smoking"]
                self.assertFalse(hit["violation"])
                self.assertFalse(hit["alert"])
                self.assertEqual(hit["evidence_status"], "insufficient")
                self.assertTrue(hit["evidence_issues"])
                self.assertEqual(sanitize(clean, self.bs, frame_count=4), clean)

    def test_valid_frame_references_do_not_replace_observable_facts(self):
        for facts in ("", "未知", [], None):
            obj = self.claim("smoking", facts, evidence={"tobacco": 1, "use": 2})
            self.assertFalse(sanitize(obj, self.bs, frame_count=4)["violations"]["smoking"]["violation"])

    def test_littering_sequence_cannot_be_single_frame_or_reversed(self):
        for refs in ((1, 1, 1, 1), (3, 2, 1, 4), (1, 3, 2, 4)):
            obj = self.claim("littering", "声称物品脱手并落地", evidence=dict(zip(
                ("held", "released", "landed", "abandoned"), refs)))
            hit = sanitize(obj, self.bs, frame_count=4)["violations"]["littering"]
            self.assertFalse(hit["violation"])
            self.assertIn("时序", build_system_prompt(self.bs))

    def test_known_configured_risks_cannot_bypass_evidence_gate(self):
        for rid, name in (("smoking", "吸烟"), ("suspected_smoking_risk", "疑似吸烟风险"),
                          ("different_id", "乱扔垃圾")):
            obj = self.claim("smoking", "看不清手中物")
            obj["risks"] = [{"risk_id": rid, "name": name, "severity": "high",
                             "confidence": 0.99, "observable": "模型声称有烟雾或物品落地",
                             "alert": True, "alert_reason": "模型声称证据充分"}]
            clean = sanitize(obj, self.bs, frame_count=4)
            self.assertFalse(clean["risks"][0]["alert"])
            self.assertEqual(sanitize(clean, self.bs, frame_count=4), clean)
            self.assertEqual(derive_events([WindowResult("w", "camera", 0, 2, parsed=clean)],
                                          self.bs, {"mode": "immediate", "open_risks": True}), [])

    def test_confirmed_behavior_duplicate_does_not_create_review_warning(self):
        obj = self.claim("smoking", "可辨认烟支，点燃并吸食", evidence={"tobacco": 1, "use": 2})
        obj["risks"] = [{"risk_id": "smoking", "name": "吸烟", "severity": "high",
                         "confidence": 0.99, "observable": "可见吸烟", "alert": True,
                         "alert_reason": "明确吸烟"}]
        clean = sanitize(obj, self.bs, frame_count=2)
        self.assertTrue(clean["violations"]["smoking"]["violation"])
        self.assertEqual(clean["violations"]["smoking"]["evidence_issues"], [])
        self.assertEqual(clean["risks"], [])
        self.assertEqual(sanitize(clean, self.bs, frame_count=2), clean)

    def test_uncertain_outputs_are_not_schema_errors_requiring_retries(self):
        obj = self.claim("smoking", "无法看清", evidence={"tobacco": 0, "use": 0})
        self.assertTrue(validate(obj, self.bs, strict=False)[0])
        self.assertFalse(sanitize(obj, self.bs)["violations"]["smoking"]["violation"])

    def test_custom_checks_work_without_hard_coded_behavior_ids(self):
        custom = Behavior("example", "测试行为", "可见动作", evidence_checks={"object": "物体可见"})
        bs = BehaviorSet([custom], {})
        obj = {"scene": "场景", "violations": [{"behavior_id": "example", "violation": True,
                                              "confidence": 0.99, "observable": "物体可见"}]}
        self.assertFalse(sanitize(obj, bs)["violations"]["example"]["violation"])
        obj["violations"][0]["evidence"] = {"object": 1}
        self.assertTrue(sanitize(obj, bs, frame_count=1)["violations"]["example"]["violation"])

    def test_prompt_removes_positive_anchors_and_includes_counterexamples_without_fewshot(self):
        messages = build_messages(self.bs, ["frame1", "frame2"], ["t=8", "t=9"],
                                  with_few_shot=False, with_alert=True, with_actions=True,
                                  with_risks=True)
        prompt = messages[0]["content"]
        self.assertNotIn("第3帧手由持物变空，物体落地", prompt)
        self.assertNotIn("例如：只在某一帧边缘看到疑似烟雾", prompt)
        for text in ("指甲剪", "提着垃圾袋", "tobacco", "landed", "evidence_checks"):
            self.assertIn(text, prompt)
        self.assertIn("帧1；t=8", messages[-1]["content"][1]["text"])
        with self.assertRaises(ValueError):
            build_messages(self.bs, ["frame1", "frame2"], ["t=8"])

    def test_new_config_is_validated(self):
        for kwargs in ({"evidence_checks": []}, {"evidence_checks": {"bad-key": "说明"}},
                       {"evidence_sequence": ["missing"]}):
            with self.assertRaises(ValueError):
                Behavior("test", "测试", "测试定义", **kwargs)


if __name__ == "__main__":
    unittest.main()
