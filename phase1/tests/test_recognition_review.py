"""Exercise model-assisted review without accepting an unsupported model claim."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from civility.config import load_behaviors
from civility.prompts import sanitize, _few_shot
from civility.recognition import generation_budget, infer_with_review


class RecognitionReviewTests(unittest.TestCase):
    def setUp(self):
        self.bs = load_behaviors()
        self.frames, self.labels = ["a", "b", "c"], ["t=0", "t=1", "t=2"]

    def response(self, evidence, *, violation=True, error=None):
        obj = {"scene": "一人持杯后丢到地面", "people": [], "risks": [], "violations": [
            {"behavior_id": b.id, "violation": b.id == "littering" and violation,
             **({"observable": "第1帧手持杯，第2帧脱手，第3帧杯落地，人物继续行走",
                 "evidence": evidence, "confidence": .95, "alert": True,
                 "alert_reason": "可见主动丢弃动作"} if b.id == "littering" and violation else {"alert": False})}
            for b in self.bs.enabled]}
        return {"parsed": sanitize(obj, self.bs, frame_count=3), "raw": json.dumps(obj),
                "error": error, "latency_sec": 1, "tokens_out": 100}

    def run_model(self, responses, **options):
        calls = []
        def send(frames, labels, **kwargs):
            calls.append(kwargs)
            return copy.deepcopy(responses[len(calls) - 1])
        result = infer_with_review(send, self.frames, self.labels, self.bs,
                                  with_alert=True, with_actions=True, with_risks=True,
                                  max_new_tokens=1000, **options)
        return result, calls

    def test_missing_reference_gets_independent_model_review_then_alert(self):
        first = self.response({"held": 1, "released": 2})
        second = self.response({"held": 1, "released": 2, "landed": 3})
        result, calls = self.run_model([first, second])
        self.assertTrue(result["parsed"]["violations"]["littering"]["alert"])
        self.assertEqual(result["verification"]["status"], "confirmed")
        self.assertEqual([b["id"] for b in calls[1]["behaviors"]["behaviors"]], ["littering"])
        self.assertFalse(calls[1]["with_risks"])
        self.assertEqual(result["raw"], first["raw"])
        self.assertEqual(result["latency_sec"], 2)

    def test_model_review_can_reject_claim_instead_of_forcing_true(self):
        first = self.response({"held": 1, "released": 2})
        result, calls = self.run_model([first, self.response({}, violation=False)])
        self.assertFalse(result["parsed"]["violations"]["littering"]["violation"])
        self.assertEqual(result["verification"]["status"], "rejected")
        self.assertEqual(len(calls), 2)

    def test_still_missing_or_invalid_evidence_stays_pending_without_loop(self):
        first = self.response({"held": 1, "released": 2})
        for second in (first, self.response({"held": 1, "released": 2, "landed": 99}),
                       self.response({}, error="connection failed")):
            with self.subTest(second=second):
                result, calls = self.run_model([first, second])
                self.assertFalse(result["parsed"]["violations"]["littering"]["alert"])
                self.assertEqual(result["verification"]["status"], "unresolved")
                self.assertEqual(len(calls), 2)

    def test_valid_positive_and_negative_need_no_review(self):
        for first in (self.response({"held": 1, "released": 2, "landed": 3}),
                      self.response({}, violation=False)):
            result, calls = self.run_model([first])
            self.assertEqual(len(calls), 1)
            self.assertNotIn("verification", result)

    def test_disabling_review_preserves_pending_evidence(self):
        result, calls = self.run_model([self.response({"held": 1})], review_enabled=False)
        self.assertEqual(len(calls), 1)
        self.assertFalse(result["parsed"]["violations"]["littering"]["violation"])

    def test_missing_risks_are_not_silently_repaired_before_ingress_check(self):
        first = {"parsed": {"scene": "走廊", "violations": {}}, "error": None}
        result, calls = self.run_model([first])
        self.assertNotIn("risks", result["parsed"])
        self.assertEqual(len(calls), 1)

    def test_model_review_failure_keeps_other_confirmed_behavior(self):
        first = self.response({"held": 1})
        first["parsed"]["violations"]["smoking"] = {
            "violation": True, "alert": True, "confidence": .95, "observable": "可见烟支用于口部吸食",
            "evidence": {"tobacco": 1, "use": 2}}
        result, _ = self.run_model([first, {"error": "timeout"}])
        self.assertTrue(result["parsed"]["violations"]["smoking"]["alert"])
        self.assertFalse(result["parsed"]["violations"]["littering"]["alert"])

    def test_length_truncation_retries_once_with_larger_budget(self):
        first = {"parsed": None, "raw": '{"scene":', "error": "输出里找不到合法 JSON",
                 "finish_reason": "length", "latency_sec": 1, "tokens_out": 1000}
        result, calls = self.run_model([first, self.response({"held": 1, "released": 2, "landed": 3})])
        self.assertGreater(calls[1]["max_new_tokens"], calls[0]["max_new_tokens"])
        self.assertEqual(result["generation_retry"]["reason"], "length")
        self.assertTrue(result["parsed"]["violations"]["littering"]["violation"])

    def test_budget_increases_with_actions_and_configured_evidence(self):
        base = generation_budget(self.bs, with_alert=True, with_actions=False, with_risks=True)
        actions = generation_budget(self.bs, with_alert=True, with_actions=True, with_risks=True)
        self.assertGreater(actions, base)
        self.assertGreaterEqual(base, 896)

    def test_all_teaching_examples_include_action_process_when_actions_enabled(self):
        examples = _few_shot(self.bs, with_alert=True, with_actions=True, with_risks=True)
        for _, answer in examples:
            for person in json.loads(answer).get('people', []):
                self.assertTrue(person.get('chg'), answer)


if __name__ == "__main__":
    unittest.main()
