"""Zero-cost evidence protocol regressions across package, backends and SSE.

These tests replay model responses; they do not measure visual recognition accuracy.
Run with: python tests/test_evidence_protocol.py
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
import tempfile
import types
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from PIL import Image

from civility import cli
from civility.config import Behavior, BehaviorSet, load_behaviors
from civility.extract import Window, write_manifest
from civility.prompts import sanitize
from civility.runner import behavior_fingerprint
from civility.vlm import CloudBackend, ClusterBackend, MockBackend, WindowRequest
from cluster import run_cluster_vlm, vlm_service
from webui import server as S


def evidence_behaviors():
    return BehaviorSet([
        Behavior("smoking", "吸烟", "可辨认烟具与实际吸食动作同时可见",
                 evidence_checks={"tobacco": "烟具可辨认", "use": "实际吸食可见"}),
        Behavior("littering", "乱扔垃圾", "同一物品被丢弃至地面后遗弃",
                 evidence_checks={"held": "手持物品", "released": "物品脱手",
                                  "landed": "同一物品落地", "abandoned": "人物离开未捡回"},
                 evidence_sequence=["held", "released", "landed", "abandoned"]),
    ], {"max_new_tokens": 260})


def smoking_claim(*, evidence=None, as_mapping=False):
    claim = {"violation": True, "confidence": .99, "observable": "手靠近嘴，声称烟雾可见",
             "alert": True, "alert_reason": "声称吸烟动作明确"}
    if evidence is not None:
        claim["evidence"] = evidence
    false_item = {"violation": False, "alert": False}
    violations = ({"smoking": claim, "littering": false_item} if as_mapping else [
        {"behavior_id": "smoking", **claim},
        {"behavior_id": "littering", **false_item},
    ])
    return {"scene": "一人手持物品", "people": [], "violations": violations, "risks": []}


class EvidencePackagingTests(unittest.TestCase):
    def test_definition_snapshots_remain_separate_inside_same_batch(self):
        batcher = vlm_service.Batcher()
        first = evidence_behaviors()
        second = copy.deepcopy(first)
        second.by_id("smoking").definition = "另一套临时定义"
        batcher.submit([{}], ["t=0"], 512, False, True, False, behaviors=first)
        batcher.submit([{}], ["t=0"], 512, False, True, False, behaviors=second)
        class Processor:
            tokenizer = types.SimpleNamespace(padding_side="right")

            def __init__(self):
                self.messages = []

            def apply_chat_template(self, messages, **kwargs):
                self.messages.append(messages)
                return "text"

            def __call__(self, **kwargs):
                return kwargs
        processor = Processor()
        with patch.dict(vlm_service.STATE, {"behaviors": first, "processor": processor,
                                           "has_qvu": False}), \
                patch.object(vlm_service, "decode_images", side_effect=[[object()], [object()]]):
            vlm_service._build_inputs(batcher.q)
            self.assertIs(vlm_service.STATE["behaviors"], first)
        self.assertNotIn("另一套临时定义", processor.messages[0][0]["content"])
        self.assertIn("另一套临时定义", processor.messages[1][0]["content"])

    def test_cli_package_round_trips_checks_through_both_cluster_loaders(self):
        bs = evidence_behaviors()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = root / "frame.jpg"
            Image.new("RGB", (8, 8), "gray").save(frame)
            manifest = write_manifest([Window("w0", "test.mp4", 0, 2,
                                              [str(frame)], ["t=0"])], root / "manifest.jsonl")
            cfg = {"paths": {"results_dir": str(root / "runs")},
                   "aggregate": {"mode": "ai", "open_risks": True}}
            # The package payload is real; unrelated source trees are empty test fixtures.
            source = root / "source"
            source.mkdir()
            args = types.SimpleNamespace(manifest=str(manifest))
            with patch.object(cli, "_cfg", return_value=(cfg, bs)), \
                    patch.object(cli, "ROOT", source), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.cmd_cluster_prepare(args), 0)
            package = root / "runs" / "cluster" / "pkg" / "behaviors.json"
            payload = json.loads(package.read_text(encoding="utf-8"))
            self.assertEqual(payload["behaviors"][0]["evidence_checks"],
                             bs.by_id("smoking").evidence_checks)
            self.assertEqual(payload["behaviors"][1]["evidence_sequence"],
                             bs.by_id("littering").evidence_sequence)
            for loader in (vlm_service.load_behaviors_from, run_cluster_vlm.load_behaviors_json):
                with self.subTest(loader=loader.__name__):
                    restored = loader(package)
                    for original in bs.enabled:
                        self.assertEqual(asdict(restored.by_id(original.id)), asdict(original))

    def test_old_cluster_package_defaults_are_explicitly_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "behaviors.json"
            package.write_text(json.dumps({"behaviors": [{
                "id": "custom", "name": "自定义", "definition": "已有定义"}]}), encoding="utf-8")
            for loader in (vlm_service.load_behaviors_from, run_cluster_vlm.load_behaviors_json):
                restored = loader(package).by_id("custom")
                self.assertEqual(restored.evidence_checks, {})
                self.assertEqual(restored.evidence_sequence, [])

    def test_changing_only_evidence_invalidates_cached_results(self):
        bs = evidence_behaviors()
        baseline = behavior_fingerprint(bs)
        changed_description = copy.deepcopy(bs)
        changed_description.by_id("smoking").evidence_checks["use"] = "嘴部必须清楚可见"
        changed_sequence = copy.deepcopy(bs)
        changed_sequence.by_id("littering").evidence_sequence.reverse()
        removed_checks = copy.deepcopy(bs)
        removed_checks.by_id("smoking").evidence_checks = {}
        for changed in (changed_description, changed_sequence, removed_checks):
            self.assertNotEqual(behavior_fingerprint(changed), baseline)
        reordered = copy.deepcopy(bs)
        reordered.by_id("smoking").evidence_checks = dict(reversed(
            list(reordered.by_id("smoking").evidence_checks.items())))
        self.assertEqual(behavior_fingerprint(reordered), baseline)


class EvidenceBackendTests(unittest.TestCase):
    def setUp(self):
        self.bs = evidence_behaviors()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.frames = []
        for i in range(4):
            path = Path(self.tmp.name) / f"f_{i:05d}.jpg"
            Image.new("RGB", (12, 8), (i * 50, 20, 30)).save(path)
            self.frames.append(str(path))

    def request(self, count=2, video="input.mp4"):
        return WindowRequest("w0", video, 0, 2, self.frames[:count],
                             labels=[f"t={i}" for i in range(count)], with_alert=True)

    def cloud_result(self, obj, *, count=2, stitch=False):
        backend = CloudBackend("https://unused.invalid", "fake", "not-a-key",
                               max_retries=1, stitch=stitch)
        response = {"choices": [{"message": {"content": json.dumps(obj, ensure_ascii=False)}}]}
        with patch.object(backend, "_post", return_value=response) as post, \
                patch("civility.vlm.time.sleep"):
            result = backend.infer(self.request(count), self.bs)
        self.assertEqual(post.call_count, 1, "不足证据应降级，不能反复要求模型补出证据")
        self.assertIsNone(result.error)
        self.assertIsInstance(result.parsed["violations"], dict)
        return result.parsed, post.call_args.args[0]

    def assert_suppressed(self, parsed):
        item = parsed["violations"]["smoking"]
        self.assertFalse(item["violation"])
        self.assertFalse(item["alert"])
        self.assertEqual(item["evidence_status"], "insufficient")
        self.assertTrue(item["evidence_issues"])

    def test_cloud_normalizes_legacy_and_out_of_bounds_claims(self):
        for evidence in (None, {"tobacco": 1, "use": 3},
                         {"tobacco": True, "use": 2}, {"tobacco": 0, "use": 2}):
            with self.subTest(evidence=evidence):
                parsed, _ = self.cloud_result(smoking_claim(evidence=evidence))
                self.assert_suppressed(parsed)

    def test_cloud_accepts_valid_source_frame_references(self):
        parsed, _ = self.cloud_result(smoking_claim(evidence={"tobacco": 1, "use": 2}))
        self.assertTrue(parsed["violations"]["smoking"]["violation"])
        self.assertTrue(parsed["violations"]["smoking"]["alert"])

    def test_stitched_cloud_uses_source_frame_count_not_image_count(self):
        parsed, payload = self.cloud_result(
            smoking_claim(evidence={"tobacco": 1, "use": 4}), count=4, stitch=True)
        self.assertTrue(parsed["violations"]["smoking"]["violation"])
        images = [part for part in payload["messages"][-1]["content"]
                  if part["type"] == "image_url"]
        self.assertEqual(len(images), 1, "本例必须真正走拼图分支")
        parsed, _ = self.cloud_result(
            smoking_claim(evidence={"tobacco": 1, "use": 5}), count=4, stitch=True)
        self.assert_suppressed(parsed)

    def test_cluster_readback_does_not_trust_old_or_out_of_bounds_claims(self):
        for evidence in (None, {"tobacco": 1, "use": 3}):
            with self.subTest(evidence=evidence):
                path = Path(self.tmp.name) / "results.jsonl"
                path.write_text(json.dumps({"window_id": "w0", "parsed": smoking_claim(
                    evidence=evidence, as_mapping=True)}) + "\n", encoding="utf-8")
                result = ClusterBackend(path).infer(self.request(), self.bs)
                self.assertIsNone(result.error)
                self.assert_suppressed(result.parsed)

    def test_mock_chain_remains_usable_without_inventing_extra_frames(self):
        bs = load_behaviors()
        mock = MockBackend(noise=0, latency=0, scene="littering")
        for count in (1, 2, 4):
            with self.subTest(count=count):
                result = mock.infer(self.request(count, "littering_pos.mp4"), bs)
                parsed = sanitize(result.parsed, bs, frame_count=count)
                item = parsed["violations"]["littering"]
                self.assertEqual(item["violation"], count > 1)
                if item["violation"]:
                    self.assertTrue(all(1 <= ref <= count for ref in item["evidence"].values()))


class EvidenceWebIngressTests(unittest.TestCase):
    def test_old_remote_claim_is_suppressed_in_results_and_real_sse(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = Path(tmp) / "f_00000.jpg"
            Image.new("RGB", (12, 8), "gray").save(frame)
            cfg = copy.deepcopy(S.STATE.cfg)
            cfg["paths"]["frames_dir"] = tmp
            cfg["live"].update(pipeline=False, motion_gate=False)
            cfg["aggregate"].update(open_risks=True, min_confidence=.45,
                                     min_hits=2, window_k=2)
            with patch.object(S.STATE, "cfg", cfg), \
                    patch.object(S.STATE, "behaviors", evidence_behaviors()), \
                    patch.object(S.STATE, "jobs", {}), \
                    patch.object(S.STATE, "uploads", {}), \
                    patch.object(S.STATE, "session", None):
                client = TestClient(S.create_app())
                try:
                    for mode in ("ai", "immediate", "smoothed"):
                        for as_mapping in (False, True):
                            with self.subTest(mode=mode, mapping=as_mapping):
                                cfg["aggregate"]["mode"] = mode
                                obj = smoking_claim(as_mapping=as_mapping)
                                class Session:
                                    state = "ready"

                                    def infer(self, *args, **kwargs):
                                        return {"ok": True, "valid": True, "error": None,
                                                "parsed": copy.deepcopy(obj), "raw": "{}"}

                                S.STATE.session = Session()
                                jid = f"legacy-{mode}-{int(as_mapping)}"
                                job = S.JobRecord(jid, "sample.mp4", cached_windows=[
                                    Window(f"w{i}", "sample.mp4", i * 2, i * 2 + 2,
                                           [str(frame)], [f"t={i * 2}"], idx0=0)
                                    for i in range(2)])
                                S.STATE.jobs[jid] = job
                                S.STATE.uploads[jid] = frame
                                S.run_job(jid)
                                self.assertEqual(job.state, "done", job.error)
                                self.assertEqual(job.alerts, [])
                                for record in job.window_results:
                                    self.assertFalse(record["violations"]["smoking"]["violation"])
                                    self.assertTrue(record["evidence_warnings"])
                                result = client.get(f"/api/job/{jid}").json()
                                self.assertEqual(result["alerts"], [])
                                stream = client.get(f"/api/job/{jid}/stream")
                                events = [json.loads(line[6:]) for line in stream.text.splitlines()
                                          if line.startswith("data: ")]
                                windows = [event for event in events if event["type"] == "window"]
                                self.assertEqual(len(windows), 2)
                                for event in windows:
                                    self.assertEqual(event["hits"], [])
                                    self.assertTrue(event["evidence_warnings"])
                                self.assertTrue(all(not event["alerts"] for event in events
                                                    if event["type"] == "alerts"))
                finally:
                    client.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
