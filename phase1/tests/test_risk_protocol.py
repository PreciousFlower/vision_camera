"""开放风险请求贯穿客户端、HTTP、批量和 CLI；全部离线验证。"""
from __future__ import annotations

import concurrent.futures
import contextlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from cluster import run_cluster_vlm, vlm_service  # noqa: E402
from civility.cli import _request_options  # noqa: E402
from civility.config import Behavior, BehaviorSet  # noqa: E402
from civility.extract import Window  # noqa: E402
from civility.runner import behavior_fingerprint, run_windows  # noqa: E402
from civility.vlm import WindowRequest, WindowResult  # noqa: E402
from webui.cluster_client import ServiceSession  # noqa: E402


BS = BehaviorSet([Behavior("smoking", "吸烟", "可见烟雾与烟支")], {})


class RecordingProcessor:
    def __init__(self):
        self.tokenizer = types.SimpleNamespace(padding_side="right")
        self.messages = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(messages)
        return messages[0]["content"]

    def __call__(self, **kwargs):
        return kwargs


class TestRiskProtocol(unittest.TestCase):
    def fake_service_batch(self, objects, options=None, previous_latencies=None, token_rows=None):
        """执行真实结果解析与校验，只有张量生成由本地假对象替代。"""
        class Inputs(dict):
            input_ids = types.SimpleNamespace(shape=(len(objects), 3))

            def to(self, device):
                return self

        class Generated:
            shape = (len(objects), max(map(len, token_rows)) if token_rows else 2)

            def __getitem__(self, key):
                if isinstance(key, int) and token_rows is not None:
                    return types.SimpleNamespace(tolist=lambda: token_rows[key])
                return self

        model = types.SimpleNamespace(device="fake", generate=lambda **kwargs: Generated())
        outputs = [json.dumps(obj, ensure_ascii=False) for obj in objects]
        processor = types.SimpleNamespace(batch_decode=lambda *args, **kwargs: outputs,
                                          tokenizer=types.SimpleNamespace(eos_token_id=9))
        fake_torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
        options = options if options is not None else [False] * len(objects)
        items = [vlm_service._Item([], [], 512, False, False, False,
                                  concurrent.futures.Future(), enabled)
                 for enabled in options]
        with patch.dict(sys.modules, {"torch": fake_torch}), \
                patch.dict(vlm_service.STATE, {"ready": True, "behaviors": BehaviorSet([], {}),
                                               "model": model, "processor": processor,
                                               "latencies": list(previous_latencies or [])}), \
                patch.object(vlm_service, "_build_inputs", return_value=Inputs()):
            results = vlm_service.infer_batch(items)
            latencies = list(vlm_service.STATE["latencies"])
        return results, latencies

    def test_batch_padding_does_not_inflate_tokens_and_truncation_is_explicit(self):
        results, _ = self.fake_service_batch(
            [{'scene': '走廊', 'violations': [], 'risks': []}, 'cut output'],
            options=[True, True], token_rows=[[1, 2, 9] + [9] * 509, [1] * 512])
        self.assertEqual(results[0]['tokens_out'], 3)
        self.assertEqual(results[0]['finish_reason'], 'stop')
        self.assertEqual(results[1]['tokens_out'], 512)
        self.assertEqual(results[1]['finish_reason'], 'length')
        self.assertIsNone(results[1]['parsed'])

    def test_client_sends_explicit_flags_and_old_defaults(self):
        session = ServiceSession(types.SimpleNamespace(), model="fake")
        response = types.SimpleNamespace(status_code=200, json=lambda: {"ok": True})
        with patch.object(session, "base_url", return_value="http://local.test"), \
                patch("requests.post", return_value=response) as post:
            session.infer(["image"], ["frame"])
            payload = post.call_args.kwargs["json"]
            self.assertFalse(payload["with_risks"])
            self.assertFalse(payload["with_actions"])
            session.infer(["image"], ["frame"], with_alert=True,
                          with_actions=True, with_risks=True)
            payload = post.call_args.kwargs["json"]
            self.assertEqual(payload["frames"], [{"b64": "image"}])
            self.assertTrue(all(payload[key] for key in
                                ("with_alert", "with_actions", "with_risks")))

    def test_http_route_preserves_options_and_defaults(self):
        for requested in (False, True):
            with self.subTest(requested=requested):
                handler = object.__new__(vlm_service.Handler)
                handler.path = "/infer"
                handler.headers = {}
                body = {"frames": [{"b64": "image"}], "with_alert": True,
                        "with_actions": True}
                if requested:
                    body["with_risks"] = True
                replies = []
                handler._read_json = lambda: body
                handler._json = lambda code, data: replies.append((code, data))
                with patch.dict(vlm_service.STATE, {"ready": True, "token": None}), \
                        patch.object(vlm_service, "infer_one",
                                     return_value={"error": None}) as infer:
                    handler.do_POST()
                self.assertEqual(infer.call_args.args[4:], (True, True, requested))
                self.assertEqual(replies[0][0], 200)

    def test_infer_one_keeps_risk_flag_in_batch_queue(self):
        future = concurrent.futures.Future()
        future.set_result({"parsed": {"risks": []}})
        with patch.object(vlm_service.BATCHER, "submit", return_value=future) as submit:
            result = vlm_service.infer_one([], [], 512, False,
                                          with_alert=True, with_actions=True,
                                          with_risks=True)
        self.assertEqual(submit.call_args.args[4:], (True, True, True))
        self.assertIn("risks", result["parsed"])

    def test_batch_items_keep_individual_flags(self):
        batcher = vlm_service.Batcher()
        batcher.submit([], [], 512, False, True, False)
        batcher.submit([], [], 512, False, True, True, with_risks=True)
        self.assertFalse(batcher.q[0].with_risks)
        self.assertTrue(batcher.q[1].with_risks)
        self.assertTrue(batcher.q[1].with_actions)

    def test_batch_prompt_and_processor_fallback_receive_real_images(self):
        batcher = vlm_service.Batcher()
        batcher.submit([{}], ["frame"], 512, False, True, True, True)
        batcher.submit([{}], ["frame"], 512, False, False, False)
        processor = RecordingProcessor()
        decoded = [object(), object()]
        with patch.dict(vlm_service.STATE, {"behaviors": BS, "processor": processor,
                                           "has_qvu": False}), \
                patch.object(vlm_service, "decode_images", side_effect=[[decoded[0]],
                                                                        [decoded[1]]]):
            inputs = vlm_service._build_inputs(batcher.q)
        self.assertIn('"risks"', processor.messages[0][0]["content"])
        self.assertNotIn('"risks"', processor.messages[1][0]["content"])
        self.assertEqual(inputs["images"], decoded)
        self.assertEqual(processor.tokenizer.padding_side, "left")

    def test_pipeline_passes_all_options_and_remains_compatible(self):
        class Backend:
            name = "fake"

            def __init__(self):
                self.requests = []

            def infer(self, request, behaviors):
                self.requests.append(request)
                return WindowResult(request.window_id, request.video,
                                    request.t_start, request.t_end, parsed={})

        window = Window("w", "v.mp4", 0, 2, [], [])
        with tempfile.TemporaryDirectory() as tmp:
            backend = Backend()
            run_windows([window], backend, BS, Path(tmp) / "new.jsonl",
                        verbose=False, with_alert=True, with_actions=True,
                        with_risks=True)
            request = backend.requests[-1]
            self.assertTrue(request.with_alert and request.with_actions
                            and request.with_risks)
            run_windows([window], backend, BS, Path(tmp) / "legacy.jsonl",
                        verbose=False)
            request = backend.requests[-1]
            self.assertFalse(request.with_alert or request.with_actions
                             or request.with_risks)
        self.assertFalse(WindowRequest("w", "v", 0, 2, []).with_risks)

    def test_cli_setting_and_cache_fingerprint_follow_requested_schema(self):
        self.assertTrue(_request_options({})["with_risks"])
        self.assertEqual(_request_options({"aggregate": {"mode": "ai",
                                                          "open_risks": False}}),
                         {"with_alert": True, "with_actions": False,
                          "with_risks": False})
        self.assertFalse(_request_options({"aggregate": {"mode": "immediate"}})
                         ["with_alert"])
        old = behavior_fingerprint(BS)
        new = behavior_fingerprint(BS, with_risks=True)
        self.assertNotEqual(old, new)
        self.assertNotEqual(new, behavior_fingerprint(BS, with_risks=True,
                                                     with_alert=True))

    def test_offline_job_package_options_and_old_package_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "behaviors.json"
            path.write_text(json.dumps({"behaviors": []}), encoding="utf-8")
            self.assertFalse(any(run_cluster_vlm.load_request_options(path).values()))
            path.write_text(json.dumps({"request_options": {"with_risks": True,
                                                             "with_alert": True}}),
                            encoding="utf-8")
            self.assertEqual(run_cluster_vlm.load_request_options(path),
                             {"with_alert": True, "with_actions": False,
                              "with_risks": True})

    def test_long_running_service_keeps_only_recent_latency_samples(self):
        results, latencies = self.fake_service_batch(
            [{"scene": "无人", "violations": [], "risks": []}],
            previous_latencies=[1.0] * 50)
        self.assertEqual(len(latencies), 50)
        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0]["error"])

    def test_invalid_risks_report_failure_and_keep_scene_for_diagnosis(self):
        objects = [{"scene": "有人倒地", "violations": [], "risks": {}},
                   {"scene": "有人倒地", "violations": [], "risks": [{
                       "risk_id": "fall", "name": "人员倒地", "severity": "high",
                       "confidence": .95, "observable": "人倒在地面", "alert": "true",
                       "alert_reason": "需现场确认", }]}]
        results, _ = self.fake_service_batch(objects, options=[True, True])
        for result in results:
            self.assertFalse(result["valid"])
            self.assertIn("Schema", result["error"])
            self.assertTrue(result["problems"])
            self.assertEqual(result["parsed"]["scene"], "有人倒地")
            self.assertEqual(result["parsed"]["risks"], [])

    def test_missing_risks_fails_only_for_requests_that_asked_for_it(self):
        obj = {"scene": "走廊", "violations": []}
        results, _ = self.fake_service_batch([obj, obj], options=[False, True])
        self.assertTrue(results[0]["valid"])
        self.assertIsNone(results[0]["error"])
        self.assertFalse(results[1]["valid"])
        self.assertIn("risks", results[1]["error"])

    def test_explicit_empty_risks_is_valid_checked_scene(self):
        results, _ = self.fake_service_batch(
            [{"scene": "无人", "violations": [], "risks": []}], options=[True])
        self.assertTrue(results[0]["valid"])
        self.assertIsNone(results[0]["error"])
        self.assertEqual(results[0]["parsed"]["risks"], [])


if __name__ == "__main__":
    unittest.main()
