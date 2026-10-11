"""Video QA references, persisted history and failure boundaries; no paid GPU."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from civility.video_qa import build_qa_messages, validate_answer, validate_request
from cluster import vlm_service as V
from webui import server as S
from webui.video_library import VideoLibrary


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.evidence = [{"frame_id": "f_0", "time_sec": 0.0}]
        self.body = {"frames": [{"b64": "aA=="}], "evidence": self.evidence, "question": "哪一帧吸烟？"}

    def test_request_keeps_question_and_image_evidence(self):
        qa = validate_request(self.body)
        image = object()
        messages = build_qa_messages([image], qa)
        self.assertIs(messages[1]["content"][-1]["image"], image)
        self.assertIn("哪一帧吸烟", messages[1]["content"][0]["text"])
        self.assertIn("抽样帧", messages[0]["content"])

    def test_invalid_requests_are_rejected(self):
        for change in ({"question": " "}, {"question": 123}, {"question": "a" * 2001},
                       {"frames": []}, {"frames": [{}]}, {"evidence": []}, {"context": "x" * 24001}):
            with self.subTest(change=list(change)):
                with self.assertRaises(ValueError):
                    validate_request(self.body | change)

    def test_answer_rejects_nonexistent_or_unverifiable_citations(self):
        for refs in ([], [{"frame_id": "f_999", "description": "吸烟"}], [{"frame_id": "f_0"}], ["f_0"], [{"frame_id": [], "description": "x"}]):
            with self.assertRaises(ValueError):
                validate_answer({"answer": "有人吸烟", "citations": refs}, self.evidence)

    def test_answer_deduplicates_refs_and_ignores_invented_timestamps(self):
        ref = {"frame_id": "f_0", "description": "手持物品，无法确认吸烟", "time_sec": 999}
        result = validate_answer({"answer": "无法确认", "citations": [ref, ref]}, self.evidence)
        self.assertEqual(result["citations"], [{"frame_id": "f_0", "description": ref["description"]}])

    def test_batcher_separates_long_qa_from_detection(self):
        batcher = V.Batcher()
        with patch.object(V, "BATCH_WAIT_MS", 0):
            detection = batcher.submit([], [], 512, False, False, False)
            qa = batcher.submit([], [], 768, False, False, False, qa={"question": "内容"})
            later = batcher.submit([], [], 512, False, False, False)
            self.assertEqual([it.future for it in batcher._take_batch()], [detection, later])
            self.assertEqual([it.future for it in batcher._take_batch()], [qa])

    def test_model_http_qa_health_and_schema_boundaries(self):
        import requests
        with patch.dict(V.STATE, ready=True, model_name="deployed", behaviors=None, token=None), \
                patch.object(V.BATCHER, "submit") as submit:
            submit.return_value.result.return_value = {"parsed": {"answer": "回答", "citations": []}, "error": None}
            http = ThreadingHTTPServer(("127.0.0.1", 0), V.Handler)
            thread = threading.Thread(target=http.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{http.server_port}"
            try:
                self.assertEqual(requests.get(base + "/health", timeout=5).json()["video_qa_protocol"], 1)
                response = requests.post(base + "/video-qa", json=self.body, timeout=5)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(submit.call_args.kwargs["qa"]["question"], self.body["question"])
                for change in ({"frames": []}, {"max_new_tokens": True}, {"max_new_tokens": 10000}):
                    self.assertEqual(requests.post(base + "/video-qa", json=self.body | change, timeout=5).status_code, 400)
                with patch.dict(V.STATE, ready=False):
                    self.assertEqual(requests.post(base + "/video-qa", json=self.body, timeout=5).status_code, 503)
            finally:
                http.shutdown()
                http.server_close()
                thread.join(timeout=5)


class LibraryApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.frames = self.root / "frames"
        self.uploads = self.root / "uploads"
        self.uploads.mkdir()
        self.library = VideoLibrary(self.root / "analyses", self.frames, self.uploads)
        self.saved = S.STATE.jobs, S.STATE.session
        S.STATE.jobs = {}
        self.session = types.SimpleNamespace(state="ready", model="deployed-fixture", ask_video=Mock())
        S.STATE.session = self.session
        self.patch = patch.object(S, "video_library", return_value=self.library)
        self.patch.start()
        self.runtime = patch.object(S, "ensure_current_runtime")
        self.runtime.start()
        self.client = TestClient(S.create_app())
        self.job = S.JobRecord("qa-fixture", "fixture.mp4", state="done", windows_seen=1)
        self.job._archive_source = self.uploads / "fixture.mp4"
        self.job._archive_source.write_bytes(b"original fixture video")
        self.row = {"window_id": "w0", "t_start": 0, "t_end": 3, "frame_indices": [0, 9, 17],
                    "frame_labels": ["t=0.0s", "t=3.0s", "t=5.7s"], "frame_times": [0, 3, 17/3],
                    "scene": "手持物品", "violations": {"smoking": {"violation": True, "evidence": {"cigarette": 2}}}}
        self.library.begin(self.job)
        self.library.record(self.job, self.row)
        self.library.save(self.job)
        directory = self.frames / self.job.job_id
        directory.mkdir(parents=True)
        for index in self.row["frame_indices"]:
            (directory / f"f_{index:05d}.jpg").write_bytes(f"real frame {index}".encode())
        self.session.ask_video.side_effect = lambda frames, question, evidence, context: {
            "parsed": {"answer": "请看引用画面", "citations": [{"frame_id": evidence[0]["frame_id"],
                       "description": "手持物品"}], "limitations": "抽样画面"}, "model": "deployed-fixture"}

    def tearDown(self):
        self.client.close()
        self.patch.stop()
        self.runtime.stop()
        S.STATE.jobs, S.STATE.session = self.saved
        self.temp.cleanup()

    def ask(self, body=None):
        return self.client.post("/api/videos/qa-fixture/ask", json=body or {"question": "哪一帧吸烟？"})

    def test_history_survives_in_memory_job_loss_and_hides_local_paths(self):
        result = self.client.get("/api/videos").json()["videos"]
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["video_id"], self.job.job_id)
        self.assertNotIn("source_path", result[0])
        self.assertTrue(result[0]["source_available"])

    def test_api_uses_actual_images_and_deployed_model_with_trusted_refs(self):
        result = self.ask()
        self.assertEqual(result.status_code, 200, result.text)
        answer = result.json()
        self.assertEqual(answer["basis"], "deployed_vlm_visual_qa")
        self.assertEqual(answer["model"], "deployed-fixture")
        reference = answer["citations"][0]
        self.assertEqual(reference["time_sec"], 0)
        self.assertEqual(reference["sample_number"], 1)
        self.assertNotIn("path", reference)
        frames = self.session.ask_video.call_args.args[0]
        self.assertEqual(base64.b64decode(frames[0]), b"real frame 0")
        response = self.client.get(reference["image_url"])
        self.assertEqual(response.content, b"real frame 0")

    def test_hallucinated_refs_are_rejected_at_local_ingress(self):
        self.session.ask_video.side_effect = None
        self.session.ask_video.return_value = {"parsed": {"answer": "吸烟", "citations": [{"frame_id": "f_999", "description": "吸烟"}]}}
        self.assertEqual(self.ask().status_code, 422)

    def test_unready_model_and_invalid_question_do_not_invoke_model(self):
        S.STATE.session = None
        self.assertEqual(self.ask().status_code, 503)
        self.assertEqual(self.ask({"question": " "}).status_code, 400)
        self.session.ask_video.assert_not_called()

    def test_active_analysis_is_not_queried(self):
        self.job.state = "running"
        S.STATE.jobs[self.job.job_id] = self.job
        self.assertEqual(self.ask().status_code, 409)
        self.session.ask_video.assert_not_called()

    def test_only_one_question_can_use_model_and_watchdog_does_not_release(self):
        with S.QA_BUSY:
            self.assertEqual(self.ask().status_code, 429)
            self.assertTrue(S._active_job())
        self.assertFalse(S.QA_BUSY.locked())

    def test_question_failure_releases_slot(self):
        self.session.ask_video.side_effect = S.ClusterError("service too old")
        self.assertEqual(self.ask().status_code, 502)
        self.assertFalse(S.QA_BUSY.locked())

    def test_reanalysis_invalidates_citations(self):
        first = self.ask().json()["citations"][0]["image_url"]
        self.library.begin(self.job)
        self.library.record(self.job, self.row)
        self.library.save(self.job)
        self.assertEqual(self.client.get(first).status_code, 409)

    def test_missing_frames_and_original_give_explicit_error(self):
        self.job._archive_source.unlink()
        for frame in (self.frames / self.job.job_id).glob("*.jpg"):
            frame.unlink()
        response = self.ask()
        self.assertEqual(response.status_code, 422)
        self.assertIn("候选证据帧", response.text)
        self.session.ask_video.assert_not_called()

    def test_frame_can_be_recovered_from_original_at_archived_timestamp(self):
        (self.frames / self.job.job_id / "f_00017.jpg").unlink()
        def recover(command, **kwargs):
            self.assertAlmostEqual(float(command[command.index("-ss") + 1]), 17/3)
            Path(command[-1]).write_bytes(b"recovered at exact timestamp")
        with patch("webui.video_library.subprocess.run", side_effect=recover):
            path, info = self.library.frame(self.job.job_id, "f_17")
        self.assertTrue(info["recovered"])
        self.assertEqual(path.read_bytes(), b"recovered at exact timestamp")
        self.assertTrue(path.is_relative_to(self.frames))
        self.job._archive_source.unlink()
        self.assertEqual(self.library.frame(self.job.job_id, "f_17")[0], path)

    def test_lookup_cannot_expose_unregistered_frames_or_source_outside_uploads(self):
        self.assertEqual(self.client.get("/api/videos/qa-fixture/frames/f_999").status_code, 404)
        self.assertIsNone(self.library.source({"source_path": str(Path(__file__).resolve())}))
        with self.assertRaises(ValueError):
            self.library.folder("../secret")

    def test_partial_final_journal_line_does_not_hide_history(self):
        path = self.library.folder(self.job.job_id) / "windows.jsonl"
        with path.open("a", encoding="utf-8") as out:
            out.write('{"broken":')
        self.assertEqual(len(self.library.windows(self.job.job_id)), 1)

    def test_retrieval_prefers_relevant_action_and_keeps_temporal_endpoints(self):
        normal = self.row | {"window_id": "w-normal", "t_start": 30, "t_end": 33, "scene": "正常", "violations": {}}
        self.library.record(self.job, normal)
        evidence, context, _ = self.library.retrieve(self.job.job_id, "吸烟发生在哪里")
        self.assertEqual([e["frame_id"] for e in evidence], ["f_0", "f_9", "f_17"])
        self.assertNotIn("w-normal", context)


if __name__ == "__main__":
    unittest.main()
