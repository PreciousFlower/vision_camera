"""Review regressions for starting an uploaded file; no model or camera."""
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from webui import server as S


class DetectStartTests(unittest.TestCase):
    def setUp(self):
        self.saved = (S.STATE.jobs, S.STATE.session)
        S.STATE.jobs = {}
        S.STATE.session = types.SimpleNamespace(state="ready")
        self.client = TestClient(S.create_app())
        self.client.__enter__()
        self.runtime = patch.object(S, "ensure_current_runtime")
        self.runtime.start()

    def tearDown(self):
        self.runtime.stop()
        self.client.__exit__(None, None, None)
        S.STATE.jobs, S.STATE.session = self.saved

    def job(self, state="queued"):
        job = S.JobRecord("start-review", "fixture.mp4", state=state,
                          use_mock=False, with_actions=True, loop=False)
        S.STATE.jobs[job.job_id] = job
        return job

    def start(self, **body):
        return self.client.post("/api/detect/start-review", json={'profile_id':'current',**body})

    def test_rejected_duplicate_start_preserves_active_settings(self):
        for state in ("running", "extracting"):
            with self.subTest(state=state):
                job = self.job(state)
                response = self.start(use_mock=True, with_actions=False,
                                      mock_scene="smoking", loop=True)
                self.assertEqual(response.status_code, 400)
                self.assertFalse(job.use_mock)
                self.assertTrue(job.with_actions)
                self.assertFalse(job.loop)
                self.assertEqual(job.mock_scene, "")
                self.assertEqual(job.state, state)

    def test_pending_start_is_reserved_but_new_upload_can_start(self):
        self.job()
        worker = Mock()
        with patch.object(S, "threading", types.SimpleNamespace(Thread=Mock(return_value=worker))):
            first = self.start(use_mock=True)
            second = self.start(use_mock=True)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 400)
        self.assertEqual(worker.start.call_count, 1)

    def test_terminal_restart_resets_old_error_and_clears_same_event_queue(self):
        job = self.job("error")
        job.error = "previous inference failed"
        job.avg_latency = 17
        job.windows_done = 8
        job.windows_total = 8
        job.events.put({"type": "error", "error": job.error})
        old_events = job.events
        with patch.object(S, "threading", types.SimpleNamespace(Thread=Mock())):
            response = self.start(use_mock=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(job.state, "queued")
        self.assertEqual(job.error, "")
        self.assertEqual(job.avg_latency, 0)
        self.assertEqual(job.windows_done, 0)
        self.assertEqual(job.windows_total, 0)
        self.assertIs(job.events, old_events)
        self.assertTrue(job.events.empty())

    def test_first_start_keeps_preexisting_subscriber_queue(self):
        job = self.job()
        subscribed_queue = job.events
        with patch.object(S, "threading", types.SimpleNamespace(Thread=Mock())):
            response = self.start(use_mock=True)
        self.assertEqual(response.status_code, 200)
        self.assertIs(job.events, subscribed_queue)
        job.events.put({"type": "window", "window_id": "first"})
        self.assertEqual(subscribed_queue.get_nowait()["window_id"], "first")

    def test_new_upload_with_unready_service_is_unchanged(self):
        job = self.job()
        S.STATE.session = None
        response = self.start(with_actions=False, loop=True)
        self.assertEqual(response.status_code, 400)
        self.assertTrue(job.with_actions)
        self.assertFalse(job.loop)
        self.assertEqual(job.state, "queued")

    def test_failed_thread_start_does_not_reserve_job_forever(self):
        job = self.job()
        worker = Mock()
        worker.start.side_effect = RuntimeError("thread unavailable")
        worker.is_alive.return_value = False
        with patch.object(S, "threading", types.SimpleNamespace(Thread=Mock(return_value=worker))):
            response = self.start(use_mock=True)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(job.state, "error")
        self.assertIn("thread unavailable", job.error)
        self.assertFalse(job._detect_start_reserved)
        with patch.object(S, "threading", types.SimpleNamespace(Thread=Mock())):
            response = self.start(use_mock=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(job.error, "")


if __name__ == "__main__":
    unittest.main()
