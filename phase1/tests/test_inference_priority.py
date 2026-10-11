"""Exercise the actual model scheduler without loading a model or using a GPU."""
from __future__ import annotations

from pathlib import Path
import concurrent.futures
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cluster import vlm_service as V


class InferencePriorityTests(unittest.TestCase):
    @staticmethod
    def submit(batcher, qa=False):
        return batcher.submit([], [], 768 if qa else 512, False, False, False,
                              qa={"question": "内容"} if qa else None)

    def test_queued_qa_does_not_block_detection(self):
        batcher = V.Batcher()
        with patch.object(V, "BATCH_WAIT_MS", 0):
            qa = batcher.submit([], [], 768, False, False, False, qa={"question": "内容"})
            detection = batcher.submit([], [], 512, False, False, False)
            selected = batcher._take_batch()
            self.assertEqual([it.future for it in selected], [detection],
                             "a queued video question must not delay new detection frames")
            self.assertEqual([it.future for it in batcher._take_batch()], [qa])

    def test_detection_batch_combines_requests_across_qa_queue_entries(self):
        batcher = V.Batcher()
        first = self.submit(batcher)
        qa = self.submit(batcher, qa=True)
        second = self.submit(batcher)
        with patch.object(V, "BATCH_WAIT_MS", 0):
            batch = batcher._take_batch()
        self.assertEqual([it.future for it in batch], [first, second])
        self.assertTrue(all(it.qa is None for it in batch))
        self.assertEqual([it.future for it in batcher.q], [qa])

    def test_aged_qa_gets_one_turn_then_detection_wins_again(self):
        batcher = V.Batcher(qa_max_wait_sec=20)
        with patch.object(V.time, "perf_counter", return_value=100), \
                patch.object(V, "BATCH_WAIT_MS", 0):
            first_qa = self.submit(batcher, qa=True)
            second_qa = self.submit(batcher, qa=True)
            detection = self.submit(batcher)
            batcher.q[0].queued_at = batcher.q[1].queued_at = 79
            self.assertEqual([it.future for it in batcher._take_batch()], [first_qa])
            self.assertEqual([it.future for it in batcher._take_batch()], [detection])
            self.assertEqual([it.future for it in batcher._take_batch()], [second_qa])
        self.assertEqual(batcher.stats["qa_wait_protected"], 1)

    def test_younger_qa_has_no_fairness_override(self):
        batcher = V.Batcher(qa_max_wait_sec=20)
        with patch.object(V.time, "perf_counter", return_value=100), \
                patch.object(V, "BATCH_WAIT_MS", 0):
            self.submit(batcher, qa=True)
            detection = self.submit(batcher)
            batcher.q[0].queued_at = 80.001
            self.assertEqual([it.future for it in batcher._take_batch()], [detection])

    def test_batch_limit_preserves_detection_fifo(self):
        batcher = V.Batcher()
        detection = [self.submit(batcher) for _ in range(5)]
        with patch.object(V, "BATCH_MAX", 2), patch.object(V, "BATCH_WAIT_MS", 0):
            self.assertEqual([it.future for it in batcher._take_batch()], detection[:2])
            self.assertEqual([it.future for it in batcher._take_batch()], detection[2:4])
            self.assertEqual([it.future for it in batcher._take_batch()], detection[4:])

    def test_cancelled_future_does_not_take_a_model_slot(self):
        batcher = V.Batcher()
        cancelled = self.submit(batcher)
        accepted = self.submit(batcher)
        self.assertTrue(cancelled.cancel())
        with patch.object(V, "BATCH_WAIT_MS", 0):
            self.assertEqual([it.future for it in batcher._take_batch()], [accepted])
        self.assertEqual(batcher.stats["cancelled"], 1)

    def test_cancellation_during_claim_still_skips_model_work(self):
        batcher = V.Batcher()
        abandoned = self.submit(batcher)
        accepted = self.submit(batcher)
        original_claim = abandoned.set_running_or_notify_cancel

        def cancel_at_claim():
            abandoned.cancel()
            return original_claim()

        with patch.object(abandoned, "set_running_or_notify_cancel", side_effect=cancel_at_claim), \
                patch.object(V, "BATCH_WAIT_MS", 0):
            self.assertEqual([it.future for it in batcher._take_batch()], [accepted])
        self.assertTrue(abandoned.cancelled())

    def test_expired_request_gets_timeout_without_model_work(self):
        batcher = V.Batcher()
        with patch.object(V.time, "perf_counter", return_value=100):
            expired = self.submit(batcher)
            accepted = self.submit(batcher)
            batcher.q[0].deadline = 99
            with patch.object(V, "BATCH_WAIT_MS", 0):
                self.assertEqual([it.future for it in batcher._take_batch()], [accepted])
        with self.assertRaises(concurrent.futures.TimeoutError):
            expired.result()
        self.assertEqual(batcher.stats["expired"], 1)

    def test_timed_out_waiter_cancels_only_pending_work(self):
        batcher = V.Batcher()
        pending = self.submit(batcher)
        with patch.object(V, "BATCH_TIMEOUT_SEC", 0):
            with self.assertRaises(concurrent.futures.TimeoutError):
                V.wait_for_result(pending)
        self.assertTrue(pending.cancelled())

    def test_timed_out_running_qa_is_not_interrupted(self):
        batcher = V.Batcher()
        qa = self.submit(batcher, qa=True)
        with patch.object(V, "BATCH_WAIT_MS", 0):
            self.assertEqual([it.future for it in batcher._take_batch()], [qa])
        with patch.object(V, "BATCH_TIMEOUT_SEC", 0):
            with self.assertRaises(concurrent.futures.TimeoutError):
                V.wait_for_result(qa)
        self.assertTrue(qa.running())
        qa.set_result({"answer": "正常完成"})
        self.assertEqual(qa.result(), {"answer": "正常完成"})

    def test_bounded_queue_rejects_and_reuses_cancelled_capacity(self):
        batcher = V.Batcher(max_queue=2)
        first = self.submit(batcher)
        self.submit(batcher)
        with self.assertRaises(V.QueueFullError):
            self.submit(batcher)
        first.cancel()
        self.submit(batcher)
        self.assertEqual(len(batcher.q), 2)
        self.assertEqual(batcher.stats["rejected"], 1)

    def test_qa_capacity_keeps_room_for_detection(self):
        batcher = V.Batcher(max_queue=4, max_qa_queue=1)
        self.submit(batcher, qa=True)
        with self.assertRaises(V.QueueFullError):
            self.submit(batcher, qa=True)
        detection = self.submit(batcher)
        self.assertEqual(batcher.snapshot()["queued_detection"], 1)
        self.assertEqual(batcher.snapshot()["queued_qa"], 1)
        with patch.object(V, "BATCH_WAIT_MS", 0):
            self.assertEqual([it.future for it in batcher._take_batch()], [detection])

    def test_capacity_stays_bounded_during_batch_collection(self):
        batcher = V.Batcher(max_queue=1)
        self.submit(batcher)
        rejected = []
        clock = [100.0]

        def add_during_wait(timeout):
            try:
                self.submit(batcher)
            except V.QueueFullError:
                rejected.append(True)
            clock[0] += 0.1

        with patch.object(V, "BATCH_WAIT_MS", 25), patch.object(V, "BATCH_MAX", 2), \
                patch.object(batcher.cv, "wait", side_effect=add_during_wait), \
                patch.object(V.time, "perf_counter", side_effect=lambda: clock[0]):
            # Queue timestamps originate from the real clock; only collection
            # time is pinned, so accepted requests do not accidentally expire.
            batch = batcher._take_batch()
        self.assertEqual(len(batch), 1)
        self.assertEqual(rejected, [True])

    def test_http_overload_and_timeout_are_explicit(self):
        import requests
        body = {"frames": [{"b64": "aA=="}], "evidence": [{"frame_id": "f0", "time_sec": 0}],
                "question": "画面内容？"}
        with patch.dict(V.STATE, ready=True, token=None), patch.object(V, "BATCHER") as batcher:
            http = ThreadingHTTPServer(("127.0.0.1", 0), V.Handler)
            thread = threading.Thread(target=http.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{http.server_port}"
            try:
                batcher.submit.side_effect = V.QueueFullError("推理队列已满")
                for path in ("/infer", "/video-qa"):
                    response = requests.post(base + path, json=body, timeout=5)
                    self.assertEqual(response.status_code, 503)
                    self.assertTrue(response.json()["retryable"])
                batcher.submit.side_effect = None
                for path in ("/infer", "/video-qa"):
                    pending = concurrent.futures.Future()
                    batcher.submit.return_value = pending
                    with patch.object(V, "BATCH_TIMEOUT_SEC", 0):
                        response = requests.post(base + path, json=body, timeout=5)
                    self.assertEqual(response.status_code, 504)
                    self.assertTrue(pending.cancelled())
            finally:
                http.shutdown()
                http.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
