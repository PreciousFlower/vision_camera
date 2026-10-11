"""Do not submit duplicate GPU work while the service is busy or timed out."""
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from webui.cluster_client import ClusterError, ServiceSession


class InferenceClientBusyTests(unittest.TestCase):
    def setUp(self):
        self.session = ServiceSession.__new__(ServiceSession)
        self.session.base_url = lambda: "http://worker.test"

    @staticmethod
    def response(status):
        return types.SimpleNamespace(status_code=status, text="service busy",
            json=lambda: {"parsed": {}, "error": None})

    def test_full_queue_is_reported_without_automatic_resubmission(self):
        with patch("requests.post", side_effect=[self.response(503), self.response(200)]) as send, \
                patch("webui.cluster_client.time.sleep"):
            with self.assertRaises(ClusterError):
                self.session.infer(["image"], ["t=1s"])
        self.assertEqual(send.call_count, 1)

    def test_timed_out_gpu_request_is_not_submitted_twice(self):
        with patch("requests.post", side_effect=[self.response(504), self.response(200)]) as send, \
                patch("webui.cluster_client.time.sleep"):
            with self.assertRaises(ClusterError):
                self.session.infer(["image"], ["t=1s"])
        self.assertEqual(send.call_count, 1)

    def test_transient_gateway_failure_keeps_existing_retry(self):
        with patch("requests.post", side_effect=[self.response(502), self.response(200)]) as send, \
                patch("webui.cluster_client.time.sleep"):
            result = self.session.infer(["image"], ["t=1s"])
        self.assertEqual(send.call_count, 2)
        self.assertIsNone(result["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
