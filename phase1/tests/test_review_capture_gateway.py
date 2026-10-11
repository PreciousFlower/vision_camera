"""Review regressions using the real probe code, without a camera or GPU."""
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cluster.vlm_gateway import Gateway
from webui import capture


class ResponseSocket:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def settimeout(self, *args):
        pass

    def sendall(self, *args):
        pass

    def recv(self, size):
        data, self.response = self.response[:size], self.response[size:]
        return data


class GatewayHealthTests(unittest.TestCase):
    def probe(self, body, status=200):
        encoded = body if isinstance(body, bytes) else json.dumps(body).encode()
        response = (f'HTTP/1.0 {status} fixture\r\nContent-Type: application/json\r\n'
                    'Connection: close\r\n\r\n').encode() + encoded
        with patch('cluster.vlm_gateway.socket.create_connection',
                   return_value=ResponseSocket(response)):
            return Gateway(0, [('127.0.0.1', 1234)])._probe(0)

    def test_loading_and_other_true_fields_do_not_mean_ready(self):
        self.assertFalse(self.probe({'ready': False, 'loading': True,
                                     'vision_cache': {'supported': True}}))

    def test_real_ready_boolean_is_required(self):
        self.assertTrue(self.probe({'ready': True, 'loading': False}))
        for body in ({'ready': 'true'}, {'ready': 1}, {'loading': True},
                     ['ready', True], b'not json: "ready" true'):
            with self.subTest(body=body):
                self.assertFalse(self.probe(body))

    def test_error_http_status_does_not_mean_ready(self):
        self.assertFalse(self.probe({'ready': True}, status=503))


class CaptureProbeIsolationTests(unittest.TestCase):
    def probe_with_late_tcp(self, udp_available):
        unblock_tcp = threading.Event()
        tcp_finished = threading.Event()

        class FakeCapture:
            def __init__(self, width, block=False):
                self.width, self.block = width, block

            def getBackendName(self):
                return 'fixture'

            def get(self, key):
                return self.width

            def read(self):
                if self.block:
                    unblock_tcp.wait(2)
                    self.block = False
                return True, object()

            def release(self):
                if self.width == 111:
                    tcp_finished.set()

        def open_fixture(spec, *, transport, **kwargs):
            if transport == 'tcp':
                return FakeCapture(111, True), 'rtsp', spec, ''
            if udp_available:
                return FakeCapture(222), 'rtsp', spec, ''
            return None, 'rtsp', spec, 'RTSP UDP source cannot connect'

        original_join = threading.Thread.join
        joins = 0

        def controlled_join(thread, timeout=None):
            nonlocal joins
            joins += 1
            original_join(thread, timeout)
            if joins == 2:
                # Finish late TCP after UDP has completed, before probe_source
                # reads UDP's result. Each attempt must retain its own result.
                unblock_tcp.set()
                self.assertTrue(tcp_finished.wait(1), 'TCP fixture did not finish')

        try:
            with patch.object(capture, 'open_source', side_effect=open_fixture), \
                    patch.object(threading.Thread, 'join', controlled_join):
                return capture.probe_source('rtsp://127.0.0.1/fixture', timeout_sec=.03)
        finally:
            unblock_tcp.set()
            tcp_finished.wait(1)

    def test_late_tcp_success_cannot_overwrite_udp_failure(self):
        result = self.probe_with_late_tcp(udp_available=False)
        self.assertFalse(result.ok)
        self.assertIn('UDP source cannot connect', result.error)

    def test_late_tcp_success_cannot_replace_udp_metadata(self):
        result = self.probe_with_late_tcp(udp_available=True)
        self.assertTrue(result.ok)
        self.assertEqual(result.transport, 'udp')
        self.assertEqual(result.width, 222)


if __name__ == '__main__':
    unittest.main()
