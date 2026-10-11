"""Exercise actual SSH forwarding pump with local sockets and backpressure."""
from pathlib import Path
import socket
import sys
import threading
import time
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from webui.cluster_client import Tunnel


class TunnelBackpressureTests(unittest.TestCase):
    def setUp(self):
        self.local, forwarded = socket.socketpair()
        channel, self.backend = socket.socketpair()
        self.sockets = [self.local, forwarded, channel, self.backend]
        self.local.settimeout(3)
        self.backend.settimeout(3)
        forwarded.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)
        self.tunnel = Tunnel(None, 'fixture', 18080)
        self.tunnel._running = True
        self.tunnel._transport = types.SimpleNamespace(open_channel=lambda *args: channel)
        self.channel = channel
        self.thread = threading.Thread(target=self.tunnel._pump,
                                       args=(forwarded, ('127.0.0.1', 0)), daemon=True)
        self.thread.start()
        self.producers = []

    def tearDown(self):
        self.tunnel._running = False
        for sock in self.sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        self.thread.join(1)
        for producer in self.producers:
            producer.join(1)

    def start_producer(self, operation):
        errors = []
        def produce():
            try:
                operation()
            except OSError as exc:
                errors.append(exc)
        producer = threading.Thread(target=produce, daemon=True)
        self.producers.append(producer)
        producer.start()
        return producer, errors

    @staticmethod
    def read_response(sock):
        chunks = []
        while True:
            chunk = sock.recv(8192)
            if not chunk:
                return b''.join(chunks)
            chunks.append(chunk)

    def test_large_response_survives_slow_local_receiver(self):
        response = bytes(range(256)) * 4096
        def send_response():
            self.backend.sendall(response)
            self.backend.shutdown(socket.SHUT_WR)
        producer, errors = self.start_producer(send_response)
        time.sleep(.05)  # Fill the forwarder's send buffer before consuming it.
        received = self.read_response(self.local)
        producer.join(1)
        self.thread.join(1)
        self.assertEqual(received, response, self.tunnel.last_error)
        self.assertFalse(errors)
        self.assertFalse(self.thread.is_alive())

    def test_request_and_response_both_arrive_complete(self):
        request = bytes(range(255, -1, -1)) * 4096
        response = b'{"frame":"' + b'x' * 1048576 + b'"}'
        requests = []
        def exchange():
            pieces = []
            remaining = len(request)
            while remaining:
                chunk = self.backend.recv(min(65536, remaining))
                if not chunk:
                    break
                pieces.append(chunk)
                remaining -= len(chunk)
            requests.append(b''.join(pieces))
            self.backend.sendall(response)
            self.backend.shutdown(socket.SHUT_WR)
        producer, errors = self.start_producer(exchange)
        self.local.sendall(request)
        time.sleep(.05)
        received = self.read_response(self.local)
        producer.join(1)
        self.thread.join(1)
        self.assertEqual(requests, [request])
        self.assertEqual(received, response, self.tunnel.last_error)
        self.assertFalse(errors)
        self.assertFalse(self.thread.is_alive())

    def test_client_disconnect_releases_both_connection_sockets(self):
        self.local.close()
        self.thread.join(1)
        self.assertFalse(self.thread.is_alive())
        self.assertEqual(self.channel.fileno(), -1)
        self.assertEqual(self.sockets[1].fileno(), -1)


if __name__ == '__main__':
    unittest.main()
