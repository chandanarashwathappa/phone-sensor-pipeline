"""Tests PhyphoxSource against a fake phyphox HTTP server (same JSON format)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest

from sensor_sources import PhyphoxSource

FS = 100.0
T = np.arange(0, 2, 1 / FS)


class FakePhyphox(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/config":
            return self._send({"buffers": [{"name": n} for n in ("acc_time", "accX", "accY", "accZ")]})
        if u.path == "/control":
            return self._send({"result": True})
        q = parse_qs(u.query)
        since = float(q["acc_time"][0])
        mask = T > since
        buf = {"acc_time": {"buffer": T[mask].tolist()}}
        for i, ch in enumerate(("accX", "accY", "accZ")):
            vals = np.sin(2 * np.pi * 10 * T[mask] + i).tolist()
            buf[ch] = {"buffer": vals}
        self._send({"buffer": buf, "status": {"measuring": True}})


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), FakePhyphox)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_reads_buffers_and_polls_incrementally(server):
    src = PhyphoxSource(server)
    assert "accX" in src.buffer_names()
    first = src.read_new()
    assert len(first) == len(T) - 1  # strictly after t=0
    assert set(first.channels) == {"accX", "accY", "accZ"}
    second = src.read_new()
    assert len(second) == 0  # nothing new since last timestamp
