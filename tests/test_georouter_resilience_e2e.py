"""E2E tests for BridgeServer resilience: metadata singleflight, breaker floor 503, 2-proxy failover (Wave 8)."""

from __future__ import annotations

import concurrent.futures
import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from model_georouter import bridge_server
from model_georouter.bridge_server import BridgeServer
from model_georouter.config import ConfigManager


def _http_get(port: int, path: str, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", path, headers=headers or {})
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def test_bridge_metadata_cache_singleflight_and_breaker_floor_503(tmp_path) -> None:
    hits = {"models": 0, "fail": False}
    lock = threading.Lock()

    class _Up(BaseHTTPRequestHandler):
        def log_message(self, *_a):
            pass

        def do_GET(self):
            with lock:
                hits["models"] += 1
                fail = hits["fail"]
            if fail:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "15")
                self.end_headers()
                self.wfile.write(b'{"error":"bad"}')
                return
            auth = self.headers.get("Authorization", "")
            tag = "a" if "alpha" in auth else "b"
            payload = json.dumps({"data": [{"id": f"m-{tag}"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    up = ThreadingHTTPServer(("127.0.0.1", 0), _Up)
    up_port = up.server_address[1]
    threading.Thread(target=up.serve_forever, daemon=True).start()

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps({
            "server": {"listen": "127.0.0.1", "port": 0, "total_egress_deadline": 40, "log_dir": str(tmp_path / "logs"), "f08_policy": "v2", "f08_v2_risk_ack": "AD-003/2026-11-14"},
            "pools": {"direct": {"type": "direct"}},
            "rules": [],
            "default_pool": "direct",
            "upstreams": {"mock": {"host": "127.0.0.1", "port": up_port, "use_tls": False, "base_path": "/v1"}},
            "adapters": {"m": {"prefixes": ["/m/v1"], "upstream": "mock", "pool": "direct"}},
        }),
        encoding="utf-8",
    )
    bridge_server._BREAKERS.clear()
    srv = BridgeServer(ConfigManager(cfg_path))
    bport = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            fa = [ex.submit(_http_get, bport, "/m/v1/models", {"Authorization": "Bearer alpha"}) for _ in range(5)]
            fb = [ex.submit(_http_get, bport, "/m/v1/models", {"Authorization": "Bearer beta"}) for _ in range(5)]
            ra = [f.result() for f in fa]
            rb = [f.result() for f in fb]
        assert hits["models"] <= 2
        assert all(st == 200 and b"m-a" in body for st, body in ra)
        assert all(st == 200 and b"m-b" in body for st, body in rb)

        # Trip route circuit breaker (3x 500 on distinct uncached keys) -> 4th request fast-fails 503
        hits["fail"] = True
        for i in range(3):
            st, _ = _http_get(bport, "/m/v1/models", {"Authorization": f"Bearer trip-{i}"})
            assert st == 500
        before_hits = hits["models"]
        st4, _ = _http_get(bport, "/m/v1/models", {"Authorization": "Bearer trip-open"})
        assert st4 == 503
        assert hits["models"] == before_hits
    finally:
        srv.shutdown()
        srv.server_close()
        up.shutdown()
        up.server_close()
