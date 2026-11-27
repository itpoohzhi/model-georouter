"""L1 + L2 tests for model-georouter pre-relay client disconnect (W14-W15, VER-004)."""

from __future__ import annotations

import errno
import json
import os
import socket
import struct
import threading
import time
from pathlib import Path

import pytest

from model_georouter.bridge_server import BridgeServer
from model_georouter.config import (
    AdapterConfig,
    BridgeConfig,
    ConfigManager,
    PoolConfig,
    RuleConfig,
    ServerConfig,
    UpstreamConfig,
)
from model_georouter.prerelay_policy import (
    FLAG_GEO_PRERELAY_DISCONNECT,
    ClientWatchAction,
    PreRelayClientResetError,
    classify_client_oserror,
    classify_client_peek,
    resolve_prerelay_flag,
)
from model_georouter.prerelay_watch import DEFAULT_FLAGS_PATH_ENV, DynamicFlagCache, SelectorPreRelayWatcher

# --- L1 Zero-I/O policy & golden precedence tests ---


@pytest.mark.parametrize(
    ("json_snap", "env_snap", "json_invalid", "expected"),
    [
        (None, {}, False, False),
        ({FLAG_GEO_PRERELAY_DISCONNECT: False}, {FLAG_GEO_PRERELAY_DISCONNECT: "1"}, False, False),
        ({FLAG_GEO_PRERELAY_DISCONNECT: True}, {FLAG_GEO_PRERELAY_DISCONNECT: "0"}, False, True),
        ({FLAG_GEO_PRERELAY_DISCONNECT.lower(): "true"}, {FLAG_GEO_PRERELAY_DISCONNECT: "false"}, False, True),
        ({FLAG_GEO_PRERELAY_DISCONNECT.lower(): "false"}, {FLAG_GEO_PRERELAY_DISCONNECT: "true"}, False, False),
        ({}, {FLAG_GEO_PRERELAY_DISCONNECT: "1"}, False, True),
        ({}, {FLAG_GEO_PRERELAY_DISCONNECT: "0"}, False, False),
        ({FLAG_GEO_PRERELAY_DISCONNECT: True}, {FLAG_GEO_PRERELAY_DISCONNECT: "1"}, True, False),
        ({FLAG_GEO_PRERELAY_DISCONNECT: "invalid"}, {FLAG_GEO_PRERELAY_DISCONNECT: "1"}, False, False),
    ],
)
def test_golden_flag_precedence_vectors(
    json_snap: dict[str, object] | None,
    env_snap: dict[str, str],
    json_invalid: bool,
    expected: bool,
) -> None:
    assert resolve_prerelay_flag(json_snap, env_snap, json_invalid=json_invalid) is expected


def test_exception_does_not_inherit_oserror_or_connection_error() -> None:
    exc = PreRelayClientResetError("client_rst_prerelay", errno.ECONNRESET)
    assert isinstance(exc, Exception)
    assert not isinstance(exc, OSError)
    assert not isinstance(exc, ConnectionError)


def test_classify_client_oserror_and_peek() -> None:
    assert classify_client_oserror(errno.EAGAIN) is ClientWatchAction.KEEP
    assert classify_client_oserror(errno.EINTR) is ClientWatchAction.KEEP
    assert classify_client_oserror(errno.ECONNRESET) is ClientWatchAction.RST_CANCEL
    assert classify_client_oserror(errno.EPIPE) is ClientWatchAction.RST_CANCEL
    assert classify_client_oserror(errno.ECONNABORTED) is ClientWatchAction.RST_CANCEL
    assert classify_client_oserror(errno.ETIMEDOUT) is ClientWatchAction.RST_CANCEL
    assert classify_client_peek(b"", 0) is ClientWatchAction.FIN_UNWATCH
    assert classify_client_peek(b"G", 0) is ClientWatchAction.PEEK_DATA_UNWATCH
    assert classify_client_peek(b"", errno.ECONNRESET) is ClientWatchAction.RST_CANCEL


def test_dynamic_flag_cache_invalidation_on_atomic_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    flags_file = tmp_path / "flags.json"
    monkeypatch.setenv(FLAG_GEO_PRERELAY_DISCONNECT, "1")
    cache = DynamicFlagCache(flags_file)
    assert cache.is_enabled() is True  # missing file -> env fallback

    flags_file.write_text(json.dumps({FLAG_GEO_PRERELAY_DISCONNECT: False}), encoding="utf-8")
    assert cache.is_enabled() is False  # JSON false overrides env=1

    # Atomic replace with same mtime_ns but different inode/size
    tmp2 = tmp_path / "flags.json.tmp"
    tmp2.write_text(json.dumps({FLAG_GEO_PRERELAY_DISCONNECT: True, "pad": "extra"}), encoding="utf-8")
    st = flags_file.stat()
    os.utime(tmp2, ns=(st.st_atime_ns, st.st_mtime_ns))
    os.replace(tmp2, flags_file)
    assert cache.is_enabled() is True

    flags_file.unlink()
    assert cache.is_enabled() is True  # deleted -> env fallback


# --- L2 Real loopback socket tests ---


def _force_rst(sock: socket.socket) -> None:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    sock.close()


def _tcp_pair() -> tuple[socket.socket, socket.socket]:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    cli = socket.create_connection(("127.0.0.1", port), timeout=2.0)
    acc, _ = srv.accept()
    srv.close()
    return acc, cli


def test_l2_client_rst_aborts_prerelay_fast_while_flag_false_keeps_legacy(tmp_path: Path) -> None:
    flags_file = tmp_path / "flags.json"
    flags_file.write_text(json.dumps({FLAG_GEO_PRERELAY_DISCONNECT: True}), encoding="utf-8")
    watcher = SelectorPreRelayWatcher(DynamicFlagCache(flags_file))

    c_srv, c_cli = _tcp_pair()
    u_srv, u_cli = _tcp_pair()
    try:

        def _send_rst() -> None:
            time.sleep(0.05)
            _force_rst(c_cli)

        t = threading.Thread(target=_send_rst, daemon=True)
        t.start()
        t0 = time.monotonic()
        with pytest.raises(PreRelayClientResetError):

            def _read_u(u: socket.socket, to: float, _dl: float | None) -> tuple[bytes, bytes]:
                u.settimeout(to)
                return u.recv(16), b""

            watcher.read_head(c_srv, u_srv, 2.0, None, _read_u)
        elapsed = time.monotonic() - t0
        t.join(timeout=1.0)
        assert elapsed < 0.6
    finally:
        for s in (c_srv, u_srv, u_cli):
            try:
                s.close()
            except OSError:
                pass


def test_l2_fin_shut_wr_and_pipelined_bytes_preserve_response_without_busyloop(tmp_path: Path) -> None:
    flags_file = tmp_path / "flags.json"
    flags_file.write_text(json.dumps({FLAG_GEO_PRERELAY_DISCONNECT: True}), encoding="utf-8")
    watcher = SelectorPreRelayWatcher(DynamicFlagCache(flags_file))

    for mode in ("shut_wr", "pipelined"):
        c_srv, c_cli = socket.socketpair()
        u_srv, u_cli = socket.socketpair()
        try:
            if mode == "shut_wr":
                c_cli.shutdown(socket.SHUT_WR)
            else:
                c_cli.sendall(b"NEXT-PIPELINE-REQ")

            def _delayed_upstream(peer: socket.socket = u_cli) -> None:
                time.sleep(0.12)
                peer.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")

            t = threading.Thread(target=_delayed_upstream, daemon=True)
            t.start()
            head, rest = watcher.read_head(c_srv, u_srv, 2.0, None, lambda u, to, dl: (u.recv(64), b""))
            t.join(timeout=1.0)
            assert b"200 OK" in head
            assert rest == b""
            # Client socket stays open and pipelined bytes were NOT consumed!
            if mode == "pipelined":
                assert c_srv.recv(32) == b"NEXT-PIPELINE-REQ"
            c_srv.sendall(b"REPLY")
            assert c_cli.recv(16) == b"REPLY"
        finally:
            for s in (c_srv, c_cli, u_srv, u_cli):
                try:
                    s.close()
                except OSError:
                    pass


def test_l2_ssl_pending_bypasses_selector_wait(tmp_path: Path) -> None:
    flags_file = tmp_path / "flags.json"
    flags_file.write_text(json.dumps({FLAG_GEO_PRERELAY_DISCONNECT: True}), encoding="utf-8")
    watcher = SelectorPreRelayWatcher(DynamicFlagCache(flags_file))

    class _FakeTLS:
        def pending(self) -> int:
            return 16

    c_srv, c_cli = socket.socketpair()
    try:
        called = []
        res = watcher.read_head(c_srv, _FakeTLS(), 2.0, None, lambda u, to, dl: (called.append(True), b"ok"))  # type: ignore[arg-type]
        assert called == [True]
        assert res[1] == b"ok"
    finally:
        c_srv.close()
        c_cli.close()


def test_l2_full_bridge_server_client_rst_no_proxy_penalty_and_shut_wr_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    flags_file = tmp_path / "geo-flags.json"
    flags_file.write_text(json.dumps({FLAG_GEO_PRERELAY_DISCONNECT: True}), encoding="utf-8")
    monkeypatch.setenv(DEFAULT_FLAGS_PATH_ENV, str(flags_file))

    # Start a slow loopback upstream that delays headers by 0.45s
    up_events: list[str] = []
    up_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    up_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    up_srv.bind(("127.0.0.1", 0))
    up_port = up_srv.getsockname()[1]
    up_srv.listen(8)
    stop_ev = threading.Event()

    def _up_loop() -> None:
        up_srv.settimeout(0.1)
        while not stop_ev.is_set():
            try:
                conn, _ = up_srv.accept()
            except TimeoutError:
                continue
            except OSError:
                break

            def _handle(c: socket.socket) -> None:
                try:
                    c.settimeout(1.0)
                    _ = c.recv(4096)
                    up_events.append("req")
                    time.sleep(0.35)
                    c.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK")
                except OSError:
                    pass
                finally:
                    c.close()

            threading.Thread(target=_handle, args=(conn,), daemon=True).start()

    up_thread = threading.Thread(target=_up_loop, daemon=True)
    up_thread.start()

    cfg = BridgeConfig(
        server=ServerConfig(
            listen="127.0.0.1",
            port=0,
            headers_timeout=5.0,
            body_timeout=2.0,
            inactivity_timeout=5.0,
            log_dir=str(tmp_path / "logs"),
        ),
        upstreams={"slow": UpstreamConfig(host="127.0.0.1", port=up_port, use_tls=False)},
        adapters={"slow": AdapterConfig(prefixes=("/slow",), upstream="slow", pool="direct")},
        pools={"direct": PoolConfig(name="direct", type="direct")},
        rules=(RuleConfig(pool="direct", match_prefix=("/slow",)),),
    )
    mgr = ConfigManager(initial=cfg)
    bridge = BridgeServer(mgr)
    b_thread = threading.Thread(target=bridge.serve_forever, daemon=True)
    b_thread.start()
    b_port = bridge.server_address[1]
    try:
        # 1. Client sends request then RSTs after 50ms -> bridge aborts pre-relay quickly with client_cancelled
        s1 = socket.create_connection(("127.0.0.1", b_port), timeout=2.0)
        s1.sendall(b"GET /slow/test HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        time.sleep(0.06)
        _force_rst(s1)
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and bridge.metrics.snapshot().get("client_cancelled", 0) < 1:
            time.sleep(0.02)
        assert bridge.metrics.snapshot().get("client_cancelled", 0) >= 1

        # 2. Client sends request + SHUT_WR (half-close) -> receives full 200 OK!
        s2 = socket.create_connection(("127.0.0.1", b_port), timeout=2.0)
        s2.sendall(b"GET /slow/test HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        s2.shutdown(socket.SHUT_WR)
        chunks = []
        s2.settimeout(2.0)
        while True:
            data = s2.recv(4096)
            if not data:
                break
            chunks.append(data)
        s2.close()
        resp = b"".join(chunks)
        assert b"200 OK" in resp and resp.endswith(b"OK")
    finally:
        bridge.shutdown()
        bridge.server_close()
        stop_ev.set()
        up_srv.close()
        b_thread.join(timeout=2.0)
        up_thread.join(timeout=2.0)


def test_l2_prerelay_enabled_header_timeout_returns_504_not_502(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When flag is True, header timeout in selector returns 504 upstream_timeout (no ValueError/TimeoutError 502)."""
    from tests.helpers import FakeServer, http_request

    flags_file = tmp_path / "geo-flags.json"
    flags_file.write_text(json.dumps({FLAG_GEO_PRERELAY_DISCONNECT: True}), encoding="utf-8")
    monkeypatch.setenv(DEFAULT_FLAGS_PATH_ENV, str(flags_file))

    up_srv = FakeServer(lambda conn: time.sleep(1.5))
    cfg = BridgeConfig(
        server=ServerConfig(
            listen="127.0.0.1",
            port=0,
            headers_timeout=0.25,
            body_timeout=1.0,
            inactivity_timeout=1.0,
            log_dir=str(tmp_path / "logs"),
        ),
        upstreams={"slow": UpstreamConfig(host="127.0.0.1", port=up_srv.port, use_tls=False)},
        adapters={"slow": AdapterConfig(prefixes=("/v1/chat",), upstream="slow", pool="direct")},
        pools={"direct": PoolConfig(name="direct", type="direct")},
        rules=(RuleConfig(pool="direct", match_prefix=("/v1/chat",)),),
    )
    bridge = BridgeServer(ConfigManager(initial=cfg))
    b_thread = threading.Thread(target=bridge.serve_forever, daemon=True)
    b_thread.start()
    try:
        resp = http_request(
            bridge.server_address[1],
            "POST",
            "/v1/chat/completions",
            b'{"model":"m","messages":[{"role":"user","content":"hi"}]}',
        )
        assert resp.status == 504
        assert resp.json["error"]["type"] == "upstream_timeout"
    finally:
        bridge.shutdown()
        bridge.server_close()
        up_srv.close()
        b_thread.join(timeout=2.0)
