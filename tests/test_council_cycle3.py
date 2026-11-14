"""Регрессионные тесты замечаний Совета Тимлидов Cycle 3 (RW-001 .. RW-004)."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from _typeshed import ReadableBuffer
from pathlib import Path

import pytest

from model_georouter import config as config_module
from model_georouter import proxy_pool
from model_georouter._typing import override
from model_georouter.bridge_server import OVERLOADED_503, Metrics, SlotPool
from model_georouter.config import default_config_path, default_log_dir, parse_config
from model_georouter.errors import BadRequestError, EgressTimeoutError
from model_georouter.model_router import BodyInspector
from tests.helpers import HttpUpstream, Req, make_config
from tests.v3_types import ConfigData, StartBridge, Track

ROOT = Path(__file__).resolve().parent.parent


def _feed(body: bytes, step: int | None = None) -> BodyInspector:
    inspector = BodyInspector(1 << 20)
    for i in range(0, len(body), step or len(body)):
        inspector.feed(body[i : i + (step or len(body))])
    return inspector


# ───────────────────────────── RW-001: дубли `model` разных типов ─────────────────────────────


@pytest.mark.parametrize("step", [None, 1, 5])
@pytest.mark.parametrize(
    "body",
    [
        b'{"model":"a","model":null}',
        b'{"model":null,"model":"a"}',
        b'{"model":"a","x":1,"model":42}',
        b'{"model":7,"model":"a"}',
        b'{"model":"a","model":["a"]}',
        b'{"model":{"n":"a"},"model":"a"}',
        b'{"model":"a","model":{"n":"a"}}',
        b'{"model":null,"model":null}',
        b'{"model":"a","model":"b"}',
        b'{"model":"a","model":true}',
    ],
)
def test_rw001_conflicting_duplicate_model_keys_are_rejected(body: bytes, step: int | None) -> None:
    with pytest.raises(BadRequestError, match="Conflicting duplicate model keys"):
        _ = _feed(body, step)


@pytest.mark.parametrize("step", [None, 1, 3])
def test_rw001_identical_string_duplicates_are_allowed(step: int | None) -> None:
    assert _feed(b'{"model":"a","z":[1],"model":"a","model":"a"}', step).model == "a"
    assert _feed(b'{"model":"a","m":{"model":null}}', step).model == "a"


def test_rw001_conflicting_model_gets_400_over_the_wire(track: Track, start_bridge: StartBridge) -> None:
    up = track(HttpUpstream(lambda conn, req: conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")))
    bridge = start_bridge(make_config(cast(int, up.port)))
    body = b'{"model":"claude-x","model":null}'
    with socket.create_connection(("127.0.0.1", bridge.port), timeout=5) as sock:
        sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: h\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
        assert sock.recv(64).startswith(b"HTTP/1.1 400 ")
    assert up.requests == []


# ───────────────────────────── RW-002: пути конфигурации по умолчанию ─────────────────────────────


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def _touch(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if path.name != "logs":
        _ = (path / "config.json").write_text("{}")


def test_rw002_default_is_model_georouter_when_nothing_exists(home: Path) -> None:
    _ = home
    assert default_config_path() == Path("~/.config/model-georouter/config.json")
    assert default_log_dir() == "~/.config/model-georouter/logs"


def test_rw002_legacy_is_used_when_only_legacy_exists(home: Path) -> None:
    _touch(home / ".config/universal-ai-bridge")
    (home / ".config/universal-ai-bridge/logs").mkdir()
    assert default_config_path() == Path("~/.config/universal-ai-bridge/config.json")
    assert default_log_dir() == "~/.config/universal-ai-bridge/logs"


def test_rw002_new_path_wins_over_legacy(home: Path) -> None:
    for name in ("model-georouter", "universal-ai-bridge"):
        _touch(home / ".config" / name)
        (home / ".config" / name / "logs").mkdir()
    assert default_config_path() == Path("~/.config/model-georouter/config.json")
    assert default_log_dir() == "~/.config/model-georouter/logs"


def test_rw002_server_config_log_dir_follows_resolution(home: Path) -> None:
    raw: ConfigData = {"server": {"listen": "127.0.0.1", "port": 1}, "pools": {}, "rules": []}
    assert parse_config(raw).server.log_dir == "~/.config/model-georouter/logs"
    (home / ".config/universal-ai-bridge/logs").mkdir(parents=True)
    assert parse_config(raw).server.log_dir == "~/.config/universal-ai-bridge/logs"
    assert config_module.ServerConfig(listen="127.0.0.1", port=1).log_dir == "~/.config/universal-ai-bridge/logs"
    raw["server"]["log_dir"] = "/x"
    assert parse_config(raw).server.log_dir == "/x"


def test_rw002_cli_default_uses_resolved_path(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _ = monkeypatch
    from model_georouter.__main__ import main

    _touch(home / ".config/universal-ai-bridge")
    assert main([]) == 2  # пустой `{}` из legacy — невалидный конфиг: доказывает, что файл прочитан
    assert "config error" in capsys.readouterr().err
    assert "model-georouter" in (ROOT / "README.md").read_text()


# ───────────────────────────── RW-003: ограничение потоков DNS ─────────────────────────────


def test_rw003_dns_threads_are_capped_by_semaphore(monkeypatch: pytest.MonkeyPatch) -> None:
    sem = threading.BoundedSemaphore(2)
    monkeypatch.setattr(proxy_pool, "_DNS_SEMAPHORE", sem)
    release = threading.Event()
    started: list[int] = []

    def blocked(*args: object, **kwargs: object) -> list[object]:
        _ = (args, kwargs)
        started.append(1)
        _ = release.wait(5)
        return []

    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    try:
        for _ in range(2):
            with pytest.raises(TimeoutError):
                _ = cast(Callable[[str, int, float], object], vars(proxy_pool)["_resolve"])(
                    "slow-dns.invalid", 80, time.monotonic() + 0.1
                )
        threads_before = threading.active_count()
        began = time.monotonic()
        with pytest.raises(EgressTimeoutError, match="concurrency limit"):
            _ = cast(Callable[[str, int, float], object], vars(proxy_pool)["_resolve"])(
                "slow-dns.invalid", 80, time.monotonic() + 0.2
            )
        assert time.monotonic() - began < 2.0
        assert len(started) == 2 and threading.active_count() == threads_before  # третий поток не создан
    finally:
        release.set()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not sem.acquire(blocking=False):
        time.sleep(0.01)
    sem.release()
    for _ in range(2):  # слоты возвращены после завершения резолвера
        assert sem.acquire(blocking=False)


# ───────────────────────────── RW-004: полная доставка inline 503 ─────────────────────────────


def test_rw004_inline_503_is_fully_delivered(track: Track, start_bridge: StartBridge) -> None:
    def delayed(conn: socket.socket, _req: Req) -> None:
        time.sleep(1)
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")

    up = track(HttpUpstream(delayed))
    bridge = start_bridge(make_config(cast(int, up.port), server={"max_connections": 1}))
    holder = socket.create_connection(("127.0.0.1", bridge.port), timeout=5)
    try:
        holder.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: h\r\nContent-Length: 100\r\n\r\n")
        time.sleep(0.3)
        for _ in range(20):
            with socket.create_connection(("127.0.0.1", bridge.port), timeout=5) as sock:
                data = b""
                while chunk := sock.recv(4096):
                    data += chunk
            assert data == OVERLOADED_503 and data.endswith(b"Service Unavailable\n")
    finally:
        holder.close()


def test_rw004_inline_503_uses_sendall_with_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    _ = monkeypatch
    from model_georouter.bridge_server import BridgeServer

    calls: list[tuple[object, ...]] = []

    class FakeSock(socket.socket):
        @override
        def settimeout(self, value: float | None) -> None:
            calls.append(("settimeout", value))

        @override
        def setblocking(self, flag: bool) -> None:
            calls.append(("setblocking", flag))

        @override
        def sendall(self, data: ReadableBuffer, flags: int = 0) -> None:
            _ = flags
            calls.append(("sendall", data))

        @override
        def shutdown(self, how: int) -> None:
            calls.append(("shutdown", how))

        @override
        def recv(self, size: int, flags: int = 0) -> bytes:
            _ = (size, flags)
            return b""

    server = BridgeServer.__new__(BridgeServer)
    server.ingress = SlotPool(0)
    monkeypatch.setattr(server, "_check_reload", lambda: None)

    def shutdown_request(_request: object) -> None:
        calls.append(("closed",))

    monkeypatch.setattr(server, "shutdown_request", shutdown_request)
    server.metrics = Metrics()
    with FakeSock() as sock:
        server.process_request(sock, ("127.0.0.1", 1))
    assert calls[0] == ("settimeout", 0.5) and calls[1] == ("sendall", OVERLOADED_503) and calls[-1] == ("closed",)
    assert len(OVERLOADED_503.split(b"\r\n\r\n", 1)[1]) == 20


def test_rw004_ingress_wait_timeout_is_deprecated_but_validated() -> None:
    from model_georouter.config import ConfigError

    readme = (ROOT / "README.md").read_text()
    assert "ingress_wait_timeout" in readme and "ignored" in readme.lower()
    base: ConfigData = {
        "server": {"listen": "127.0.0.1", "port": 1, "ingress_wait_timeout": 2},
        "pools": {},
        "rules": [],
    }
    assert parse_config(base).server.ingress_wait_timeout == 2.0
    base["server"]["ingress_wait_timeout"] = 0
    with pytest.raises(ConfigError):
        _ = parse_config(base)
