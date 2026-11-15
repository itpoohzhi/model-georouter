"""Тесты атомарной записи, прав `0600`, схемы `router-upstream/v1`, очистки кредов и троттлинга."""

from __future__ import annotations

import json
import os
import socket
import stat
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from model_georouter.bridge_server import BridgeServer
from model_georouter.config import ConfigManager
from model_georouter.egress_hedge import HedgeOutcome, HedgeRecorder, HedgeRecorderProtocol
from model_georouter.errors import AllProxiesPenalizedError, EgressConnectError
from model_georouter.upstream_state import SCHEMA_VERSION, UpstreamStateWriter, strip_credentials
from tests.helpers import FakeConnectProxy, HttpUpstream, Req, free_port, http_request, respond
from tests.v3_types import Track


def _read_state(path: Path) -> dict[str, object]:
    """Читает JSON-файл состояния со строгим приведением типа словаря."""
    parsed = cast(object, json.loads(path.read_text(encoding="utf-8")))
    assert isinstance(parsed, dict)
    return cast(dict[str, object], parsed)


def _unhedged_file_config(tmp_path: Path, upstream_port: int, proxies: list[str]) -> ConfigManager:
    """Создаёт файловый конфиг v2 с ephemeral-портами без production-ограничения `egress_id`."""
    config: dict[str, object] = {
        "server": {
            "listen": "127.0.0.1",
            "port": 0,
            "hedged_connect_enabled": False,
            "f08_policy": "v2",
            "f08_v2_risk_ack": "AD-003/2026-11-14",
            "total_egress_deadline": 40.0,
        },
        "pools": {"route-de": {"type": "http_connect", "proxies": proxies}},
        "upstreams": {"opencode-ai": {"host": "127.0.0.1", "port": upstream_port, "use_tls": False}},
        "adapters": {"cloudcode": {"prefixes": ["/cloudcode"], "upstream": "opencode-ai", "pool": "route-de"}},
        "rules": [],
    }
    cfg_path = tmp_path / "config.json"
    _ = cfg_path.write_text(json.dumps(config), encoding="utf-8")
    return ConfigManager(cfg_path)


@pytest.mark.parametrize("status", [200, 401])
@pytest.mark.parametrize("existing_state_dir", [False, True])
def test_bridge_server_writes_upstream_state_on_unhedged_route_de(
    tmp_path: Path, track: Track, status: int, existing_state_dir: bool
) -> None:
    """Живой мост сам пишет состояние без хеджирования и сужает права существующего каталога."""

    def upstream_handler(conn: socket.socket, req: Req) -> None:
        assert req.path == "/v1internal:loadCodeAssist"
        respond(conn, status, b"{}")

    upstream = track(HttpUpstream(upstream_handler))
    proxy = track(FakeConnectProxy(("127.0.0.1", cast(int, upstream.port))))
    manager = _unhedged_file_config(tmp_path, cast(int, upstream.port), [f"http://127.0.0.1:{cast(int, proxy.port)}"])
    state_file = tmp_path / "state" / "upstream-state.json"
    if existing_state_dir:
        state_file.parent.mkdir()
        state_file.parent.chmod(0o755)
    with BridgeServer(manager) as server:
        thread = server.start_background()
        try:
            response = http_request(server.server_address[1], "POST", "/cloudcode/v1internal:loadCodeAssist", b"{}")
            assert response.status == status
            assert len(proxy.connects) == len(upstream.requests) == 1
            assert state_file.is_file()
            assert stat.S_IMODE(state_file.stat().st_mode) == 0o600
            assert stat.S_IMODE(state_file.parent.stat().st_mode) == 0o700
            payload = _read_state(state_file)
            assert payload["schema"] == "router-upstream/v1"
            assert payload["pool"] == "route-de"
            assert payload["active_upstream"] == f"127.0.0.1:{cast(int, proxy.port)}"
            assert payload["failover_count"] == 0
            assert payload["last_failover_at"] is None
            assert "@" not in state_file.read_text(encoding="utf-8")
        finally:
            server.stop()
            thread.join(1.0)
            assert not thread.is_alive()


def test_bridge_server_tracks_failover_on_unhedged_route_de(tmp_path: Path, track: Track) -> None:
    """Отказ первой прокси учитывается как один failover без записи её учётных данных."""
    upstream = track(HttpUpstream(lambda conn, _req: respond(conn, 401, b"{}")))
    live_proxy = track(FakeConnectProxy(("127.0.0.1", cast(int, upstream.port))))
    manager = _unhedged_file_config(
        tmp_path,
        cast(int, upstream.port),
        [f"http://127.0.0.1:{free_port()}", f"http://u:secret@127.0.0.1:{cast(int, live_proxy.port)}"],
    )
    with BridgeServer(manager) as server:
        thread = server.start_background()
        try:
            response = http_request(server.server_address[1], "POST", "/cloudcode/v1internal:loadCodeAssist", b"{}")
            assert response.status == 401
            assert len(live_proxy.connects) == len(upstream.requests) == 1
            state_file = tmp_path / "state" / "upstream-state.json"
            assert state_file.is_file()
            assert stat.S_IMODE(state_file.stat().st_mode) == 0o600
            assert stat.S_IMODE(state_file.parent.stat().st_mode) == 0o700
            payload = _read_state(state_file)
            assert payload["schema"] == "router-upstream/v1"
            assert payload["pool"] == "route-de"
            assert payload["active_upstream"] == f"127.0.0.1:{cast(int, live_proxy.port)}"
            assert payload["failover_count"] == 1
            assert isinstance(payload["last_failover_at"], str)
            raw = state_file.read_text(encoding="utf-8")
            assert "@" not in raw and "u:secret" not in raw
        finally:
            server.stop()
            thread.join(1.0)
            assert not thread.is_alive()


@pytest.mark.parametrize("file_backed", [False, True])
def test_bridge_server_preserves_injected_state_writer(tmp_path: Path, file_backed: bool) -> None:
    """Явно переданный рекордер имеет приоритет над автоматическим файловым путём."""
    manager = _unhedged_file_config(tmp_path, free_port(), [f"http://127.0.0.1:{free_port()}"])
    if not file_backed:
        manager = ConfigManager(initial=manager.get())
    writer = UpstreamStateWriter(tmp_path / "custom-state.json")
    with BridgeServer(manager, state_writer=writer) as server:
        assert server.state_writer is writer
        assert server.egress.recorder is writer
    assert not writer.path.exists()
    assert not (tmp_path / "state").exists()


def test_bridge_server_without_config_path_has_no_default_state_writer(tmp_path: Path) -> None:
    """Конфигурация в памяти не создаёт рекордер с неявным путём на диске."""
    manager = _unhedged_file_config(tmp_path, free_port(), [f"http://127.0.0.1:{free_port()}"])
    with BridgeServer(ConfigManager(initial=manager.get())) as server:
        assert server.state_writer is None
        assert server.egress.recorder is None
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize(
    "error, status",
    [(EgressConnectError("test connect failure"), 502), (AllProxiesPenalizedError("route-de"), 502)],
)
def test_bridge_server_records_unhedged_route_de_failure(tmp_path: Path, error: Exception, status: int) -> None:
    """Нехеджированная ошибка уведомляет рекордер без изменения исходного HTTP-статуса."""
    manager = _unhedged_file_config(tmp_path, free_port(), [f"http://127.0.0.1:{free_port()}"])
    recorder = HedgeRecorder()
    with BridgeServer(manager) as server:
        server.egress.recorder = recorder
        thread = server.start_background()
        try:
            with patch.object(server.proxy_manager, "_connect", side_effect=error):
                response = http_request(server.server_address[1], "POST", "/cloudcode/v1internal:loadCodeAssist", b"{}")
            assert response.status == status
            assert recorder.failures == [f"route-de:{type(error).__name__}"]
            assert recorder.outcomes == []
            assert not (tmp_path / "state").exists()
        finally:
            server.stop()
            thread.join(1.0)
            assert not thread.is_alive()


def test_strip_credentials_removes_scheme_and_userinfo() -> None:
    """Креды (включая пароли со спецсимволами `/`, `?`, `#`, `@`) и схема удаляются из адреса прокси."""
    assert strip_credentials("http://u:p@127.0.0.1:10828") == "127.0.0.1:10828"
    assert strip_credentials("user:pass@host:1080") == "host:1080"
    assert strip_credentials("socks5://user:pa/ss@10.0.0.1:1080") == "10.0.0.1:1080"
    assert strip_credentials("socks5://user:pa/ss?x#y@10.0.0.1:1080/p?q=1#f") == "10.0.0.1:1080"
    assert strip_credentials("socks5h://sec:ret@p@ss@[::1]:1080/path?x=1") == "[::1]:1080"
    assert strip_credentials("127.0.0.1:10828") == "127.0.0.1:10828"
    assert strip_credentials("direct") == "direct"


def test_schema_permissions_and_credential_stripping(tmp_path: Path) -> None:
    """Проверяет схему `router-upstream/v1`, права `0600` и отсутствие утечки кредов на диск."""
    state_file = tmp_path / "router-upstream.json"
    writer = UpstreamStateWriter(state_file, pool="user:secret@route-de")

    assert writer.update("http://u:p@127.0.0.1:10828") is True
    assert state_file.exists()
    assert [p for p in tmp_path.iterdir() if p != state_file] == []
    assert stat.S_IMODE(state_file.stat().st_mode) == 0o600

    raw_text = state_file.read_text(encoding="utf-8")
    assert "u:p" not in raw_text
    assert "user:secret" not in raw_text

    payload = _read_state(state_file)
    assert set(payload.keys()) == {
        "schema",
        "pool",
        "active_upstream",
        "failover_count",
        "last_failover_at",
        "generated_at",
    }
    assert payload["schema"] == SCHEMA_VERSION == "router-upstream/v1"
    assert payload["pool"] == "route-de"
    assert payload["active_upstream"] == "127.0.0.1:10828"
    assert payload["failover_count"] == 0
    assert payload["last_failover_at"] is None
    gen_at = payload["generated_at"]
    assert isinstance(gen_at, str) and gen_at.endswith("Z")


def test_atomic_write_cleans_up_tmp_and_preserves_previous_file(tmp_path: Path) -> None:
    """При ошибке `os.replace` старый файл не бьётся, а временный `.tmp`-файл удаляется."""
    state_file = tmp_path / "router-upstream.json"
    writer = UpstreamStateWriter(state_file, min_interval_s=0.0)
    assert writer.update("http://u:p@127.0.0.1:10828") is True
    before = state_file.read_text(encoding="utf-8")

    replaced_calls: list[tuple[Path, Path]] = []

    def failing_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        replaced_calls.append((Path(src), Path(dst)))
        raise OSError("simulated replace failure")

    with patch.object(os, "replace", side_effect=failing_replace):
        try:
            _ = writer.update("http://u:p@127.0.0.1:10829", force=True)
        except OSError as exc:
            assert "simulated replace failure" in str(exc)
        else:
            raise AssertionError("Expected OSError from failing_replace")

    assert len(replaced_calls) == 1
    assert replaced_calls[0][0].parent == tmp_path
    assert [p for p in tmp_path.iterdir() if p != state_file] == []
    assert state_file.read_text(encoding="utf-8") == before


def test_throttling_at_most_one_write_per_second_and_force_flush(tmp_path: Path) -> None:
    """Обновления (включая смену upstream и `failover=True`) троттлятся (`<= 1` зап./с) при `force=False`, а `force=True` пишет сразу."""
    now = 100.0
    state_file = tmp_path / "router-upstream.json"
    writer = UpstreamStateWriter(state_file, min_interval_s=1.0, clock=lambda: now)
    replace_count = 0
    real_replace = os.replace

    def counting_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        nonlocal replace_count
        replace_count += 1
        real_replace(src, dst)

    with patch.object(os, "replace", side_effect=counting_replace):
        assert writer.update("http://u:p@127.0.0.1:10828") is True
        assert replace_count == 1

        for step in (0.1, 0.3, 0.6, 0.99):
            now = 100.0 + step
            assert writer.update("http://u:p@127.0.0.1:10828", failover=True) is False
        assert replace_count == 1
        assert writer.snapshot().failover_count == 0

        # Смена active_upstream внутри окна троттлинга (< 1.0 с) меняет память, но соблюдает троттлинг при force=False
        now = 100.5
        assert writer.update("http://u:p@127.0.0.1:10829") is False
        assert replace_count == 1
        assert writer.snapshot().active_upstream == "127.0.0.1:10829"
        assert writer.snapshot().failover_count == 1

        # По истечении min_interval_s (1.0 с) запись проходит; повторный failover=True того же апстрима счётчик не растит
        now = 101.2
        assert writer.update("http://u:p@127.0.0.1:10829", failover=True) is True
        assert replace_count == 2

        data = _read_state(state_file)
        assert data["active_upstream"] == "127.0.0.1:10829"
        assert data["failover_count"] == 1
        assert isinstance(data["last_failover_at"], str)

        now = 101.5
        assert writer.update("http://u:p@127.0.0.1:10829") is False
        assert replace_count == 2
        assert writer.flush(force=True) is True
        assert replace_count == 3


def test_hedge_recorder_protocol_integration(tmp_path: Path) -> None:
    """`UpstreamStateWriter` совместим с `HedgeRecorderProtocol` и не накручивает `failover_count` на том же upstream."""
    state_file = tmp_path / "state.json"
    writer = UpstreamStateWriter(state_file, min_interval_s=0.0)
    assert isinstance(writer, HedgeRecorderProtocol)

    writer.record(HedgeOutcome("route-de", "http://u:p@127.0.0.1:10828", None, 2, True, True, True))
    writer.record(HedgeOutcome("route-de", "http://u:p@127.0.0.1:10828", None, 2, True, True, True))
    data = _read_state(state_file)
    assert data["active_upstream"] == "127.0.0.1:10828"
    assert data["failover_count"] == 1
    assert data["last_failover_at"] is not None
