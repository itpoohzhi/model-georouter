"""Тесты атомарной записи, прав `0600`, схемы `router-upstream/v1`, очистки кредов и троттлинга."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import cast
from unittest.mock import patch

from model_georouter.egress_hedge import HedgeOutcome, HedgeRecorderProtocol
from model_georouter.upstream_state import SCHEMA_VERSION, UpstreamStateWriter, strip_credentials


def _read_state(path: Path) -> dict[str, object]:
    """Читает JSON-файл состояния со строгим приведением типа словаря."""
    parsed = cast(object, json.loads(path.read_text(encoding="utf-8")))
    assert isinstance(parsed, dict)
    return cast(dict[str, object], parsed)


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
