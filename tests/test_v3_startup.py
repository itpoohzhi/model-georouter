"""Регрессии запуска v3 через реальные снимки конфигурации."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_georouter.bridge_server import BridgeServer
from model_georouter.config import ConfigManager, load_config_file, parse_config
from model_georouter.config_guard import validate
from model_georouter.egress_admission import EgressAdmission
from model_georouter.errors import ConfigError
from tests.helpers import http_request


def live_config(timeout: float | None = None) -> dict[str, object]:
    """Воспроизводит существенные поля живого конфига без секретов и боевого порта."""
    return {
        "server": {
            "listen": "127.0.0.1",
            "port": 0,
            "proxy_slots": 80,
            "total_egress_deadline": 40,
            "f08_policy": "v2",
            "f08_v2_risk_ack": "AD-003/2026-11-14",
            "client_timeout_s": timeout,
            "hedged_connect_enabled": False,
        },
        "pools": {
            "route-de": {
                "type": "http_connect",
                "proxies": ["http://127.0.0.1:10827"],
                "egress_id": "test-egress",
            }
        },
        "rules": [],
    }


@pytest.mark.parametrize("guarded", [False, True])
@pytest.mark.parametrize("timeout", [None, 180.0])
def test_admission_reads_v2_from_config(tmp_path: Path, guarded: bool, timeout: float | None) -> None:
    path = tmp_path / "config.json"
    _ = path.write_text(json.dumps(live_config(timeout)))
    bridge = load_config_file(path)
    adm = EgressAdmission(config=validate(bridge) if guarded else bridge)
    assert adm.f08_policy == "v2"
    assert adm.f08_v2_budget is not None
    with adm.admit("route-de", "/streamGenerateContent", started_at=100, now=150) as ticket:
        assert ticket.header_deadline_s == pytest.approx(100)


@pytest.mark.parametrize("guarded", [False, True])
def test_admission_does_not_invent_risk_ack(guarded: bool) -> None:
    bridge = parse_config(live_config())
    from dataclasses import replace

    bridge = replace(bridge, server=replace(bridge.server, f08_v2_risk_ack=None))
    with pytest.raises(ConfigError):
        _ = EgressAdmission(config=validate(bridge) if guarded else bridge)


@pytest.mark.parametrize("guarded", [False, True])
def test_explicit_admission_values_take_priority(guarded: bool) -> None:
    bridge = parse_config(live_config())
    cfg = validate(bridge) if guarded else bridge
    with pytest.raises(ConfigError):
        _ = EgressAdmission(config=cfg, f08_v2_risk_ack="invalid", client_timeout_s=1.0)
    assert EgressAdmission(config=cfg, client_timeout_s=180.0).f08_v2_budget is not None


def test_hedge_flag_defaults_and_strict_boolean() -> None:
    section: dict[str, object] = {"listen": "127.0.0.1", "port": 0}
    minimal: dict[str, object] = {"server": section, "pools": {}, "rules": []}
    assert parse_config(minimal).server.hedged_connect_enabled is False
    assert parse_config(live_config()).server.hedged_connect_enabled is False
    for invalid in ("false", 0, None):
        with pytest.raises(ConfigError, match="hedged_connect_enabled"):
            _ = parse_config({**minimal, "server": {**section, "hedged_connect_enabled": invalid}})


def test_bridge_starts_v2_and_applies_disabled_hedge() -> None:
    server = BridgeServer(ConfigManager.from_dict(live_config()))
    thread = server.start_background()
    try:
        assert server.egress.policy.enabled is False
        assert server.admission.f08_policy == "v2"
        assert http_request(server.server_address[1], "GET", "/health").status == 200
    finally:
        server.stop()
        thread.join(2)
    assert not thread.is_alive()
