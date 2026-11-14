# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnusedCallResult=false, reportUntypedFunctionDecorator=false, reportAny=false, reportExplicitAny=false
"""Тесты стартовой валидации конфигурации v3 (`model_georouter.config_guard`, W1-B)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import model_georouter.config_guard as cg
from model_georouter.config import ConfigManager, parse_config
from model_georouter.errors import ConfigError


def _valid_v3_config(**overrides: Any) -> dict[str, Any]:
    """Возвращает эталонную валидную конфигурацию v3 для проверок W1-B."""
    cfg: dict[str, Any] = {
        "server": {
            "listen": "127.0.0.1",
            "port": 10830,
            "proxy_slots": 80,
            "max_inflight_per_backend": 40,
            "max_inflight_long_header": 8,
            "total_egress_deadline": 40,
            "f08_policy": "v1",
        },
        "pools": {
            "direct": {"type": "direct"},
            "route-de": {
                "type": "http_connect",
                "egress_id": "de-213.165.42.220",
                "proxies": ["http://127.0.0.1:10827", "http://127.0.0.1:10828", "http://127.0.0.1:10829"],
            },
        },
        "rules": [{"match_prefix": ["claude-"], "pool": "route-de"}],
    }
    cfg.update(overrides)
    return cfg


def test_valid_v3_config_passes_and_returns_guarded_snapshot() -> None:
    """Эталонный конфиг с выделенными DE портами 10827..10829, egress_id и дедлайном 40 проходит валидацию."""
    guarded = cg.validate_config(_valid_v3_config())
    assert isinstance(guarded, cg.GuardedConfig)
    assert guarded.bridge.server.listen == "127.0.0.1"
    assert (guarded.max_inflight_per_backend, guarded.max_inflight_long_header) == (40, 8)
    assert (guarded.f08_policy, guarded.route_de_egress_id) == ("v1", "de-213.165.42.220")
    assert cg.ALLOWED_ROUTE_DE_PORTS == frozenset({10827, 10828, 10829})
    assert cg.FORBIDDEN_ROUTE_DE_PORTS == frozenset({10810, 10820})
    assert len(cg.ALLOWED_ROUTE_DE_BACKENDS) == 3


@pytest.mark.parametrize("bad_listen", ["0.0.0.0", "::", "::1", "192.168.1.10", "10.0.0.1", "localhost", "", 127])
def test_rejects_listen_host_not_equal_to_ipv4_loopback(bad_listen: object) -> None:
    """Любой `server.listen` кроме `'127.0.0.1'` отклоняется с `ConfigError`."""
    with pytest.raises(ConfigError, match="listen"):
        cg.validate_listen_host(bad_listen)
    cfg = _valid_v3_config()
    cfg["server"]["listen"] = bad_listen
    with pytest.raises(ConfigError, match="listen"):
        cg.validate_config(cfg)


@pytest.mark.parametrize("forbidden_port", [10820, 10810])
def test_route_de_explicitly_rejects_forbidden_ports_10820_and_10810(forbidden_port: int) -> None:
    """Порты :10820 и :10810 в пуле `route-de` явно запрещены."""
    cfg = _valid_v3_config()
    cfg["pools"]["route-de"]["proxies"] = [f"http://127.0.0.1:{forbidden_port}"]
    with pytest.raises(ConfigError, match=f":{forbidden_port}"):
        cg.validate_config(cfg)


def test_route_de_forbidden_proxy_redacts_credentials_in_config_error(tmp_path: Path) -> None:
    """При `http://secret_user:secret_pass@127.0.0.1:10820` в `route-de` `ConfigError` не содержит логин и пароль."""
    cfg = _valid_v3_config()
    cfg["pools"]["route-de"]["proxies"] = ["http://secret_user:secret_pass@127.0.0.1:10820"]
    for check in (
        lambda: parse_config(cfg),
        lambda: ConfigManager.from_dict(cfg),
        lambda: cg.validate_config(cfg),
        lambda: cg.validate_route_de_pool(cfg["pools"]["route-de"]),
    ):
        with pytest.raises(ConfigError) as exc_info:
            _ = check()
        msg = str(exc_info.value)
        assert "127.0.0.1:10820" in msg and "secret_user" not in msg and "secret_pass" not in msg
    cfg_file = tmp_path / "reload.json"
    cfg_file.write_text(json.dumps(_valid_v3_config()), encoding="utf-8")
    mgr = ConfigManager(cfg_file, poll_interval=0.0)
    cfg_file.write_text(json.dumps(cfg), encoding="utf-8")
    assert mgr.reload_if_changed() is False and mgr.last_error is not None
    assert "secret_user" not in mgr.last_error and "secret_pass" not in mgr.last_error


@pytest.mark.parametrize(
    "bad_proxy",
    [
        "http://127.0.0.1:10821",
        "http://127.0.0.1:3128",
        "http://localhost:10827",
        "http://192.168.1.2:10827",
        "http://de-node.example:10828",
        "socks5h://10.0.0.1:10829",
        "",
        "not-a-url",
    ],
)
def test_route_de_rejects_non_dedicated_de_loopback_urls(bad_proxy: str) -> None:
    """В пуле `route-de` разрешены только `127.0.0.1:10827`, `127.0.0.1:10828`, `127.0.0.1:10829`."""
    cfg = _valid_v3_config()
    cfg["pools"]["route-de"]["proxies"] = [bad_proxy]
    with pytest.raises(ConfigError, match="route-de"):
        cg.validate_config(cfg)


@pytest.mark.parametrize("bad_egress_id", [None, "", "   ", 123, []])
def test_route_de_requires_non_empty_egress_id(bad_egress_id: object) -> None:
    """Пул `route-de` обязан иметь непустой строковый `egress_id` (D3-M4)."""
    cfg = _valid_v3_config()
    if bad_egress_id is None:
        cfg["pools"]["route-de"].pop("egress_id")
    else:
        cfg["pools"]["route-de"]["egress_id"] = bad_egress_id
    with pytest.raises(ConfigError, match="egress_id"):
        cg.validate_config(cfg)


def test_route_de_rejects_direct_type_and_empty_proxies() -> None:
    """Пул `route-de` не может быть `direct` или иметь пустой список прокси."""
    with pytest.raises(ConfigError, match="direct"):
        cg.validate_route_de_pool({"type": "direct", "egress_id": "de-213.165.42.220", "proxies": []})
    with pytest.raises(ConfigError, match="proxies"):
        cg.validate_route_de_pool({"type": "http_connect", "egress_id": "de-213.165.42.220", "proxies": []})


@pytest.mark.parametrize(
    ("proxy_slots", "per_backend", "long_header"), [(80, 1, 1), (80, 40, 8), (16, 8, 4), (2, 1, 1)]
)
def test_inflight_limits_accept_valid_bounds(proxy_slots: int, per_backend: int, long_header: int) -> None:
    """Проверка границ `max_inflight_per_backend <= proxy_slots // 2` и `max_inflight_long_header > 0`."""
    assert cg.validate_inflight_limits(proxy_slots, per_backend, long_header) == (per_backend, long_header)


@pytest.mark.parametrize(
    ("proxy_slots", "per_backend", "long_header", "fragment"),
    [
        (80, 0, 8, "max_inflight_per_backend"),
        (80, -1, 8, "max_inflight_per_backend"),
        (80, 41, 8, "max_inflight_per_backend"),
        (16, 9, 8, "max_inflight_per_backend"),
        (80, True, 8, "max_inflight_per_backend"),
        (80, "40", 8, "max_inflight_per_backend"),
        (80, 40, 0, "max_inflight_long_header"),
        (80, 40, -2, "max_inflight_long_header"),
        (80, 40, False, "max_inflight_long_header"),
        (1, 1, 1, "proxy_slots"),
    ],
)
def test_inflight_limits_reject_invalid_values(
    proxy_slots: object, per_backend: object, long_header: object, fragment: str
) -> None:
    """Невалидные `max_inflight_per_backend` и `max_inflight_long_header` поднимают `ConfigError`."""
    with pytest.raises(ConfigError, match=fragment):
        cg.validate_inflight_limits(proxy_slots, per_backend, long_header)
    cfg = _valid_v3_config()
    cfg["server"].update(
        proxy_slots=proxy_slots, max_inflight_per_backend=per_backend, max_inflight_long_header=long_header
    )
    with pytest.raises(ConfigError):
        cg.validate_config(cfg)


def test_f08_policy_v1_and_default_require_total_egress_deadline_40() -> None:
    """При `f08_policy == 'v1'` или по умолчанию `total_egress_deadline` обязан быть равен 40."""
    assert (
        cg.validate_f08_deadline(40, "v1"),
        cg.validate_f08_deadline(40.0),
        cg.validate_f08_deadline(20.0, "v2"),
    ) == ("v1", "v1", "v2")
    for bad_deadline in (20, 20.0, 39.9, 40.1, True):
        with pytest.raises(ConfigError, match="total_egress_deadline"):
            cg.validate_f08_deadline(bad_deadline, "v1")
    cfg = _valid_v3_config()
    cfg["server"].pop("f08_policy")
    cfg["server"]["total_egress_deadline"] = 20
    with pytest.raises(ConfigError, match="total_egress_deadline"):
        cg.validate_config(cfg)
    with pytest.raises(ConfigError, match="f08_policy"):
        cg.validate_f08_deadline(40, "")


def test_validate_config_with_bridge_config_snapshot() -> None:
    """Проверка работы `validate_config` и `validate` поверх снимка `BridgeConfig` (включая `f08_policy == 'v2'`)."""
    raw = _valid_v3_config(pools={"direct": {"type": "direct"}}, rules=[])
    bridge = parse_config(raw)
    guarded = cg.validate_config(bridge)
    assert guarded.bridge is bridge and guarded.route_de_egress_id == ""
    v_guarded = cg.validate(parse_config(_valid_v3_config()))
    assert (v_guarded.max_inflight_per_backend, v_guarded.max_inflight_long_header) == (40, 8)
    assert (v_guarded.f08_policy, v_guarded.route_de_egress_id) == ("v1", "de-213.165.42.220")
    v2_cfg = _valid_v3_config()
    v2_cfg["server"].update(f08_policy="v2", f08_v2_risk_ack="AD-003/2026-11-14", max_inflight_long_header=12)
    v2_guarded = cg.validate(parse_config(v2_cfg))
    assert (v2_guarded.f08_policy, v2_guarded.f08_v2_risk_ack, v2_guarded.max_inflight_long_header) == (
        "v2",
        "AD-003/2026-11-14",
        12,
    )
    assert cg.validate_f08_deadline(40, "v2", f08_v2_risk_ack="AD-003/2026-11-14") == "v2"
    assert cg.validate_f08_deadline(40, "v2", client_timeout_s=180.0) == "v2"
    with pytest.raises(ConfigError):
        cg.validate_f08_deadline(40, "v2", f08_v2_risk_ack="bad")


def test_cli_entrypoint_returns_zero_on_valid_and_nonzero_on_invalid(tmp_path: Path) -> None:
    """CLI `python -m model_georouter.config_guard <path>` возвращает 0 на валидном конфиге и != 0 при нарушении."""
    valid_file = tmp_path / "valid.json"
    valid_file.write_text(json.dumps(_valid_v3_config()), encoding="utf-8")
    assert cg.load_and_validate_config(valid_file).route_de_egress_id == "de-213.165.42.220"
    assert (cg.main([str(valid_file)]), cg.main(["--config", str(valid_file)])) == (0, 0) and cg.main([]) != 0
    invalid_file = tmp_path / "invalid.json"
    bad_cfg = _valid_v3_config()
    bad_cfg["pools"]["route-de"]["proxies"] = ["http://127.0.0.1:10820"]
    invalid_file.write_text(json.dumps(bad_cfg), encoding="utf-8")
    assert cg.main([str(invalid_file)]) != 0
    proc = subprocess.run(
        [sys.executable, "-m", "model_georouter.config_guard", str(invalid_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0 and "10820" in proc.stderr
