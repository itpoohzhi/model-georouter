from __future__ import annotations

import dataclasses
import json
import logging
import os

import pytest

from universal_ai_bridge.config import ConfigManager, load_config_file, parse_config
from universal_ai_bridge.errors import ConfigError

MINIMAL = {"server": {"listen": "127.0.0.1", "port": 10830}, "pools": {}, "rules": []}


def minimal(**overrides) -> dict:
    data = json.loads(json.dumps(MINIMAL))
    data.update(overrides)
    return data


def test_minimal_config_gets_spec_defaults():
    cfg = parse_config(MINIMAL)
    server = cfg.server
    assert (server.connect_timeout, server.total_egress_deadline) == (12.0, 20.0)
    assert (server.headers_timeout, server.inactivity_timeout, server.body_timeout) == (120.0, 300.0, 30.0)
    assert server.pre_send_retries == 1
    assert server.body_buffer_max_bytes == 16777216
    assert (server.direct_slots, server.proxy_slots) == (16, 16)
    assert server.proxy_fail_penalty_seconds == 30
    assert server.geo_cache_ttl_seconds == 86400
    assert "RegionError" in server.geo_error_signatures
    assert cfg.default_pool == "direct"
    assert cfg.pools["direct"].type == "direct"  # direct добавляется автоматически
    assert cfg.geo_fallback_pool == ""  # route-de не объявлен — replay отключён


def test_default_adapters_and_upstreams():
    cfg = parse_config(MINIMAL)
    assert cfg.adapters["opencode"].prefixes == ("/v1", "/go/v1")
    assert cfg.adapters["cordis"].prefixes == ("/zen/v1", "/zen/go/v1")
    assert cfg.adapters["generic"].prefixes == ("/messages", "/chat/completions")
    assert cfg.adapters["opencode"].upstream in cfg.upstreams


def test_explicit_values_override_defaults():
    data = minimal()
    data["server"].update(direct_slots=2, proxy_slots=3, connect_timeout=1.5, geo_error_signatures=["blocked-x"])
    data["pools"] = {"route-de": {"type": "http_connect", "proxies": ["http://u:p@127.0.0.1:3128"]}}
    cfg = parse_config(data)
    assert (cfg.server.direct_slots, cfg.server.proxy_slots, cfg.server.connect_timeout) == (2, 3, 1.5)
    assert cfg.server.geo_error_signatures == ("blocked-x",)
    assert cfg.geo_fallback_pool == "route-de"  # дефолт схемы, раз пул объявлен
    assert cfg.pools["route-de"].strategy == "first_available"
    assert cfg.pools["route-de"].remote_dns is True


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda d: d.pop("server"), "server"),
        (lambda d: d.pop("pools"), "pools"),
        (lambda d: d.pop("rules"), "rules"),
        (lambda d: d["server"].update(listen="0.0.0.0"), "listen"),
        (lambda d: d["server"].pop("port"), "port"),
        (lambda d: d["server"].update(port="10830"), "port"),
        (lambda d: d["server"].update(port=True), "port"),
        (lambda d: d["server"].update(port=70000), "port"),
        (lambda d: d["server"].update(connect_timeout=0), "connect_timeout"),
        (lambda d: d["server"].update(headers_timeout="x"), "headers_timeout"),
        (lambda d: d["server"].update(direct_slots=0), "direct_slots"),
        (lambda d: d["server"].update(pre_send_retries=-1), "pre_send_retries"),
        (lambda d: d["server"].update(geo_error_signatures="RegionError"), "geo_error_signatures"),
        (lambda d: d["server"].update(geo_error_signatures=[""]), "geo_error_signatures"),
        (lambda d: d.update(pools={"p": {"type": "vpn"}}), "type"),
        (lambda d: d.update(pools={"p": {}}), "type"),
        (lambda d: d.update(pools={"p": {"type": "direct", "strategy": "round_robin"}}), "strategy"),
        (lambda d: d.update(pools={"p": {"type": "http_connect"}}), "proxies"),
        (lambda d: d.update(pools={"p": {"type": "socks5", "proxies": ["http://h:1"]}}), "scheme"),
        (lambda d: d.update(pools={"p": {"type": "http_connect", "proxies": ["socks5://h:1"]}}), "scheme"),
        (lambda d: d.update(pools={"p": {"type": "http_connect", "proxies": ["ftp://h:1"]}}), "proxy URL"),
        (lambda d: d.update(rules=[{"pool": "nope", "match_prefix": ["a"]}]), "unknown pool"),
        (lambda d: d.update(rules=[{"match_prefix": ["a"]}]), "pool"),
        (lambda d: d.update(rules=[{"pool": "direct"}]), "match_prefix"),
        (lambda d: d.update(rules=[{"pool": "direct", "match_regex": "("}]), "regular expression"),
        (lambda d: d.update(default_pool="ghost"), "default_pool"),
        (lambda d: d.update(geo_fallback_pool="ghost"), "geo_fallback_pool"),
        (lambda d: d.update(adapters={"opencode": {"upstream": "ghost"}}), "upstream"),
        (lambda d: d.update(adapters={"mystery": {}}), "unknown adapter"),
        (lambda d: d.update(adapters={"opencode": {"prefixes": ["v1"]}}), "prefix"),
        (lambda d: d.update(upstreams={"x": {"port": 1}}), "host"),
    ],
)
def test_invalid_configs_are_rejected(mutate, fragment):
    data = minimal()
    mutate(data)
    with pytest.raises(ConfigError) as info:
        parse_config(data)
    assert fragment in str(info.value)


def test_non_object_config_rejected():
    with pytest.raises(ConfigError):
        parse_config([])


def test_snapshot_is_immutable():
    cfg = parse_config(MINIMAL)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.server.port = 1  # type: ignore[misc]
    with pytest.raises(TypeError):
        cfg.pools["x"] = None  # type: ignore[index]


def test_load_config_file_invalid_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not json")
    with pytest.raises(ConfigError, match="invalid JSON"):
        load_config_file(path)


# ───────────────────────────── hot-reload ─────────────────────────────


def write_config(path, data, bump):
    path.write_text(json.dumps(data))
    stamp = 1_700_000_000 + bump
    os.utime(path, (stamp, stamp))


def test_hot_reload_swaps_snapshot_atomically(tmp_path):
    path = tmp_path / "config.json"
    data = minimal()
    write_config(path, data, 1)
    manager = ConfigManager(path, poll_interval=0)
    held = manager.get()  # активный запрос держит «свой» снимок
    assert held.server.direct_slots == 16
    assert manager.reload_if_changed() is False  # файл не менялся

    data["server"]["direct_slots"] = 4
    write_config(path, data, 2)
    fresh = manager.get()
    assert fresh is not held
    assert fresh.server.direct_slots == 4
    assert held.server.direct_slots == 16  # прежний снимок не мутирован
    assert manager.get() is fresh  # без изменений снимок тот же


def test_invalid_reload_keeps_previous_snapshot_and_logs(tmp_path, caplog):
    path = tmp_path / "config.json"
    write_config(path, minimal(), 1)
    manager = ConfigManager(path, poll_interval=0)
    good = manager.get()

    with caplog.at_level(logging.ERROR, logger="universal_ai_bridge.config"):
        path.write_text("{broken")
        os.utime(path, (1_700_000_100, 1_700_000_100))
        assert manager.get() is good
        assert "keeping previous snapshot" in caplog.text
    assert manager.last_error is not None

    bad = minimal()
    bad["server"]["port"] = "nope"
    write_config(path, bad, 3)
    assert manager.get() is good  # валидный по JSON, но невалидный по схеме

    ok = minimal()
    ok["server"]["proxy_slots"] = 7
    write_config(path, ok, 4)
    recovered = manager.get()
    assert recovered is not good and recovered.server.proxy_slots == 7
    assert manager.last_error is None


def test_invalid_file_is_not_re_logged_until_it_changes(tmp_path, caplog):
    path = tmp_path / "config.json"
    write_config(path, minimal(), 1)
    manager = ConfigManager(path, poll_interval=0)
    path.write_text("{broken")
    os.utime(path, (1_700_000_100, 1_700_000_100))
    with caplog.at_level(logging.ERROR, logger="universal_ai_bridge.config"):
        manager.get()
        manager.get()
        manager.get()
    assert caplog.text.count("config reload rejected") == 1


def test_missing_file_after_start_keeps_snapshot(tmp_path):
    path = tmp_path / "config.json"
    write_config(path, minimal(), 1)
    manager = ConfigManager(path, poll_interval=0)
    snapshot = manager.get()
    path.unlink()
    assert manager.get() is snapshot
    assert "unavailable" in manager.last_error


def test_poll_interval_throttles_stat_checks(tmp_path):
    path = tmp_path / "config.json"
    data = minimal()
    write_config(path, data, 1)
    now = [0.0]
    manager = ConfigManager(path, poll_interval=10.0, clock=lambda: now[0])
    first = manager.get()
    data["server"]["direct_slots"] = 3
    write_config(path, data, 2)
    now[0] = 5.0
    assert manager.get() is first
    now[0] = 11.0
    assert manager.get().server.direct_slots == 3


def test_invalid_initial_file_raises(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"server": {}}))
    with pytest.raises(ConfigError):
        ConfigManager(path)
