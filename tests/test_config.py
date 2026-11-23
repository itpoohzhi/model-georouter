from __future__ import annotations

import dataclasses
import json
import logging
import os

import pytest

from model_georouter.config import AdapterConfig, ConfigManager, load_config_file, parse_config
from model_georouter.errors import ConfigError

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
    assert (server.retry_sleep_seconds, server.retry_backoff_factor, server.retry_max_sleep_seconds) == (0.5, 2.0, 3.0)
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


# Новые адаптеры из конфига (SG-пул вендоров): имя свободное, upstream и prefixes обязательны.
SG_CONFIG = {
    "upstreams": {"vendor-ai": {"host": "api.vendor.example", "base_path": "/v1"}},
    "pools": {"route-sg": {"type": "http_connect", "proxies": ["http://u:p@127.0.0.1:3128"]}},
    "adapters": {"sg-vendor": {"prefixes": ["/sg-vendor"], "upstream": "vendor-ai", "pool": "route-sg"}},
}


def test_custom_adapter_from_config_is_accepted():
    cfg = parse_config(minimal(**SG_CONFIG))
    adapter = cfg.adapters["sg-vendor"]
    assert adapter == AdapterConfig(("/sg-vendor",), "vendor-ai", "route-sg")
    assert cfg.upstreams["vendor-ai"].base_path == "/v1"


def test_custom_adapter_pool_is_optional_and_builtin_defaults_survive():
    cfg = parse_config(
        minimal(
            upstreams={"vendor-ai": {"host": "api.vendor.example"}},
            adapters={"sg-vendor": {"prefixes": ["/sg-vendor"], "upstream": "vendor-ai"}},
        )
    )
    assert cfg.adapters["sg-vendor"].pool == ""
    assert cfg.adapters["cloudcode"].pool == "route-de"  # встроенные адаптеры не задеты
    assert cfg.adapters["opencode"].prefixes == ("/v1", "/go/v1")


@pytest.mark.parametrize(
    "adapter, fragment",
    [
        ({"prefixes": ["/sg-vendor"]}, "new adapter requires 'upstream'"),
        ({"upstream": "self-ai"}, "new adapter requires 'prefixes'"),
        ({"upstream": "self-ai", "prefixes": []}, "prefixes"),
        ({"upstream": "self-ai", "prefixes": ["sg-vendor"]}, "prefix"),
        ({"upstream": "self-ai", "prefixes": ["/sg-vendor/"]}, "prefix"),
        ({"upstream": "self-ai", "prefixes": ["/sg-vendor"], "pool": "ghost"}, "unknown pool"),
    ],
)
def test_custom_adapter_requires_valid_fields(adapter, fragment):
    data = minimal(
        upstreams={"self-ai": {"host": "api.example.com"}},
        pools={"route-sg": {"type": "http_connect", "proxies": ["http://127.0.0.1:3128"]}},
        adapters={"sg-vendor": adapter},
    )
    with pytest.raises(ConfigError) as info:
        parse_config(data)
    assert fragment in str(info.value)


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
        (lambda d: d["server"].update(pre_send_retries=51), "pre_send_retries"),
        (lambda d: d["server"].update(retry_sleep_seconds=10**400), "retry_sleep_seconds"),
        (lambda d: d["server"].update(connect_timeout=10**400), "connect_timeout"),
        (lambda d: d["server"].update(retry_sleep_seconds=-0.1), "retry_sleep_seconds"),
        (lambda d: d["server"].update(retry_backoff_factor=0), "retry_backoff_factor"),
        (lambda d: d["server"].update(retry_max_sleep_seconds="3"), "retry_max_sleep_seconds"),
        (lambda d: d["server"].update(retry_sleep_seconds=float("inf")), "retry_sleep_seconds"),
        (lambda d: d["server"].update(retry_sleep_seconds=float("nan")), "retry_sleep_seconds"),
        (lambda d: d["server"].update(retry_sleep_seconds=30.5), "retry_sleep_seconds"),
        (lambda d: d["server"].update(retry_backoff_factor=float("inf")), "retry_backoff_factor"),
        (lambda d: d["server"].update(retry_backoff_factor=float("nan")), "retry_backoff_factor"),
        (lambda d: d["server"].update(retry_backoff_factor=-1.5), "retry_backoff_factor"),
        (lambda d: d["server"].update(retry_backoff_factor=10.5), "retry_backoff_factor"),
        (lambda d: d["server"].update(retry_max_sleep_seconds=float("-inf")), "retry_max_sleep_seconds"),
        (lambda d: d["server"].update(retry_max_sleep_seconds=float("nan")), "retry_max_sleep_seconds"),
        (lambda d: d["server"].update(retry_max_sleep_seconds=-1), "retry_max_sleep_seconds"),
        (lambda d: d["server"].update(retry_max_sleep_seconds=61), "retry_max_sleep_seconds"),
        (lambda d: d["server"].update(connect_timeout=float("inf")), "connect_timeout"),
        (lambda d: d["server"].update(connect_timeout=float("nan")), "connect_timeout"),
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
        (lambda d: d.update(adapters={"mystery": {}}), "new adapter requires"),
        (lambda d: d.update(adapters={"mystery": {"prefixes": ["/m"]}}), "new adapter requires"),
        (lambda d: d.update(adapters={"mystery": {"upstream": "ghost", "prefixes": ["/m"]}}), "unknown upstream"),
        (lambda d: d.update(adapters={"opencode": {"prefixes": ["v1"]}}), "prefix"),
        (lambda d: d.update(adapters={"cloudcode": {"pool": "ghost"}}), "pool"),
        (lambda d: d.update(upstreams={"x": {"port": 1}}), "host"),
    ],
)
def test_invalid_configs_are_rejected(mutate, fragment):
    data = minimal()
    mutate(data)
    with pytest.raises(ConfigError) as info:
        parse_config(data)
    assert fragment in str(info.value)


@pytest.mark.parametrize(
    "key, value",
    [
        ("retry_sleep_seconds", 0),
        ("retry_sleep_seconds", 30),
        ("retry_backoff_factor", 10),
        ("retry_max_sleep_seconds", 0),
        ("retry_max_sleep_seconds", 60.0),
    ],
)
def test_retry_settings_accept_boundaries(key, value):  # F-007: границы включительно
    data = minimal()
    data["server"][key] = value
    assert getattr(parse_config(data).server, key) == value


def test_pre_send_retries_accepts_ceiling():  # F-002 (cycle 2): потолок включительно
    data = minimal()
    data["server"]["pre_send_retries"] = 50
    assert parse_config(data).server.pre_send_retries == 50


@pytest.mark.parametrize("prefixes", [["a-", "b/"], "a-"])
def test_rule_prefixes_alias_maps_to_match_prefix(prefixes):  # F-001 (cycle 2)
    rule = parse_config(minimal(rules=[{"pool": "direct", "prefixes": prefixes}])).rules[0]
    assert rule.match_prefix == (tuple(prefixes) if isinstance(prefixes, list) else (prefixes,))


def test_canonical_match_prefix_wins_over_prefixes_alias():  # F-001 (cycle 2)
    rule = parse_config(minimal(rules=[{"pool": "direct", "match_prefix": ["x-"], "prefixes": ["y-"]}])).rules[0]
    assert rule.match_prefix == ("x-",)


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

    with caplog.at_level(logging.ERROR, logger="model_georouter.config"):
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
    with caplog.at_level(logging.ERROR, logger="model_georouter.config"):
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


# ───────────────────────────── cloudcode_claude_code ─────────────────────────────


def valid_claude_code_section(**overrides) -> dict:
    section = {
        "enabled": True,
        "sidecar_upstream": "cloudcode-claude-sidecar",
        "intercept_models": ["synthetic-claude", "google-claude"],
        "model_mapping": {
            "synthetic-claude": "claude-sonnet-5-5[1m]",
            "google-claude": "google-claude",
        },
        "google_fallback_models": ["google-claude"],
        "intercept_paths": [
            "/v1internal:fetchAvailableModels",
            "/v1internal:streamGenerateContent",
            "/v1internal:generateContent",
            "/v1internal:countTokens",
        ],
        "pre_commit_fallback": True,
        "sidecar_header_timeout_seconds": 45,
        "sidecar_keepalive_seconds": 39,
        "sidecar_non_stream_timeout_seconds": 40,
        "sidecar_idle_timeout_seconds": 120,
        "sidecar_max_execution_timeout_seconds": 300,
        "claude_bin": "~/.local/bin/claude",
        "claude_bin_pinned_version": "2.1.322",
        "max_concurrent_generations": 1,
        "catalog_model_map": {
            "synthetic-claude": {
                "donor_template_file": "fixtures/model-details.json",
                "donor_id": "google-claude",
                "display_name": "Claude Sonnet 5.5",
                "quota_text": "subscription",
            }
        },
    }
    section.update(overrides)
    return section


def config_with_claude_code(section=None, *, upstream=None, pools=None) -> dict:
    return minimal(
        upstreams={
            "cloudcode-claude-sidecar": upstream
            or {"host": "127.0.0.1", "port": 10835, "use_tls": False, "base_path": ""}
        },
        pools=pools or {},
        cloudcode_claude_code=valid_claude_code_section() if section is None else section,
    )


@pytest.mark.parametrize("section", [None, {}])
def test_cloudcode_claude_code_absent_or_empty_uses_off_defaults(section):
    data = minimal() if section is None else minimal(cloudcode_claude_code=section)
    cfg = parse_config(data).cloudcode_claude_code
    assert cfg.enabled is False
    assert cfg.sidecar_upstream == "cloudcode-claude-sidecar"
    assert cfg.intercept_models == ()
    assert dict(cfg.model_mapping) == {}
    assert cfg.google_fallback_models == ()
    assert cfg.intercept_paths == (
        "/v1internal:fetchAvailableModels",
        "/v1internal:streamGenerateContent",
        "/v1internal:generateContent",
        "/v1internal:countTokens",
    )
    assert cfg.pre_commit_fallback is True
    assert (
        cfg.sidecar_keepalive_seconds,
        cfg.sidecar_non_stream_timeout_seconds,
        cfg.sidecar_header_timeout_seconds,
        cfg.sidecar_idle_timeout_seconds,
        cfg.sidecar_max_execution_timeout_seconds,
    ) == (39, 40, 45, 120, 300)
    assert cfg.claude_bin == "~/.local/bin/claude"
    assert cfg.claude_bin_pinned_version == "2.1.322"
    assert cfg.max_concurrent_generations == 1
    assert dict(cfg.catalog_model_map) == {}


def test_cloudcode_claude_code_full_valid_section():
    cfg = parse_config(config_with_claude_code())
    section = cfg.cloudcode_claude_code
    assert section.enabled is True
    assert section.intercept_models == ("synthetic-claude", "google-claude")
    assert dict(section.model_mapping) == {
        "synthetic-claude": "claude-sonnet-5-5[1m]",
        "google-claude": "google-claude",
    }
    assert section.google_fallback_models == ("google-claude",)
    assert section.pre_commit_fallback is True
    catalog = section.catalog_model_map["synthetic-claude"]
    assert catalog.donor_template_file == "fixtures/model-details.json"
    assert catalog.donor_id == "google-claude"
    assert catalog.display_name == "Claude Sonnet 5.5"
    assert catalog.quota_text == "subscription"


def test_cloudcode_claude_code_snapshot_is_deeply_immutable():
    section = parse_config(config_with_claude_code()).cloudcode_claude_code
    with pytest.raises(dataclasses.FrozenInstanceError):
        section.enabled = False  # type: ignore[misc]
    with pytest.raises(TypeError):
        section.model_mapping["other"] = "other"  # type: ignore[index]
    with pytest.raises(TypeError):
        section.catalog_model_map["other"] = None  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        section.catalog_model_map["synthetic-claude"].donor_id = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda s: s["model_mapping"].pop("synthetic-claude"), "missing exact mapping"),
        (lambda s: s["model_mapping"].update({"synthetic-claude": ""}), "claude_model_id"),
        (lambda s: s["model_mapping"].update({"synthetic-claude": "claude sonnet"}), "whitespace"),
        (lambda s: s["model_mapping"].update({"synthetic-claude": "-claude-sonnet"}), "leading '-'"),
    ],
)
def test_cloudcode_claude_code_rejects_invalid_exact_model_mapping(mutate, fragment):
    section = valid_claude_code_section()
    mutate(section)
    with pytest.raises(ConfigError, match=fragment):
        parse_config(config_with_claude_code(section))


@pytest.mark.parametrize(
    "route_path",
    ["v1internal:generateContent", "/cloudcode/v1internal:generateContent", "/v1internal:generateContent/"],
)
def test_cloudcode_claude_code_rejects_non_normalized_intercept_paths(route_path):
    section = valid_claude_code_section(intercept_paths=[route_path])
    with pytest.raises(ConfigError, match="intercept_paths"):
        parse_config(config_with_claude_code(section))


def test_cloudcode_claude_code_rejects_fallback_outside_intercept_models():
    section = valid_claude_code_section(google_fallback_models=["unknown-google-model"])
    with pytest.raises(ConfigError, match="not a subset"):
        parse_config(config_with_claude_code(section))


def test_cloudcode_claude_code_rejects_catalog_entry_outside_intercept_models():
    section = valid_claude_code_section()
    section["catalog_model_map"]["unknown"] = {"donor_template_file": "fixtures/donor.json"}
    with pytest.raises(ConfigError, match="not in intercept_models"):
        parse_config(config_with_claude_code(section))


@pytest.mark.parametrize("entry", [{}, {"donor_template_file": ""}, {"donor_template_file": "   "}])
def test_cloudcode_claude_code_requires_catalog_donor_template(entry):
    section = valid_claude_code_section()
    section["catalog_model_map"]["synthetic-claude"] = entry
    with pytest.raises(ConfigError, match="donor_template_file"):
        parse_config(config_with_claude_code(section))


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"sidecar_keepalive_seconds": 0}, "sidecar_keepalive_seconds"),
        ({"sidecar_keepalive_seconds": 40}, "sidecar_keepalive_seconds"),
        ({"sidecar_non_stream_timeout_seconds": 45}, "sidecar_header_timeout_seconds"),
        ({"sidecar_header_timeout_seconds": 60}, "sidecar_header_timeout_seconds"),
        ({"sidecar_idle_timeout_seconds": 45}, "sidecar_idle_timeout_seconds"),
        ({"sidecar_max_execution_timeout_seconds": 100}, "sidecar_max_execution_timeout_seconds"),
        ({"sidecar_max_execution_timeout_seconds": 1801}, "sidecar_max_execution_timeout_seconds"),
        ({"max_concurrent_generations": 2}, "max_concurrent_generations"),
    ],
)
def test_cloudcode_claude_code_rejects_invalid_timeout_chain_or_concurrency(overrides, fragment):
    section = valid_claude_code_section(**overrides)
    with pytest.raises(ConfigError, match=fragment):
        parse_config(config_with_claude_code(section))


def test_cloudcode_claude_code_rejects_unknown_sidecar_upstream():
    section = valid_claude_code_section(sidecar_upstream="ghost")
    with pytest.raises(ConfigError, match="unknown upstream"):
        parse_config(config_with_claude_code(section))


@pytest.mark.parametrize(
    "upstream",
    [
        {"host": "192.0.2.1", "port": 10835, "use_tls": False},
        {"host": "127.0.0.1", "port": 10835, "use_tls": True},
    ],
)
def test_cloudcode_claude_code_rejects_remote_or_tls_sidecar(upstream):
    with pytest.raises(ConfigError, match="loopback HTTP"):
        parse_config(config_with_claude_code(upstream=upstream))


def test_cloudcode_claude_code_rejects_non_direct_pool():
    pools = {"direct": {"type": "http_connect", "proxies": ["http://127.0.0.1:3128"]}}
    with pytest.raises(ConfigError, match="direct pool"):
        parse_config(config_with_claude_code(pools=pools))


@pytest.mark.parametrize("claude_bin", ["claude", "relative/claude", ""])
def test_cloudcode_claude_code_rejects_non_absolute_claude_bin(claude_bin):
    section = valid_claude_code_section(claude_bin=claude_bin)
    with pytest.raises(ConfigError, match="claude_bin"):
        parse_config(config_with_claude_code(section))


@pytest.mark.parametrize("version", ["", "   "])
def test_cloudcode_claude_code_rejects_empty_pinned_version(version):
    section = valid_claude_code_section(claude_bin_pinned_version=version)
    with pytest.raises(ConfigError, match="claude_bin_pinned_version"):
        parse_config(config_with_claude_code(section))


def test_cloudcode_claude_code_hot_reload_on_to_off_preserves_old_snapshot(tmp_path):
    path = tmp_path / "config.json"
    data = config_with_claude_code()
    write_config(path, data, 1)
    manager = ConfigManager(path, poll_interval=0)
    held = manager.get()
    assert held.cloudcode_claude_code.enabled is True

    data["cloudcode_claude_code"]["enabled"] = False
    data["cloudcode_claude_code"]["model_mapping"]["synthetic-claude"] = "claude-opus-5-5"
    write_config(path, data, 2)
    fresh = manager.get()
    assert fresh is not held
    assert fresh.cloudcode_claude_code.enabled is False
    assert fresh.cloudcode_claude_code.model_mapping["synthetic-claude"] == "claude-opus-5-5"
    assert held.cloudcode_claude_code.enabled is True
    assert held.cloudcode_claude_code.model_mapping["synthetic-claude"] == "claude-sonnet-5-5[1m]"
