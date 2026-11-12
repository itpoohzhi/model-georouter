"""Единственный прокси без альтернативы не штрафуется за upstream header timeout (route-de 502)."""

from __future__ import annotations

import dataclasses

import pytest

from model_georouter.config import ConfigError, parse_config
from model_georouter.retry_budget import AttemptEvent, AttemptState, transition
from tests.helpers import FakeConnectProxy, FakeServer, HttpUpstream, free_port, http_request, make_config, respond

HEADER_TIMEOUT = AttemptEvent("header_timeout", has_endpoint=True)


def _penalize(**overrides) -> bool:
    state = AttemptState(pool="route-de", **overrides)
    return transition(state, HEADER_TIMEOUT).should_penalize


def test_header_timeout_without_alternative_is_not_penalized_when_flag_off():
    assert _penalize(has_alternative=False, penalize_without_alternative=False) is False


@pytest.mark.parametrize(
    "overrides",
    [
        {},  # дефолты = прежнее поведение
        {"has_alternative": False},  # флаг по умолчанию true
        {"has_alternative": False, "penalize_without_alternative": True},
        {"has_alternative": True, "penalize_without_alternative": False},
        {
            "single_proxy": True,
            "fallback_pool": "route-de",
            "has_alternative": False,
            "penalize_without_alternative": False,
        },
    ],
)
def test_header_timeout_penalty_preserved_elsewhere(overrides):
    assert _penalize(**overrides) is True


def test_header_timeout_with_alternative_still_respects_penalize_on_header_timeout():
    assert _penalize(has_alternative=True, penalize_on_header_timeout=False) is False


def test_config_flag_default_and_strict_bool():
    base = make_config(443)
    assert parse_config(base).server.penalize_header_timeout_without_alternative is True
    off = make_config(443, server={"penalize_header_timeout_without_alternative": False})
    assert parse_config(off).server.penalize_header_timeout_without_alternative is False
    for bad in ("false", 0, 1, None):
        cfg = make_config(443, server={"penalize_header_timeout_without_alternative": bad})
        with pytest.raises(ConfigError, match="penalize_header_timeout_without_alternative"):
            parse_config(cfg)


def _single_route_de_bridge(track, start_bridge, proxy, upstream_port: int, **server):
    pools = {"route-de": {"type": "http_connect", "proxies": [f"http://127.0.0.1:{proxy.port}"]}}
    rules = [{"match_prefix": ["m"], "pool": "route-de"}]
    cfg = make_config(
        upstream_port,
        pools=pools,
        rules=rules,
        geo_fallback_pool="route-de",  # self-ref как в живом route-de: fallback == pool
        server={"headers_timeout": 0.2, **server},
    )
    return start_bridge(cfg)


def _post(bridge):
    return http_request(bridge.port, "POST", "/v1/chat/completions", b'{"model":"m"}')


def test_single_proxy_not_penalized_by_header_timeout_when_flag_off(track, start_bridge):
    silent = track(FakeServer(lambda conn: (conn.recv(1024), silent.stop.wait(1.0))))
    proxy = track(FakeConnectProxy(target=("127.0.0.1", silent.port)))
    bridge = _single_route_de_bridge(
        track, start_bridge, proxy, silent.port, penalize_header_timeout_without_alternative=False
    )
    assert _post(bridge).status == 504
    endpoint = bridge.server.proxy_manager.endpoints(bridge.server.config_manager.get().pools["route-de"])[0]
    assert not bridge.server.proxy_manager.is_penalized(endpoint)
    assert _post(bridge).status == 504  # второй запрос дошёл до апстрима, а не 502 proxy_unavailable
    assert len(proxy.connects) == 2


def test_single_proxy_penalized_by_header_timeout_by_default(track, start_bridge):
    silent = track(FakeServer(lambda conn: (conn.recv(1024), silent.stop.wait(1.0))))
    proxy = track(FakeConnectProxy(target=("127.0.0.1", silent.port)))
    bridge = _single_route_de_bridge(track, start_bridge, proxy, silent.port)
    assert _post(bridge).status == 504
    endpoint = bridge.server.proxy_manager.endpoints(bridge.server.config_manager.get().pools["route-de"])[0]
    assert bridge.server.proxy_manager.is_penalized(endpoint)
    assert _post(bridge).status == 502
    assert len(proxy.connects) == 1


def test_dead_proxy_still_penalized_when_flag_off(track, start_bridge):
    up = track(HttpUpstream(lambda conn, req: respond(conn, 200, b"ok")))

    class _Dead:
        port = free_port()

    bridge = _single_route_de_bridge(
        track, start_bridge, _Dead, up.port, penalize_header_timeout_without_alternative=False
    )
    assert _post(bridge).status == 502
    endpoint = bridge.server.proxy_manager.endpoints(bridge.server.config_manager.get().pools["route-de"])[0]
    assert bridge.server.proxy_manager.is_penalized(endpoint)
    assert _post(bridge).status == 502


def test_attempt_state_defaults_keep_old_behaviour():
    state = AttemptState(pool="p")
    assert dataclasses.asdict(state)["has_alternative"] is True
    assert dataclasses.asdict(state)["penalize_without_alternative"] is True
