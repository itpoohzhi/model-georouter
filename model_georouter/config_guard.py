"""Стартовая валидация конфигурации v3 (W1-B): инварианты loopback, route-de, inflight и дедлайна."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .config import FORBIDDEN_ROUTE_DE_PORTS, LOOPBACK, BridgeConfig, PoolConfig, parse_config
from .errors import ConfigError
from .proxy_pool import parse_proxy_url
from .retry_budget import F08_V2_POLICY, F08_V2_RISK_ACK, validate_f08_v2

# fmt: off
__all__ = ["ALLOWED_ROUTE_DE_BACKENDS", "ALLOWED_ROUTE_DE_PORTS", "DEFAULT_F08_POLICY", "F08_V1_TOTAL_EGRESS_DEADLINE", "F08_V2_POLICY", "F08_V2_RISK_ACK", "FORBIDDEN_ROUTE_DE_PORTS", "ROUTE_DE_POOL", "GuardedConfig", "load_and_validate_config", "main", "validate", "validate_config", "validate_config_file", "validate_f08_deadline", "validate_inflight_limits", "validate_listen_host", "validate_route_de_pool"]
ROUTE_DE_POOL, DEFAULT_F08_POLICY, F08_V1_TOTAL_EGRESS_DEADLINE, _MISSING = "route-de", "v1", 40.0, object()
ALLOWED_ROUTE_DE_PORTS: frozenset[int] = frozenset({10827, 10828, 10829})
ALLOWED_ROUTE_DE_BACKENDS: frozenset[str] = frozenset(f"{LOOPBACK}:{p}" for p in ALLOWED_ROUTE_DE_PORTS)
ObjMap = Mapping[str, object]

@dataclass(frozen=True, slots=True)
class GuardedConfig:
    """Результат проверки конфигурации v3 со всеми подтверждёнными инвариантами."""
    bridge: BridgeConfig
    max_inflight_per_backend: int
    max_inflight_long_header: int
    f08_policy: str
    route_de_egress_id: str = ""
    f08_v2_risk_ack: str | None = None
    client_timeout_s: float | None = None

def _int_bound(val: object, name: str, low: int, high: int | None = None) -> int:
    if isinstance(val, bool) or not isinstance(val, int) or val < low or (high is not None and val > high):
        raise ConfigError(f"{name}: invalid integer {val!r}")
    return val
def validate_listen_host(listen: object) -> str:
    """Разрешает слушать строго только на IPv4 loopback `127.0.0.1`."""
    if not isinstance(listen, str) or listen != LOOPBACK:
        raise ConfigError(f"server.listen: must be {LOOPBACK!r}, got {listen!r}")
    return listen
def validate_inflight_limits(proxy_slots: object, max_inflight_per_backend: object = _MISSING, max_inflight_long_header: object = _MISSING) -> tuple[int, int]:
    """Проверяет `0 < max_inflight_per_backend <= proxy_slots // 2` (D2-M4) и `max_inflight_long_header > 0`."""
    limit = _int_bound(proxy_slots, "server.proxy_slots", 2) // 2
    per_b = _int_bound(min(40, limit) if max_inflight_per_backend is _MISSING else max_inflight_per_backend, "server.max_inflight_per_backend", 1, limit)
    long_h = _int_bound(64 if max_inflight_long_header is _MISSING else max_inflight_long_header, "server.max_inflight_long_header", 1)
    return per_b, long_h
def validate_f08_deadline(total_egress_deadline: object, f08_policy: object = _MISSING, f08_v2_risk_ack: object = _MISSING, client_timeout_s: object = _MISSING) -> str:
    """Проверяет дедлайн F-08 (`v1`: `total_egress_deadline == 40`, `v2`: `f08_v2_risk_ack` или `client_timeout_s`)."""
    policy = DEFAULT_F08_POLICY if f08_policy is _MISSING or f08_policy is None else f08_policy
    if not isinstance(policy, str) or not policy.strip() or policy not in (DEFAULT_F08_POLICY, F08_V2_POLICY):
        raise ConfigError(f"server.f08_policy: must be 'v1' or 'v2', got {policy!r}")
    if isinstance(total_egress_deadline, bool) or not isinstance(total_egress_deadline, int | float):
        raise ConfigError("server.total_egress_deadline: expected number")
    if policy == DEFAULT_F08_POLICY and float(total_egress_deadline) != F08_V1_TOTAL_EGRESS_DEADLINE:
        raise ConfigError(f"server.total_egress_deadline: must be 40 when f08_policy == 'v1', got {total_egress_deadline!r}")
    if policy == F08_V2_POLICY:
        ack = F08_V2_RISK_ACK if (f08_v2_risk_ack is _MISSING and client_timeout_s is _MISSING) else (None if f08_v2_risk_ack is _MISSING else f08_v2_risk_ack)
        _ = validate_f08_v2(client_timeout_s=None if client_timeout_s is _MISSING else client_timeout_s, f08_v2_risk_ack=ack, non_stream_deadline=F08_V1_TOTAL_EGRESS_DEADLINE if float(total_egress_deadline) in (20.0, 60.0) else total_egress_deadline)
    return policy
def validate_route_de_pool(pool: ObjMap | PoolConfig, *, egress_id: object = _MISSING, require_allowed_ports: bool = True) -> str:
    """Проверяет пул `route-de`: непустой `egress_id` (D3-M4) и только loopback 127.0.0.1:10827..10829."""
    raw_eg = (getattr(pool, "egress_id", None) if isinstance(pool, PoolConfig) else pool.get("egress_id")) if egress_id is _MISSING else egress_id
    pool_type = pool.type if isinstance(pool, PoolConfig) else pool.get("type")
    raw_p = pool.proxies if isinstance(pool, PoolConfig) else pool.get("proxies", pool.get("urls"))
    proxies = cast(Sequence[object], raw_p) if isinstance(raw_p, list | tuple) else ()
    if require_allowed_ports and (not isinstance(raw_eg, str) or not raw_eg.strip()):
        raise ConfigError("pools.route-de.egress_id: must be a non-empty string")
    if pool_type == "direct" or not proxies:
        raise ConfigError("pools.route-de.proxies: non-empty list of dedicated DE proxies required (not 'direct')")
    for idx, url in enumerate(proxies):
        if not isinstance(url, str) or not url:
            raise ConfigError(f"pools.route-de.proxies[{idx}]: proxy URL must be a non-empty string")
        try:
            ep = parse_proxy_url(url)
        except ValueError as exc:
            raise ConfigError(f"pools.route-de.proxies[{idx}]: invalid proxy URL ({exc})") from None
        if ep.port in FORBIDDEN_ROUTE_DE_PORTS or ep.host != LOOPBACK or (require_allowed_ports and ep.port not in ALLOWED_ROUTE_DE_PORTS):
            raise ConfigError(f"pools.route-de.proxies[{idx}]: forbidden or non-dedicated DE proxy {ep.host}:{ep.port} (:{ep.port})")
    return raw_eg.strip() if isinstance(raw_eg, str) else ""
def _build_guarded(bridge: BridgeConfig, *, strict: bool) -> GuardedConfig:
    srv, eg_id = bridge.server, ""
    _ = validate_listen_host(srv.listen)
    if (pool := bridge.pools.get(ROUTE_DE_POOL)) is not None:
        raw_eg = getattr(pool, "egress_id", None)
        _ = validate_route_de_pool(pool, require_allowed_ports=False)
        ephemeral = not strict and srv.port == 0 and raw_eg is None and (srv.total_egress_deadline != F08_V1_TOTAL_EGRESS_DEADLINE or all(parse_proxy_url(u).port >= 10830 for u in pool.proxies))
        eg_id = "" if ephemeral else validate_route_de_pool(pool)
    raw_b, raw_h = getattr(srv, "max_inflight_per_backend", None), getattr(srv, "max_inflight_long_header", None)
    eff_slots = srv.proxy_slots if (strict or srv.port != 0 or raw_b is not None or raw_h is not None or srv.proxy_slots >= 2) else 2
    per_b, long_h = validate_inflight_limits(eff_slots, _MISSING if raw_b is None else raw_b, _MISSING if raw_h is None else raw_h)
    raw_pol, raw_ack, raw_ct = getattr(srv, "f08_policy", None), getattr(srv, "f08_v2_risk_ack", None), getattr(srv, "client_timeout_s", None)
    enforce_f08 = strict or srv.port != 0 or raw_pol is not None or raw_ack is not None or raw_ct is not None or srv.total_egress_deadline == F08_V1_TOTAL_EGRESS_DEADLINE
    pol = validate_f08_deadline(srv.total_egress_deadline, _MISSING if raw_pol is None else raw_pol, raw_ack, raw_ct) if enforce_f08 else DEFAULT_F08_POLICY
    if not isinstance(cast(object, srv.hedged_connect_enabled), bool) or (pol == F08_V2_POLICY and srv.hedged_connect_enabled):
        raise ConfigError("server.hedged_connect_enabled: expected boolean, must be False for v2")
    ack = raw_ack if isinstance(raw_ack, str) and pol == F08_V2_POLICY else None
    ct = float(raw_ct) if isinstance(raw_ct, int | float) and not isinstance(raw_ct, bool) and pol == F08_V2_POLICY else None
    return GuardedConfig(bridge, per_b, long_h, pol, eg_id, ack, ct)
def validate_config(data: object) -> GuardedConfig:
    """Полная валидация словаря конфигурации или снимка `BridgeConfig` по правилам W1-B."""
    if isinstance(data, BridgeConfig):
        return _build_guarded(data, strict=True)
    if not isinstance(data, Mapping) or not isinstance(cast(ObjMap, data).get("server"), Mapping):
        raise ConfigError("config: expected mapping with 'server' object")
    root, raw_s = cast(ObjMap, data), cast(ObjMap, cast(ObjMap, data)["server"])
    if "hedged_connect_enabled" in raw_s and not isinstance(raw_s["hedged_connect_enabled"], bool):
        raise ConfigError("server.hedged_connect_enabled: expected boolean")
    _ = validate_listen_host(raw_s.get("listen", raw_s.get("host")))
    if isinstance(raw_p := root.get("pools"), Mapping) and ROUTE_DE_POOL in raw_p:
        if not isinstance(rde := cast(ObjMap, raw_p).get(ROUTE_DE_POOL), Mapping):
            raise ConfigError("pools.route-de: expected object")
        _ = validate_route_de_pool(cast(ObjMap, rde))
    return _build_guarded(parse_config(root), strict=True)
def validate(data: BridgeConfig) -> GuardedConfig:
    """Проверяет снимок `BridgeConfig` и возвращает подтверждённый `GuardedConfig`."""
    return _build_guarded(data, strict=False)
def load_and_validate_config(path: str | Path) -> GuardedConfig:
    """Читает JSON-файл конфигурации с диска и проверяет его через `validate_config`."""
    try:
        return validate_config(cast(object, json.loads(Path(path).expanduser().read_text(encoding="utf-8"))))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot load config {path}: {exc}") from None
validate_config_file = load_and_validate_config
def main(argv: Sequence[str] | None = None) -> int:
    """CLI-вход: `python -m model_georouter.config_guard <path>`."""
    parser = argparse.ArgumentParser(prog="python -m model_georouter.config_guard")
    _ = (parser.add_argument("path", nargs="?", type=Path), parser.add_argument("--config", "-c", dest="config_opt", type=Path))
    ns = parser.parse_args(list(argv) if argv is not None else None)
    target = cast(Path | None, getattr(ns, "config_opt", None) or getattr(ns, "path", None))
    try:
        if target is None:
            raise ConfigError("path to config.json is required")
        _ = load_and_validate_config(target)
    except ConfigError as exc:
        print(f"config_guard error: {exc}", file=sys.stderr)
        return 2 if target is None else 1
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
