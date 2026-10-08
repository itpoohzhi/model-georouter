"""Слой 5: валидация JSON-конфигурации (по config-schema.json) и атомарный hot-reload."""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .errors import ConfigError
from .logging_utils import get_logger
from .proxy_pool import parse_proxy_url

__all__ = [
    "AdapterConfig",
    "BridgeConfig",
    "ConfigError",
    "ConfigManager",
    "PoolConfig",
    "RuleConfig",
    "ServerConfig",
    "UpstreamConfig",
    "default_config_path",
    "default_log_dir",
    "load_config_file",
    "parse_config",
]

LOGGER = get_logger("universal_ai_bridge.config")

LOOPBACK = "127.0.0.1"
POOL_TYPES = ("direct", "http_connect", "socks5")
STRATEGIES = ("first_available", "random")
PROXY_SCHEMES = {"http_connect": {"http"}, "socks5": {"socks5", "socks5h"}}
DEFAULT_GEO_SIGNATURES = (
    "RegionError",
    "not available in your region",
    "country not supported",
    "location not supported",
    "geoblocked",
)
CONFIG_HOME = "~/.config/model-georouter"
LEGACY_CONFIG_HOME = "~/.config/universal-ai-bridge"


def _resolve_default(relative: str) -> str:
    """`model-georouter` главнее; legacy `universal-ai-bridge` — только если нового пути нет, а старый уже существует."""
    current, legacy = f"{CONFIG_HOME}/{relative}", f"{LEGACY_CONFIG_HOME}/{relative}"
    if not Path(current).expanduser().exists() and Path(legacy).expanduser().exists():
        return legacy
    return current


def default_config_path() -> Path:
    """Путь `config.json` по умолчанию (с `~`): `model-georouter` → legacy `universal-ai-bridge`."""
    return Path(_resolve_default("config.json"))


def default_log_dir() -> str:
    """Каталог логов по умолчанию (с `~`): `model-georouter/logs` → legacy `universal-ai-bridge/logs`."""
    return _resolve_default("logs")


@dataclass(frozen=True)
class ServerConfig:
    listen: str
    port: int
    direct_slots: int = 16
    proxy_slots: int = 16
    max_connections: int = 256
    ingress_wait_timeout: float = 5.0  # deprecated/ignored: лимит соединений отказывает сразу; ключ только валидируется
    connect_timeout: float = 12.0
    total_egress_deadline: float = 20.0
    headers_timeout: float = 120.0
    inactivity_timeout: float = 300.0
    body_timeout: float = 30.0
    body_buffer_max_bytes: int = 16777216
    pre_send_retries: int = 1
    geo_cache_ttl_seconds: int = 86400
    proxy_fail_penalty_seconds: int = 30
    log_dir: str = field(default_factory=default_log_dir)
    geo_error_signatures: tuple[str, ...] = DEFAULT_GEO_SIGNATURES
    geo_cache_file: str = ""


@dataclass(frozen=True)
class UpstreamConfig:
    host: str
    port: int = 443
    use_tls: bool = True
    base_path: str = ""


@dataclass(frozen=True)
class AdapterConfig:
    prefixes: tuple[str, ...]
    upstream: str


@dataclass(frozen=True)
class PoolConfig:
    name: str
    type: str
    strategy: str = "first_available"
    remote_dns: bool = True
    proxies: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuleConfig:
    pool: str
    match_prefix: tuple[str, ...] = ()
    match_regex: str | None = None
    description: str = ""


@dataclass(frozen=True, eq=False)
class BridgeConfig:
    """Неизменяемый снимок конфигурации; активный запрос держит ссылку на «свой» снимок."""

    server: ServerConfig
    upstreams: Mapping[str, UpstreamConfig]
    adapters: Mapping[str, AdapterConfig]
    pools: Mapping[str, PoolConfig]
    rules: tuple[RuleConfig, ...]
    default_pool: str = "direct"
    geo_fallback_pool: str = ""


DEFAULT_UPSTREAMS = {
    "opencode-ai": UpstreamConfig("opencode.ai", 443, True),
    "cordis-ai": UpstreamConfig("opencode.ai", 443, True),
    "openrouter-ai": UpstreamConfig("openrouter.ai", 443, True, "/api"),
}
DEFAULT_ADAPTERS = {
    "opencode": AdapterConfig(("/v1", "/go/v1"), "opencode-ai"),
    "cordis": AdapterConfig(("/zen/v1", "/zen/go/v1"), "cordis-ai"),
    "generic": AdapterConfig(("/messages", "/chat/completions"), "openrouter-ai"),
}
_SERVER_DEFAULTS = ServerConfig(listen=LOOPBACK, port=10830)


# ───────────────────────────── валидаторы ─────────────────────────────


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path}: expected object, got {type(value).__name__}")
    return value


def _integer(section: Mapping[str, Any], key: str, default: int, path: str, minimum: int = 0, maximum=None) -> int:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path}.{key}: expected integer, got {type(value).__name__}")
    if value < minimum or (maximum is not None and value > maximum):
        raise ConfigError(f"{path}.{key}: {value} is out of range")
    return value


def _positive_number(section: Mapping[str, Any], key: str, default: float, path: str) -> float:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{path}.{key}: expected number, got {type(value).__name__}")
    if value <= 0:
        raise ConfigError(f"{path}.{key}: must be positive")
    return float(value)


def _string(section: Mapping[str, Any], key: str, default: str, path: str) -> str:
    value = section.get(key, default)
    if not isinstance(value, str):
        raise ConfigError(f"{path}.{key}: expected string, got {type(value).__name__}")
    return value


def _boolean(section: Mapping[str, Any], key: str, default: bool, path: str) -> bool:
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{path}.{key}: expected boolean, got {type(value).__name__}")
    return value


def _string_list(section: Mapping[str, Any], key: str, default: tuple[str, ...], path: str) -> tuple[str, ...]:
    value = section.get(key, list(default))
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ConfigError(f"{path}.{key}: expected list of non-empty strings")
    return tuple(value)


def _parse_server(section: Mapping[str, Any]) -> ServerConfig:
    path, d = "server", _SERVER_DEFAULTS
    for required in ("listen", "port"):
        if required not in section:
            raise ConfigError(f"{path}: missing required key {required!r}")
    listen = _string(section, "listen", d.listen, path)
    if listen != LOOPBACK:
        raise ConfigError(f"{path}.listen: must be {LOOPBACK!r} (loopback only)")
    signatures = _string_list(section, "geo_error_signatures", d.geo_error_signatures, path)
    return ServerConfig(
        listen=listen,
        port=_integer(section, "port", d.port, path, 0, 65535),
        direct_slots=_integer(section, "direct_slots", d.direct_slots, path, 1),
        proxy_slots=_integer(section, "proxy_slots", d.proxy_slots, path, 1),
        max_connections=_integer(section, "max_connections", d.max_connections, path, 1),
        ingress_wait_timeout=_positive_number(section, "ingress_wait_timeout", d.ingress_wait_timeout, path),
        connect_timeout=_positive_number(section, "connect_timeout", d.connect_timeout, path),
        total_egress_deadline=_positive_number(section, "total_egress_deadline", d.total_egress_deadline, path),
        headers_timeout=_positive_number(section, "headers_timeout", d.headers_timeout, path),
        inactivity_timeout=_positive_number(section, "inactivity_timeout", d.inactivity_timeout, path),
        body_timeout=_positive_number(section, "body_timeout", d.body_timeout, path),
        body_buffer_max_bytes=_integer(section, "body_buffer_max_bytes", d.body_buffer_max_bytes, path, 1),
        pre_send_retries=_integer(section, "pre_send_retries", d.pre_send_retries, path, 0),
        geo_cache_ttl_seconds=_integer(section, "geo_cache_ttl_seconds", d.geo_cache_ttl_seconds, path, 1),
        proxy_fail_penalty_seconds=_integer(
            section, "proxy_fail_penalty_seconds", d.proxy_fail_penalty_seconds, path, 0
        ),
        log_dir=_string(section, "log_dir", default_log_dir(), path),
        geo_error_signatures=signatures,
        geo_cache_file=_string(section, "geo_cache_file", d.geo_cache_file, path),
    )


def _parse_upstreams(raw: Any) -> dict[str, UpstreamConfig]:
    upstreams = dict(DEFAULT_UPSTREAMS)
    for name, value in _mapping(raw, "upstreams").items():
        path = f"upstreams.{name}"
        section = _mapping(value, path)
        if not isinstance(section.get("host"), str) or not section["host"]:
            raise ConfigError(f"{path}: missing required key 'host'")
        base_path = _string(section, "base_path", "", path)
        if base_path and not base_path.startswith("/"):
            raise ConfigError(f"{path}.base_path: must start with '/'")
        upstreams[name] = UpstreamConfig(
            host=section["host"],
            port=_integer(section, "port", 443, path, 1, 65535),
            use_tls=_boolean(section, "use_tls", True, path),
            base_path=base_path.rstrip("/"),
        )
    return upstreams


def _parse_adapters(raw: Any, upstreams: Mapping[str, UpstreamConfig]) -> dict[str, AdapterConfig]:
    adapters = dict(DEFAULT_ADAPTERS)
    for name, value in _mapping(raw, "adapters").items():
        path = f"adapters.{name}"
        if name not in DEFAULT_ADAPTERS:
            raise ConfigError(f"{path}: unknown adapter (expected one of {sorted(DEFAULT_ADAPTERS)})")
        section = _mapping(value, path)
        default = DEFAULT_ADAPTERS[name]
        prefixes = _string_list(section, "prefixes", default.prefixes, path)
        if any(not p.startswith("/") or (len(p) > 1 and p.endswith("/")) for p in prefixes):
            raise ConfigError(f"{path}.prefixes: each prefix must start with '/' and not end with '/'")
        adapters[name] = AdapterConfig(prefixes, _string(section, "upstream", default.upstream, path))
    for name, adapter in adapters.items():
        if adapter.upstream not in upstreams:
            raise ConfigError(f"adapters.{name}.upstream: unknown upstream {adapter.upstream!r}")
    return adapters


def _validate_proxy(url: Any, pool_type: str, path: str) -> str:
    if not isinstance(url, str) or not url:
        raise ConfigError(f"{path}: proxy URL must be a non-empty string")
    try:
        endpoint = parse_proxy_url(url)
    except ValueError as exc:
        raise ConfigError(f"{path}: invalid proxy URL ({exc})") from None
    if endpoint.scheme not in PROXY_SCHEMES[pool_type]:
        raise ConfigError(f"{path}: scheme {endpoint.scheme!r} does not match pool type {pool_type!r}")
    return url


def _parse_pools(raw: Any) -> dict[str, PoolConfig]:
    pools: dict[str, PoolConfig] = {}
    for name, value in _mapping(raw, "pools").items():
        path = f"pools.{name}"
        section = _mapping(value, path)
        if "type" not in section:
            raise ConfigError(f"{path}: missing required key 'type'")
        pool_type = section["type"]
        if pool_type not in POOL_TYPES:
            raise ConfigError(f"{path}.type: expected one of {list(POOL_TYPES)}, got {pool_type!r}")
        strategy = section.get("strategy", "first_available")
        if strategy not in STRATEGIES:
            raise ConfigError(f"{path}.strategy: expected one of {list(STRATEGIES)}, got {strategy!r}")
        proxies: tuple[str, ...] = ()
        if pool_type != "direct":
            listed = section.get("proxies")
            if not isinstance(listed, list) or not listed:
                raise ConfigError(f"{path}.proxies: non-empty list required for pool type {pool_type!r}")
            proxies = tuple(_validate_proxy(url, pool_type, f"{path}.proxies[{i}]") for i, url in enumerate(listed))
        pools[name] = PoolConfig(name, pool_type, strategy, _boolean(section, "remote_dns", True, path), proxies)
    pools.setdefault("direct", PoolConfig("direct", "direct"))
    return pools


def _parse_rules(raw: Any, pools: Mapping[str, PoolConfig]) -> tuple[RuleConfig, ...]:
    if not isinstance(raw, list):
        raise ConfigError(f"rules: expected array, got {type(raw).__name__}")
    rules = []
    for index, value in enumerate(raw):
        path = f"rules[{index}]"
        section = _mapping(value, path)
        if "pool" not in section:
            raise ConfigError(f"{path}: missing required key 'pool'")
        pool = _string(section, "pool", "", path)
        if pool not in pools:
            raise ConfigError(f"{path}.pool: unknown pool {pool!r}")
        prefixes = _string_list(section, "match_prefix", (), path)
        regex = section.get("match_regex")
        if regex is not None:
            if not isinstance(regex, str) or not regex:
                raise ConfigError(f"{path}.match_regex: expected non-empty string")
            try:
                re.compile(regex)
            except re.error as exc:
                raise ConfigError(f"{path}.match_regex: invalid regular expression ({exc})") from None
        if not prefixes and regex is None:
            raise ConfigError(f"{path}: needs 'match_prefix' or 'match_regex'")
        rules.append(RuleConfig(pool, prefixes, regex, _string(section, "description", "", path)))
    return tuple(rules)


def _alias(section: dict[str, Any], old: str, new: str) -> None:
    if old in section and new not in section:  # канонический ключ главнее алиаса
        section[new] = section[old]


def _normalize(data: Any) -> Any:
    """Привести устаревшие алиасы (`host`, `urls`, `prefix`) и вложенные секции `slots`/`routing` к плоской схеме."""
    if not isinstance(data, Mapping):
        return data
    root = dict(data)
    routing = root.pop("routing", None)
    if isinstance(routing, Mapping):
        for key, value in routing.items():
            root.setdefault(key, value)
    slots = root.pop("slots", None)
    if isinstance(root.get("server"), Mapping):
        server = root["server"] = dict(root["server"])
        if isinstance(slots, Mapping):
            for key, value in slots.items():
                server.setdefault(key, value)
        _alias(server, "host", "listen")
    if isinstance(root.get("pools"), Mapping):
        pools = root["pools"] = dict(root["pools"])
        for name, pool in pools.items():
            if isinstance(pool, Mapping):
                pools[name] = pool = dict(pool)
                _alias(pool, "urls", "proxies")
    if isinstance(root.get("rules"), list):
        rules = root["rules"] = list(root["rules"])
        for index, rule in enumerate(rules):
            if isinstance(rule, Mapping):
                rules[index] = rule = dict(rule)
                if isinstance(rule.get("prefix"), str):
                    rule["prefix"] = [rule["prefix"]]
                _alias(rule, "prefix", "match_prefix")
    return root


def parse_config(data: Any) -> BridgeConfig:
    """Провалидировать словарь конфигурации и собрать неизменяемый снимок."""
    root = _mapping(_normalize(data), "config")
    for required in ("server", "pools", "rules"):
        if required not in root:
            raise ConfigError(f"config: missing required key {required!r}")
    server = _parse_server(_mapping(root["server"], "server"))
    upstreams = _parse_upstreams(root.get("upstreams", {}))
    adapters = _parse_adapters(root.get("adapters", {}), upstreams)
    pools = _parse_pools(root["pools"])
    rules = _parse_rules(root["rules"], pools)
    default_pool = _string(root, "default_pool", "direct", "config")
    if default_pool not in pools:
        raise ConfigError(f"default_pool: unknown pool {default_pool!r}")
    if "geo_fallback_pool" in root:
        fallback = _string(root, "geo_fallback_pool", "", "config")
        if fallback and fallback not in pools:
            raise ConfigError(f"geo_fallback_pool: unknown pool {fallback!r}")
    else:
        fallback = "route-de" if "route-de" in pools else ""
    return BridgeConfig(
        server=server,
        upstreams=MappingProxyType(upstreams),
        adapters=MappingProxyType(adapters),
        pools=MappingProxyType(pools),
        rules=rules,
        default_pool=default_pool,
        geo_fallback_pool=fallback,
    )


def load_config_file(path: str | Path) -> BridgeConfig:
    try:
        data = json.loads(Path(path).expanduser().read_bytes())
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON: {exc}") from None
    except UnicodeDecodeError as exc:
        raise ConfigError(f"config is not valid UTF-8: {exc}") from None
    return parse_config(data)


# ───────────────────────────── hot-reload ─────────────────────────────


class ConfigManager:
    """Хранит текущий снимок; перечитывает файл при смене `os.stat()` и не теряет рабочий снимок при ошибке."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        initial: BridgeConfig | None = None,
        poll_interval: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        if path is None and initial is None:
            raise ValueError("either path or initial config is required")
        self._path = Path(path).expanduser() if path is not None else None
        self._poll_interval = poll_interval
        self._clock = clock
        self._lock = threading.Lock()
        self._last_check = clock()
        self._last_error: str | None = None
        self._signature: tuple[int, int] | None = None
        if initial is not None:
            self._snapshot = initial
        else:
            self._signature = self._stat()
            self._snapshot = load_config_file(self._path)  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, data: Any) -> ConfigManager:
        return cls(initial=parse_config(data))

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def _stat(self) -> tuple[int, int]:
        info = self._path.stat()  # type: ignore[union-attr]
        return (info.st_mtime_ns, info.st_size)

    def get(self) -> BridgeConfig:
        """Текущий снимок; не чаще `poll_interval` проверяет файл на изменения."""
        if self._path is not None and self._clock() - self._last_check >= self._poll_interval:
            self._last_check = self._clock()
            self.reload_if_changed()
        return self._snapshot

    def reload_if_changed(self) -> bool:
        """Перечитать файл, если изменился mtime/размер; невалидный файл — ошибка в лог, снимок прежний."""
        if self._path is None:
            return False
        with self._lock:
            try:
                signature = self._stat()
            except OSError as exc:
                self._last_error = f"config file is unavailable: {type(exc).__name__}"
                LOGGER.error("config reload skipped: %s", self._last_error)
                return False
            if signature == self._signature:
                return False
            self._signature = signature
            try:
                snapshot = load_config_file(self._path)
            except (ConfigError, OSError) as exc:
                self._last_error = str(exc)
                LOGGER.error("config reload rejected, keeping previous snapshot: %s", exc)
                return False
            self._snapshot = snapshot
            self._last_error = None
            LOGGER.info("config reloaded from %s", self._path)
            return True
