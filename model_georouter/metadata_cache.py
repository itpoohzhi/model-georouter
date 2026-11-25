"""Потокобезопасный HMAC-изолированный singleflight-кэш для GET */models (negative_ttl=0)."""

from __future__ import annotations

import hashlib
import hmac
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode

FLAG_METADATA_CACHE = "GEOROUTER_METADATA_CACHE_ENABLED"
_AUTH_SCOPE_HEADERS = frozenset({
    "authorization", "x-api-key", "anthropic-api-key", "x-goog-api-key",
    "api-key", "proxy-authorization", "openai-organization",
    "openai-project", "anthropic-beta", "anthropic-version", "accept",
})
_SENSITIVE_QUERY_KEYS = frozenset({"key", "api_key", "apikey", "token", "access_token"})
_HOP_BY_HOP_DROP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-length", "set-cookie",
})


def cache_enabled() -> bool:
    """True если GEOROUTER_METADATA_CACHE_ENABLED=1 (default)."""
    raw = os.environ.get(FLAG_METADATA_CACHE, "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


@dataclass(frozen=True, slots=True)
class MetadataCacheConfig:
    """Лимиты TTL, числа записей и памяти кэша /models."""

    ttl_s: float = 30.0
    max_entries: int = 256
    max_body_bytes: int = 1024 * 1024
    max_total_bytes: int = 6 * 1024 * 1024
    max_in_flight: int = 64


@dataclass(frozen=True, slots=True)
class MetadataQuery:
    """Параметры GET-запроса для расчёта канонического ключа /models."""

    route_id: str
    upstream_target: str
    method: str
    path: str
    query: str
    headers: Mapping[str, str] | Sequence[tuple[str, str]]


@dataclass(frozen=True, slots=True)
class CachedMetadataResponse:
    status: int
    headers: tuple[tuple[str, str], ...]
    body: bytes
    expires_at: float


@dataclass(slots=True)
class _InFlight:
    event: threading.Event
    result: CachedMetadataResponse | None = None


@dataclass(frozen=True, slots=True)
class CacheLookup:
    kind: str  # "hit" | "leader" | "bypass"
    key: tuple[str, str, str, str, str, str] | None = None
    entry: CachedMetadataResponse | None = None


def _norm_headers(headers: Mapping[str, str] | Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    items = headers.items() if isinstance(headers, Mapping) else headers
    return [(str(k).strip().lower(), str(v).strip()) for k, v in items]


def _has_bypass_header(lowered: list[tuple[str, str]]) -> bool:
    for name, val in lowered:
        if name == "cookie":
            return True
        if name == "cache-control" and ({t.strip().lower() for t in val.split(",")} & {"no-cache", "no-store", "private"}):
            return True
        if name == "pragma" and "no-cache" in val.lower():
            return True
    return False


def _canonical_query(query: str) -> tuple[str, bool]:
    if not query:
        return "", False
    pairs = parse_qsl(query, keep_blank_values=True)
    for key, val in pairs:
        if key.strip().lower() in _SENSITIVE_QUERY_KEYS and val:
            return "", True
    pairs.sort()
    return urlencode(pairs), False


class MetadataCache:
    """In-memory singleflight-кэш каталогов /models со строгой auth-изоляцией."""

    def __init__(self, config: MetadataCacheConfig | None = None, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._cfg = config or MetadataCacheConfig()
        self._clock = clock
        self._secret = os.urandom(32)
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[str, str, str, str, str, str], CachedMetadataResponse] = OrderedDict()
        self._total_bytes = 0
        self._in_flight: dict[tuple[str, str, str, str, str, str], _InFlight] = {}

    def _auth_hmac(self, lowered: list[tuple[str, str]]) -> str:
        scoped = sorted((k, v) for k, v in lowered if k in _AUTH_SCOPE_HEADERS)
        raw = b"__anon__" if not scoped else b"".join(f"{len(k)}:{k}={len(v)}:{v}|".encode("utf-8") for k, v in scoped)
        return hmac.new(self._secret, raw, hashlib.sha256).hexdigest()

    def build_key(self, req: MetadataQuery) -> tuple[str, str, str, str, str, str] | None:
        """Строит канонический ключ кэша или None при правиле обхода."""
        if not cache_enabled() or req.method.upper() != "GET":
            return None
        clean_path = "/" + req.path.strip("/")
        if not clean_path.endswith("/models") and clean_path != "/models":
            return None
        lowered = _norm_headers(req.headers)
        if _has_bypass_header(lowered):
            return None
        cquery, has_secret = _canonical_query(req.query)
        if has_secret:
            return None
        return (req.route_id, req.upstream_target, "GET", clean_path, cquery, self._auth_hmac(lowered))

    def lookup_or_acquire(self, req: MetadataQuery, *, wait_timeout_s: float = 10.0) -> CacheLookup:
        """Возвращает hit, leader (вызывающий идёт в upstream и обязан позвать complete) или bypass."""
        key = self.build_key(req)
        if key is None:
            return CacheLookup("bypass")
        now = self._clock()
        with self._lock:
            hit = self._entries.get(key)
            if hit is not None:
                if hit.expires_at > now:
                    self._entries.move_to_end(key)
                    return CacheLookup("hit", key=key, entry=hit)
                self._remove_locked(key)
            flight = self._in_flight.get(key)
            if flight is None:
                if len(self._in_flight) >= self._cfg.max_in_flight:
                    return CacheLookup("bypass")
                self._in_flight[key] = _InFlight(event=threading.Event())
                return CacheLookup("leader", key=key)
        flight.event.wait(timeout=wait_timeout_s)
        return CacheLookup("hit", key=key, entry=flight.result) if flight.result is not None else CacheLookup("bypass")

    def complete(self, key: tuple[str, str, str, str, str, str] | None, status: int, headers: Sequence[tuple[str, str]], body: bytes) -> None:
        """Завершает singleflight; кэширует строго 200 OK JSON (negative_ttl=0)."""
        if key is None:
            return
        entry = self._build_entry(status, headers, body)
        with self._lock:
            if entry is not None:
                self._store_locked(key, entry)
            flight = self._in_flight.pop(key, None)
            if flight is not None:
                flight.result = entry
                flight.event.set()

    def _build_entry(self, status: int, headers: Sequence[tuple[str, str]], body: bytes) -> CachedMetadataResponse | None:
        if status != 200 or len(body) > self._cfg.max_body_bytes:
            return None
        lowered = [(k.lower(), v) for k, v in headers]
        if "json" not in next((v for k, v in lowered if k == "content-type"), "").lower():
            return None
        if any(k == "set-cookie" for k, _ in lowered):
            return None
        cc = next((v.lower() for k, v in lowered if k == "cache-control"), "")
        if any(tok in cc for tok in ("no-store", "private")):
            return None
        clean_hdrs = tuple((k, v) for k, v in headers if k.lower() not in _HOP_BY_HOP_DROP)
        return CachedMetadataResponse(status=200, headers=clean_hdrs, body=body, expires_at=self._clock() + self._cfg.ttl_s)

    def _remove_locked(self, key: tuple[str, str, str, str, str, str]) -> None:
        old = self._entries.pop(key, None)
        if old is not None:
            self._total_bytes -= len(old.body)

    def _store_locked(self, key: tuple[str, str, str, str, str, str], entry: CachedMetadataResponse) -> None:
        self._remove_locked(key)
        body_len = len(entry.body)
        if body_len > self._cfg.max_total_bytes:
            return
        while self._entries and (len(self._entries) >= self._cfg.max_entries or self._total_bytes + body_len > self._cfg.max_total_bytes):
            _, evicted = self._entries.popitem(last=False)
            self._total_bytes -= len(evicted.body)
        self._entries[key] = entry
        self._total_bytes += body_len
