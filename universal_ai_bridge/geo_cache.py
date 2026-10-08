"""Потокобезопасный TTL-кэш гео-блокировок `(upstream, model)` с опциональной записью на диск."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .logging_utils import get_logger

LOGGER = get_logger("universal_ai_bridge.geo_cache")
Key = tuple[str, str]


class GeoCache:
    def __init__(
        self,
        path: str | Path | None = None,
        ttl_seconds: float = 86400,
        clock: Callable[[], float] = time.time,
    ):
        self._path = Path(path).expanduser() if path else None
        self._ttl = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()  # сериализует запись на диск; _lock под неё не удерживается
        self._version = 0
        self._written = 0
        self._entries: dict[Key, float] = {}
        self._load()

    def is_blocked(self, upstream: str, model: str | None) -> bool:
        if not model:
            return False
        key = (upstream, model)
        with self._lock:
            expires = self._entries.get(key)
            if expires is None:
                return False
            if expires > self._clock():
                return True
            del self._entries[key]
            snapshot = self._snapshot()
        self._persist(snapshot)
        return False

    def mark_blocked(self, upstream: str, model: str | None, ttl_seconds: float | None = None) -> None:
        if not model:
            return
        ttl = self._ttl if ttl_seconds is None else ttl_seconds
        with self._lock:
            self._entries[(upstream, model)] = self._clock() + ttl
            snapshot = self._snapshot()
        self._persist(snapshot)

    def flush(self) -> int:
        """Очистить кэш; вернуть число удалённых записей."""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            snapshot = self._snapshot()
        self._persist(snapshot)
        return count

    def __len__(self) -> int:
        with self._lock:
            now = self._clock()
            return sum(1 for expires in self._entries.values() if expires > now)

    def _snapshot(self) -> tuple[int, list[dict]] | None:
        """Копия записей в памяти; вызывается под `_lock`, диск не трогает."""
        if self._path is None:
            return None
        self._version += 1
        now = self._clock()
        payload = [
            {"upstream": upstream, "model": model, "expires_at": expires}
            for (upstream, model), expires in self._entries.items()
            if expires > now
        ]
        return self._version, payload

    def _persist(self, snapshot: tuple[int, list[dict]] | None) -> None:
        """Атомарная запись (tmp + replace) вне `_lock`; устаревший снимок не затирает более новый."""
        if self._path is None or snapshot is None:
            return
        version, payload = snapshot
        tmp = self._path.with_name(self._path.name + ".tmp")
        with self._io_lock:
            if version <= self._written:
                return
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(json.dumps(payload), encoding="utf-8")
                os.replace(tmp, self._path)
                self._written = version
            except OSError as exc:
                LOGGER.warning("geo cache persist failed: %s", type(exc).__name__)

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            now = self._clock()
            for item in payload:
                if float(item["expires_at"]) > now:
                    self._entries[(str(item["upstream"]), str(item["model"]))] = float(item["expires_at"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            LOGGER.warning("geo cache file ignored: %s", type(exc).__name__)
            self._entries.clear()
