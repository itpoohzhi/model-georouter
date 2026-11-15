"""Атомарная запись состояния активного upstream (`router-upstream/v1`) с троттлингом."""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import final

from .egress_hedge import ROUTE_DE_POOL, HedgeOutcome, HedgeRecorder

# fmt: off
__all__ = ["DEFAULT_THROTTLE_S", "SCHEMA_VERSION", "HedgeRecorder", "UpstreamSnapshot", "UpstreamStateRecorder", "UpstreamStateTracker", "UpstreamStateWriter", "atomic_write_json", "sanitize_upstream", "strip_credentials"]

SCHEMA_VERSION, DEFAULT_THROTTLE_S = "router-upstream/v1", 1.0

def _utc_now_iso() -> str:
    """Возвращает текущее время UTC в формате ISO-8601 с суффиксом Z."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")

def strip_credentials(value: str) -> str:
    """Удаляет схему, учётные данные (`user:pass@`) и путь/query/fragment из адреса upstream."""
    if not (cleaned := value.strip()):
        return ""
    if "://" in cleaned:
        cleaned = cleaned.split("://", 1)[1]
    if "@" in cleaned:
        cleaned = cleaned.rpartition("@")[2]
    return cleaned.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]

sanitize_upstream = strip_credentials

@dataclass(frozen=True, slots=True)
class UpstreamSnapshot:
    """Снимок состояния активного upstream по схеме `router-upstream/v1`."""
    pool: str
    active_upstream: str
    failover_count: int
    last_failover_at: str | None
    generated_at: str
    schema: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, str | int | None]:
        """Сериализует снимок в словарь схемы `router-upstream/v1` с очисткой кредов."""
        return {"schema": SCHEMA_VERSION, "pool": strip_credentials(self.pool), "active_upstream": strip_credentials(self.active_upstream), "failover_count": int(self.failover_count), "last_failover_at": self.last_failover_at, "generated_at": self.generated_at}

def atomic_write_json(path: Path | str, payload: Mapping[str, object]) -> None:
    """Атомарно записывает JSON через временный файл в той же директории с правами `0600`."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(target.parent, 0o700)
    tmp_path = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    data = (json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    try:
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            _ = os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise

@final
class UpstreamStateWriter:
    """Потокобезопасный рекордер состояния `router-upstream/v1` с троттлингом `<= 1` зап./с."""
    def __init__(self, path: Path | str, pool: str = ROUTE_DE_POOL, min_interval_s: float = DEFAULT_THROTTLE_S, clock: Callable[[], float] | None = None) -> None:
        self.path, self.pool, self.min_interval_s = Path(path), strip_credentials(pool), max(0.0, min_interval_s)
        self._clock, self._lock = clock, threading.Lock()
        self._active_upstream, self._failover_count, self._last_failover_at, self._last_write_mono = "", 0, None, None

    def snapshot(self) -> UpstreamSnapshot:
        """Возвращает текущий снимок состояния в памяти."""
        with self._lock:
            return UpstreamSnapshot(self.pool, self._active_upstream, self._failover_count, self._last_failover_at, _utc_now_iso())

    def update(self, active_upstream: str, *, pool: str | None = None, failover: bool | None = None, force: bool = False) -> bool:
        """Обновляет состояние в памяти и сбрасывает на диск с учётом троттлинга `min_interval_s` (или сразу при `force=True`)."""
        clean_up = strip_credentials(active_upstream)
        with self._lock:
            if pool is not None:
                self.pool = strip_credentials(pool)
            switched = bool(clean_up and self._active_upstream != clean_up and (self._active_upstream if failover is None else failover))
            self._active_upstream = clean_up
            if switched:
                self._failover_count, self._last_failover_at = self._failover_count + 1, _utc_now_iso()
            return self._flush_locked(force=force)

    def record(self, outcome: HedgeOutcome) -> None:
        """Реализует `HedgeRecorderProtocol` для интеграции с `egress_hedge`."""
        _ = self.update(outcome.endpoint.label if outcome.endpoint is not None else outcome.winner, pool=outcome.pool, failover=True if outcome.failover else None)

    def flush(self, force: bool = True) -> bool:
        """Сбрасывает накопленное состояние на диск (при `force=True` — без учёта троттлинга)."""
        with self._lock:
            return self._flush_locked(force=force)

    def _flush_locked(self, *, force: bool) -> bool:
        now = self._clock() if self._clock is not None else time.monotonic()
        if not force and self._last_write_mono is not None and (now - self._last_write_mono) < self.min_interval_s:
            return False
        atomic_write_json(self.path, UpstreamSnapshot(self.pool, self._active_upstream, self._failover_count, self._last_failover_at, _utc_now_iso()).to_dict())
        self._last_write_mono = now
        return True

UpstreamStateTracker = UpstreamStateRecorder = UpstreamStateWriter
