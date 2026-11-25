"""FSM здоровья прокси: экспоненциальный backoff, keep-last-alive (>=2) и HALF_OPEN."""

from __future__ import annotations

import os
import random
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .proxy_pool import EgressSettings, ProxyEndpoint

FLAG_PROXY_FSM_V2 = "GEOROUTER_PROXY_FSM_V2"


class ProxyNodeState(Enum):
    """Состояния узла прокси в пуле."""

    HEALTHY = "healthy"
    PENALIZED = "penalized"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class ProxyHealthPolicy:
    """Opt-in политика здоровья прокси; сохраняет 1-proxy fast-fail."""

    enabled: bool = True
    min_pool_for_keep_alive: int = 2
    max_cooldown_s: float = 120.0
    setup_cap_s: float = 2.5
    max_attempts: int = 2
    jitter_low: float = 0.9
    jitter_high: float = 1.1

    @classmethod
    def from_env(cls) -> ProxyHealthPolicy:
        """Создаёт политику с учётом kill-switch GEOROUTER_PROXY_FSM_V2 (default 1)."""
        raw = os.environ.get(FLAG_PROXY_FSM_V2, "1").strip().lower()
        return cls(enabled=raw not in {"0", "false", "no", "off"})


@dataclass(slots=True)
class _ProxyRecord:
    state: ProxyNodeState = ProxyNodeState.HEALTHY
    failures: int = 0
    cooldown_until: float = 0.0
    probe_inflight: bool = False


class ProxyHealthFSM:
    """Потокобезопасный учёт отказов, HALF_OPEN probe и anti-livelock пула прокси."""

    def __init__(
        self,
        policy: ProxyHealthPolicy,
        *,
        clock: Callable[[], float] = time.time,
        rng: random.Random | None = None,
    ) -> None:
        self.policy = policy
        self._clock = clock
        self._rng = rng or random.Random()
        self._lock = threading.Lock()
        self._records: dict[str, _ProxyRecord] = {}

    def record_failure(self, raw_url: str, base_seconds: float, *, pool_size: int = 2) -> float:
        """Фиксирует отказ и возвращает cooldown_until с exponential backoff + jitter."""
        now = self._clock()
        with self._lock:
            rec = self._records.setdefault(raw_url, _ProxyRecord())
            rec.failures += 1
            rec.probe_inflight = False
            rec.state = ProxyNodeState.PENALIZED
            if not self.policy.enabled or pool_size < self.policy.min_pool_for_keep_alive:
                rec.cooldown_until = now + max(0.0, float(base_seconds))
                return rec.cooldown_until
            exp = min(self.policy.max_cooldown_s, max(0.0, float(base_seconds)) * (2 ** min(6, rec.failures - 1)))
            jitter = self._rng.uniform(self.policy.jitter_low, self.policy.jitter_high)
            rec.cooldown_until = now + (exp * jitter)
            return rec.cooldown_until

    def record_success(self, raw_url: str) -> None:
        """Сбрасывает счётчик отказов и возвращает узел в HEALTHY."""
        with self._lock:
            rec = self._records.get(raw_url)
            if rec is not None:
                rec.state = ProxyNodeState.HEALTHY
                rec.failures = 0
                rec.cooldown_until = 0.0
                rec.probe_inflight = False

    def penalty_until(self, raw_url: str) -> float:
        """Возвращает timestamp окончания штрафа (или 0.0)."""
        with self._lock:
            rec = self._records.get(raw_url)
            return rec.cooldown_until if rec is not None else 0.0

    def is_penalized(self, raw_url: str) -> bool:
        """True если узел сейчас в активном штрафе."""
        return self.penalty_until(raw_url) > self._clock()

    def _pick_keep_alive_locked(self, endpoints: Sequence[ProxyEndpoint], exclude: set[str]) -> list[ProxyEndpoint]:
        best: ProxyEndpoint | None = None
        best_until = float("inf")
        for ep in endpoints:
            if ep.raw in exclude:
                continue
            rec = self._records.setdefault(ep.raw, _ProxyRecord())
            if not rec.probe_inflight and rec.cooldown_until < best_until:
                best, best_until = ep, rec.cooldown_until
        if best is None:
            return []
        best_rec = self._records[best.raw]
        best_rec.state, best_rec.probe_inflight = ProxyNodeState.HALF_OPEN, True
        return [best]

    def available_endpoints(
        self, endpoints: Sequence[ProxyEndpoint], exclude: set[str], *, allow_keep_alive: bool = False
    ) -> tuple[list[ProxyEndpoint], bool]:
        """Возвращает здоровых кандидатов либо один HALF_OPEN probe при >=2 прокси."""
        now = self._clock()
        with self._lock:
            ready = [
                (idx, ep, self._records.get(ep.raw, _ProxyRecord()).failures)
                for idx, ep in enumerate(endpoints)
                if ep.raw not in exclude and self._records.get(ep.raw, _ProxyRecord()).cooldown_until <= now
            ]
            if ready:
                if self.policy.enabled and len(endpoints) >= self.policy.min_pool_for_keep_alive:
                    ready.sort(key=lambda item: (item[2] > 0, item[2], item[0]))
                return [ep for _, ep, _ in ready], False
            if not allow_keep_alive or not self.policy.enabled or len(endpoints) < self.policy.min_pool_for_keep_alive:
                return [], False
            picked = self._pick_keep_alive_locked(endpoints, exclude)
            return (picked, bool(picked))

    def release_probe(self, raw_url: str, success: bool) -> None:
        """Освобождает флаг единственного concurrent HALF_OPEN probe."""
        if success:
            self.record_success(raw_url)
            return
        with self._lock:
            rec = self._records.get(raw_url)
            if rec is not None:
                rec.probe_inflight = False
                if rec.state is ProxyNodeState.HALF_OPEN:
                    rec.state = ProxyNodeState.PENALIZED

    def bounded_timeouts(self, settings: EgressSettings, pool_size: int) -> tuple[float, int]:
        """Ограничивает setup-таймаут попытки и число попыток на операцию (<=2)."""
        if not self.policy.enabled or pool_size < self.policy.min_pool_for_keep_alive:
            return settings.connect_timeout, settings.retries
        cap_timeout = min(settings.connect_timeout, self.policy.setup_cap_s)
        cap_retries = min(settings.retries, max(0, self.policy.max_attempts - 1))
        return cap_timeout, cap_retries
