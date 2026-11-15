"""Хеджирование TCP/TLS-рукопожатий по пулу прокси (route-de) в пределах бюджета <= 3 с."""

from __future__ import annotations

import socket
import ssl
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol, cast, final, runtime_checkable

from . import proxy_pool as _pp
from .config import BridgeConfig, PoolConfig, UpstreamConfig
from .errors import AllProxiesPenalizedError, EgressConnectError, EgressTimeoutError
from .proxy_pool import EgressSettings, ProxyEndpoint, ProxyPoolManager
from .relay import close_quietly
from .retry_budget import EgressDeadline, RetryBudgetSpec

# fmt: off
__all__ = ["DEFAULT_HEDGE_DELAY_S", "MAX_HEDGE_BUDGET_S", "ROUTE_DE_POOL", "AttemptFn", "EgressHedgeRunner", "HedgeConfig", "HedgeOutcome", "HedgePolicy", "HedgeRecorder", "HedgeRecorderProtocol", "HedgeReport", "build_egress", "connect_pool_hedged", "hedged_connect"]
ROUTE_DE_POOL, MAX_HEDGE_BUDGET_S, DEFAULT_HEDGE_DELAY_S, _CANCEL_GRACE_S = "route-de", 3.0, 0.25, 0.03
_ATTEMPT_ERRORS = (OSError, ValueError, EgressConnectError)
AttemptFn = Callable[[ProxyEndpoint | None, float], socket.socket]
WrapFn = Callable[[socket.socket], socket.socket]
AttemptHook = Callable[[ProxyEndpoint | None], None]
@dataclass(frozen=True, slots=True)
class HedgePolicy:
    """Политика хеджирования установления соединения и TLS-рукопожатия."""
    enabled: bool = False
    hedge_delay_s: float = DEFAULT_HEDGE_DELAY_S
    budget_s: float = MAX_HEDGE_BUDGET_S
    target_pool: str = ROUTE_DE_POOL
    def __post_init__(self) -> None:
        if self.hedge_delay_s <= 0 or self.budget_s <= 0 or self.budget_s > MAX_HEDGE_BUDGET_S:
            raise ValueError("hedge_delay_s and budget_s must be positive" if self.budget_s <= MAX_HEDGE_BUDGET_S else f"budget_s must not exceed {MAX_HEDGE_BUDGET_S:g}s")
    def clamp_budget(self, requested: float | None = None) -> float:
        return min(self.budget_s, MAX_HEDGE_BUDGET_S) if requested is None else min(self.budget_s, MAX_HEDGE_BUDGET_S, max(0.001, requested))
    def should_hedge(self, pool_name: str, candidate_count: int) -> bool:
        return self.enabled and pool_name == self.target_pool and candidate_count > 1
@dataclass(frozen=True, slots=True)
class HedgeOutcome:
    """Отчёт о результате попытки установления соединения (winner / failover)."""
    pool: str
    winner: str
    endpoint: ProxyEndpoint | None
    attempts: int
    hedged: bool
    failover: bool
    loser_closed: bool = False
HedgeConfig, HedgeReport = HedgePolicy, HedgeOutcome
@runtime_checkable
class HedgeRecorderProtocol(Protocol):
    def record(self, outcome: HedgeOutcome) -> None: ...
class _ServerProto(Protocol):
    retry_budget: RetryBudgetSpec
    ssl_context: ssl.SSLContext
class _HandlerProto(Protocol):
    server: _ServerProto
    _ctx: tuple[object, BridgeConfig, *tuple[object, ...]]
    _last_proxy_endpoint: ProxyEndpoint | None
    proxy_endpoint: str
@final
@dataclass
class HedgeRecorder:
    """Потокобезопасный DI-рекордер результатов хеджирования."""
    outcomes: list[HedgeOutcome] = field(default_factory=list)
    winners: list[str] = field(default_factory=list)
    failovers: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    def record(self, outcome: HedgeOutcome) -> None:
        with self._lock:
            _ = (self.outcomes.append(outcome), self.winners.append(outcome.winner), self.failovers.extend([outcome.winner] if outcome.failover else []))
    def record_failure(self, pool: str, error: Exception) -> None:
        with self._lock:
            self.failures.append(f"{pool}:{type(error).__name__}")
def _notify(recorder: object, outcome: HedgeOutcome | None = None, pool: str = "", err: Exception | None = None) -> None:
    if outcome is not None:
        _ = recorder.record(outcome) if isinstance(recorder, HedgeRecorderProtocol) else (recorder(outcome) if callable(recorder) else None)
    elif err is not None and callable(fn := getattr(recorder, "record_failure", None)):
        _ = fn(pool, err)
def _report_mgr(manager: object, ep: ProxyEndpoint | None, *, ok: bool, penalty_s: float) -> None:
    if manager is not None and ep is not None:
        if ok and callable(fn := getattr(manager, "mark_ok", None) or getattr(manager, "record_success", None)):
            _ = fn(ep)
        elif not ok and penalty_s > 0 and callable(fn := getattr(manager, "penalize", None)):
            _ = fn(ep, penalty_s)
def _normalize_err(exc: Exception, ep: ProxyEndpoint | None) -> EgressConnectError:
    if isinstance(exc, EgressConnectError):
        return exc
    via = f" via {ep.label}" if ep is not None else ""
    return EgressTimeoutError(f"hedged connect{via} timed out", phase="tls_handshake") if isinstance(exc, TimeoutError) else EgressConnectError(f"hedged connect{via} failed: {type(exc).__name__}", phase="connect")
@final
class _HedgeRace:
    """Координатор конкурентных попыток подключения с отменой проигравших и штрафованием упавших узлов."""
    def __init__(self, pool: str, fn: AttemptFn, deadline: float, manager: object = None, penalty_s: float = 30.0, on_winner: AttemptHook | None = None) -> None:
        self.pool, self.fn, self.deadline, self.manager, self.penalty_s, self.on_winner = pool, fn, deadline, manager, penalty_s, on_winner
        self._cond, self._penalized = threading.Condition(), set[str]()
        self.win_sock, self.win_ep = cast(socket.socket | None, None), cast(ProxyEndpoint | None, None)
        self.win_idx, self.launched, self.active, self.loser_closed, self.errors = -1, 0, 0, False, list[EgressConnectError]()
    def _penalize_once(self, ep: ProxyEndpoint | None) -> None:
        if ep is not None and ep.raw not in self._penalized:
            _ = (self._penalized.add(ep.raw), _report_mgr(self.manager, ep, ok=False, penalty_s=self.penalty_s))
    def _attempt(self, idx: int, ep: ProxyEndpoint | None) -> None:
        try:
            sock, err = self.fn(ep, self.deadline), None
        except _ATTEMPT_ERRORS as exc:
            sock, err = None, _normalize_err(exc, ep)
        with self._cond:
            self.active -= 1
            if err is not None:
                _ = (self.errors.append(err), self._penalize_once(ep))
            elif self.win_sock is None and time.monotonic() <= self.deadline:
                self.win_sock, self.win_ep, self.win_idx, self.loser_closed, sock = sock, ep, idx, self.active > 0 or self.launched > 1, None
            else:
                self.loser_closed = True
            self._cond.notify_all()
        if sock is not None:
            close_quietly(sock)
    def run(self, endpoints: Sequence[ProxyEndpoint | None], delay_s: float, recorder: object) -> socket.socket:
        for i, ep in enumerate(endpoints):
            with self._cond:
                if i > 0 and self.win_sock is None and self.active > 0 and (w := min(delay_s, max(0.0, self.deadline - time.monotonic()))) > 0:
                    _ = self._cond.wait(timeout=w)
                if i > 0 and (self.win_sock is not None or time.monotonic() >= self.deadline):
                    break
                idx, self.launched, self.active = self.launched, self.launched + 1, self.active + 1
            threading.Thread(target=self._attempt, args=(idx, ep), daemon=True).start()
        with self._cond:
            while self.win_sock is None and self.active > 0 and (rem := self.deadline - time.monotonic()) > 0:
                _ = self._cond.wait(timeout=rem)
            if self.win_sock is not None and self.active > 0:
                _ = self._cond.wait(timeout=min(_CANCEL_GRACE_S, max(0.0, self.deadline - time.monotonic())))
            # Живые, но медленные проигравшие не штрафуются (F-001/F-028): иначе 2-узловой пул
            # route-de после первой хедж-победы теряет узел и хеджирование выключается.
            # Штраф без победителя — только узлам, не ответившим до дедлайна (упавшие уже оштрафованы в `_attempt`).
            if self.win_sock is None:
                for ep in endpoints[: self.launched]:
                    self._penalize_once(ep)
        if self.win_sock is not None:
            _ = (_report_mgr(self.manager, self.win_ep, ok=True, penalty_s=self.penalty_s), self.on_winner(self.win_ep) if self.on_winner is not None else None, _notify(recorder, HedgeOutcome(self.pool, self.win_ep.label if self.win_ep else "direct", self.win_ep, self.launched, self.launched > 1, self.win_idx > 0, self.loser_closed)))
            return self.win_sock
        err = self.errors[-1] if (self.errors and self.active == 0) else EgressTimeoutError(f"hedged connect for {self.pool!r} exceeded budget", phase="tls_handshake")
        _notify(recorder, pool=self.pool, err=err)
        raise err
def hedged_connect(pool: PoolConfig, attempt_fn: AttemptFn, policy: HedgePolicy | None = None, recorder: object = None, candidates: Sequence[ProxyEndpoint | None] | None = None, *, manager: object = None, penalty_s: float = 30.0, on_winner: AttemptHook | None = None) -> socket.socket:
    """Выполняет одиночное или хеджированное подключение по узлам пула в пределах бюджета <= 3 с."""
    active_policy = policy or HedgePolicy()
    resolved = list(candidates) if candidates is not None else ([None] if pool.type == "direct" else [_pp.parse_proxy_url(u) for u in pool.proxies])
    if not resolved:
        _notify(recorder, pool=pool.name, err=(err := AllProxiesPenalizedError(pool.name)))
        raise err
    deadline = time.monotonic() + active_policy.clamp_budget()
    if active_policy.should_hedge(pool.name, len(resolved)):
        return _HedgeRace(pool.name, attempt_fn, deadline, manager, penalty_s, on_winner).run(resolved, active_policy.hedge_delay_s, recorder)
    ep = resolved[0]
    try:
        sock = attempt_fn(ep, deadline)
        if time.monotonic() > deadline:
            close_quietly(sock)
            raise TimeoutError("single attempt exceeded deadline")
    except _ATTEMPT_ERRORS as exc:
        _ = (_report_mgr(manager, ep, ok=False, penalty_s=penalty_s), _notify(recorder, pool=pool.name, err=(err := _normalize_err(exc, ep))))
        raise err from exc
    _ = (_report_mgr(manager, ep, ok=True, penalty_s=penalty_s), on_winner(ep) if on_winner is not None else None, _notify(recorder, HedgeOutcome(pool.name, ep.label if ep else "direct", ep, 1, False, False, False)))
    return sock
@final
class EgressHedgeRunner:
    """Адаптер хеджированного подключения поверх `ProxyPoolManager`."""
    def __init__(self, manager: ProxyPoolManager, policy: HedgePolicy | None = None, recorder: object = None) -> None:
        self.manager, self.policy, self.recorder = manager, policy or HedgePolicy(), recorder
    def connect(self, pool: PoolConfig, target: tuple[str, int], settings: EgressSettings, wrap: WrapFn | None = None, on_attempt: AttemptHook | None = None, *, handler: object = None, up: UpstreamConfig | None = None, single: bool = False) -> socket.socket:
        """Устанавливает TCP/TLS-соединение через пул с поддержкой хеджирования для `route-de`."""
        if isinstance(handler, tuple) and up is not None:
            hdl, dl, err_cls = cast(tuple[_HandlerProto, EgressDeadline, type[EgressTimeoutError]], handler)
            srv, cfg, budget = hdl.server, hdl._ctx[1].server, hdl.server.retry_budget  # pyright: ignore[reportPrivateUsage]
            req_connect = min(settings.connect_timeout, budget.connect_timeout) if single else settings.connect_timeout
            try:
                eff = replace(settings, connect_timeout=dl.clamp(req_connect), total_deadline=dl.clamp(cfg.total_egress_deadline), single_pool_max_retries=budget.single_pool_max_retries if single else settings.single_pool_max_retries)
            except TimeoutError:
                raise err_cls(f"total egress deadline of {cfg.total_egress_deadline:g}s exceeded") from None
            upstream, is_target = up, pool.name == self.policy.target_pool
            def wrap_tls(raw: socket.socket) -> socket.socket:
                if is_target:
                    raw.settimeout(self.policy.clamp_budget(eff.connect_timeout))
                try:
                    return srv.ssl_context.wrap_socket(raw, server_hostname=upstream.host)
                except TimeoutError as exc:
                    raise ssl.SSLError("tls handshake timed out") if is_target else exc
            attempts = 0
            def hook(ep: ProxyEndpoint | None) -> None:
                nonlocal attempts
                attempts += 1
                hdl._last_proxy_endpoint = ep  # pyright: ignore[reportPrivateUsage]
                hdl.proxy_endpoint = "-" if ep is None else (f"[{ep.host}]:{ep.port}" if ":" in ep.host else f"{ep.host}:{ep.port}")
            if not is_target or not self.policy.should_hedge(pool.name, len(self.manager.available(pool))):
                if not is_target:
                    return self.manager._connect(pool, upstream.host, upstream.port, eff, wrap_tls if upstream.use_tls else None, hook)  # pyright: ignore[reportPrivateUsage]
                try:
                    sock = self.manager._connect(pool, upstream.host, upstream.port, eff, wrap_tls if upstream.use_tls else None, hook)  # pyright: ignore[reportPrivateUsage]
                except AllProxiesPenalizedError as exc:
                    _notify(self.recorder, pool=pool.name, err=exc)
                    raise
                except _ATTEMPT_ERRORS as exc:
                    _notify(self.recorder, pool=pool.name, err=_normalize_err(exc, hdl._last_proxy_endpoint))  # pyright: ignore[reportPrivateUsage]
                    raise
                win_ep = hdl._last_proxy_endpoint  # pyright: ignore[reportPrivateUsage]
                _notify(self.recorder, HedgeOutcome(pool.name, win_ep.label if win_ep is not None else "direct", win_ep, max(1, attempts), False, attempts > 1, False))
                return sock
            return self.connect(pool, target, eff, wrap_tls if upstream.use_tls else None, hook)
        self.manager.ensure_available(pool)
        candidates: Sequence[ProxyEndpoint | None] = [None] if pool.type == "direct" else self.manager.available(pool)
        host, port = target
        def attempt_one(ep: ProxyEndpoint | None, deadline: float) -> socket.socket:
            _ = on_attempt(ep) if on_attempt is not None else None
            sock = _pp.open_direct(host, port, deadline) if ep is None else (_pp.open_http_connect(ep, host, port, deadline) if pool.type == "http_connect" else _pp.open_socks5(ep, host, port, deadline, pool.remote_dns or ep.scheme == "socks5h"))
            try:
                if wrap is not None:
                    if (rem := deadline - time.monotonic()) <= 0:
                        raise TimeoutError("TLS handshake deadline exceeded")
                    sock.settimeout(rem)
                    sock = wrap(sock)
                sock.settimeout(None)
                return sock
            except _ATTEMPT_ERRORS:
                close_quietly(sock)
                raise
        eff_pol = HedgePolicy(self.policy.enabled, self.policy.hedge_delay_s, self.policy.clamp_budget(settings.connect_timeout), self.policy.target_pool)
        return hedged_connect(pool, attempt_one, eff_pol, self.recorder, candidates, manager=self.manager, penalty_s=settings.penalty_seconds, on_winner=on_attempt)
def build_egress(manager: ProxyPoolManager | None = None, policy: HedgePolicy | bool | None = None, recorder: object = None) -> EgressHedgeRunner:
    return EgressHedgeRunner(manager or ProxyPoolManager(), HedgePolicy(enabled=policy) if isinstance(policy, bool) else policy, recorder)
def connect_pool_hedged(runner: EgressHedgeRunner, pool: PoolConfig, target: tuple[str, int], settings: EgressSettings, wrap: WrapFn | None = None) -> socket.socket:
    return runner.connect(pool, target, settings, wrap)
