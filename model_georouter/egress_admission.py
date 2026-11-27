"""Контроль входа egress-запросов (W1-C): in-flight лимиты по egress_id и streamGenerateContent."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import cast, final

from . import config_guard as _cg
from . import retry_budget as _rb
from .config import BridgeConfig, PoolConfig
from .errors import ConfigError, EgressTimeoutError, SlotsExhaustedError
from .proxy_pool import parse_proxy_url
from .retry_budget import EgressDeadline, RetryBudgetSpec

__all__ = ["DEFAULT_MAX_INFLIGHT_LONG_HEADER", "DEFAULT_MAX_INFLIGHT_PER_BACKEND", "DEFAULT_PROXY_SLOTS", "LONG_HEADER_MARKER", "AdmissionTicket", "EgressAdmission", "InflightLimitError", "Ticket", "is_long_header_path", "is_streaming_request"]  # fmt: skip

DEFAULT_PROXY_SLOTS, DEFAULT_MAX_INFLIGHT_PER_BACKEND, DEFAULT_MAX_INFLIGHT_LONG_HEADER = 80, 40, 24
DEFAULT_ADMISSION_WAIT_S = 0.0
LONG_HEADER_MARKER = _rb.F08_V2_STREAM_MARKER
_STREAMING_BODY_SUFFIXES = ("/chat/completions", "/messages", "/responses")
_F08_UNSET = object()


@final
class InflightLimitError(SlotsExhaustedError):
    """Быстрый отказ 503 при исчерпании in-flight лимита egress или long-header."""

    status_code, outcome = 503, "inflight_limit"

    def __init__(self, kind: str, *, pool: str = "", egress_key: str = "", limit: int = 0) -> None:
        super().__init__(kind)
        self.pool, self.egress_key, self.limit = pool, egress_key, limit


@dataclass(frozen=True, slots=True)
class Ticket:
    """Билет допуска egress-запроса с расчётным дедлайном первой головы ответа и скользящим бюджетом v2."""

    pool: str
    path: str
    egress_key: str
    header_deadline_s: float
    deadline: EgressDeadline
    is_long_header: bool
    backend: str = ""
    t_pre_s: float = 0.0
    sliding_idle_s: float | None = None

    @property
    def egress_id(self) -> str:
        """Псевдоним ключа счётчика egress."""
        return self.egress_key

    def on_chunk(self, *, now: float | None = None) -> Ticket:
        """Сбрасывает скользящий таймаут чтения (`150 s`) при получении очередного чанка потока."""
        if self.sliding_idle_s is not None:
            object.__setattr__(self, "deadline", self.deadline.slide(self.sliding_idle_s, now=now))
            object.__setattr__(self, "header_deadline_s", self.sliding_idle_s)
        return self


AdmissionTicket = Ticket


def is_long_header_path(path: str) -> bool:
    """Определяет запросы с длинным ожиданием заголовков (`streamGenerateContent`)."""
    return LONG_HEADER_MARKER in path


def is_streaming_request(path: str, *, stream_body: bool = False) -> bool:
    """Определяет streaming-запросы: `streamGenerateContent` или `"stream": true` на OpenAI/Anthropic путях."""
    if is_long_header_path(path):
        return True
    return bool(stream_body) and any(path.endswith(suffix) for suffix in _STREAMING_BODY_SUFFIXES)


def _normalize_host_port(raw: str) -> str:
    return f"{parse_proxy_url(t).host}:{parse_proxy_url(t).port}" if "://" in (t := raw.strip()) else t


@final
class _AdmitCtx:
    """Двурежимный (sync + async) контекстный менеджер допуска запроса."""

    def __init__(self, adm: EgressAdmission, pool: str, path: str, backend: str | None, deadline: EgressDeadline | None, started_at: float | None, now: float | None, is_stream: bool = False) -> None:  # fmt: skip
        self._adm, self._args, self._prev, self._ticket = (
            adm,
            (pool, path, backend, deadline, started_at, now, is_stream),
            None,
            cast(Ticket | None, None),
        )

    def __enter__(self) -> Ticket:
        self._prev, self._ticket = self._adm.enter_slot(*self._args)
        return self._ticket

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self._adm.exit_slot(self._ticket, self._prev)

    async def __aenter__(self) -> Ticket:
        return self.__enter__()

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.__exit__(exc_type, exc, tb)


@final
class EgressAdmission:
    """Менеджер квот параллельных запросов на внешний egress и long-header пути."""

    def __init__(self, pools: Mapping[str, object] | None = None, *, config: _cg.GuardedConfig | BridgeConfig | None = None, proxy_slots: int = DEFAULT_PROXY_SLOTS, max_inflight_per_backend: int = DEFAULT_MAX_INFLIGHT_PER_BACKEND, max_inflight_long_header: int = DEFAULT_MAX_INFLIGHT_LONG_HEADER, f08_policy: object = _F08_UNSET, total_egress_deadline: float = _cg.F08_V1_TOTAL_EGRESS_DEADLINE, retry_budget: RetryBudgetSpec | None = None, client_timeout_s: object = _F08_UNSET, f08_v2_risk_ack: object = _F08_UNSET, t_pre: float = _rb.F08_V2_T_PRE, t_idle: float = _rb.F08_V2_T_IDLE, f08_v2_scope: str = _rb.F08_V2_STREAM_MARKER, admission_wait_s: object = _F08_UNSET) -> None:  # fmt: skip
        source = config.server if isinstance(config, BridgeConfig) else config
        if source is not None:
            if f08_policy is _F08_UNSET:
                f08_policy = _cg.DEFAULT_F08_POLICY if source.f08_policy is None else source.f08_policy
            if f08_v2_risk_ack is _F08_UNSET:
                f08_v2_risk_ack = source.f08_v2_risk_ack
            if client_timeout_s is _F08_UNSET:
                client_timeout_s = source.client_timeout_s
        if f08_policy is _F08_UNSET:
            f08_policy = _cg.DEFAULT_F08_POLICY
        if f08_v2_risk_ack is _F08_UNSET:
            f08_v2_risk_ack = None
        if client_timeout_s is _F08_UNSET:
            client_timeout_s = None
        if f08_v2_risk_ack is not None and not isinstance(f08_v2_risk_ack, str):
            raise ConfigError("server.f08_v2_risk_ack: expected string or None")
        if client_timeout_s is not None and (
            isinstance(client_timeout_s, bool)
            or not isinstance(client_timeout_s, int | float)
            or not math.isfinite(client_timeout_s)
            or client_timeout_s <= 0
        ):
            raise ConfigError("server.client_timeout_s: expected finite positive number or None")
        if isinstance(config, _cg.GuardedConfig):
            pools, proxy_slots, total_egress_deadline = (
                config.bridge.pools if pools is None else pools,
                max(2, config.bridge.server.proxy_slots),
                config.bridge.server.total_egress_deadline,
            )
            per_b, long_h = config.max_inflight_per_backend, config.max_inflight_long_header
        else:
            if isinstance(config, BridgeConfig):
                pools, proxy_slots, total_egress_deadline = (
                    config.pools if pools is None else pools,
                    config.server.proxy_slots,
                    config.server.total_egress_deadline,
                )
            per_b, long_h = _cg.validate_inflight_limits(
                proxy_slots, max_inflight_per_backend, max_inflight_long_header
            )
        spec, v2_budget = (
            retry_budget
            or RetryBudgetSpec(
                total_egress_deadline=min(
                    _cg.F08_V1_TOTAL_EGRESS_DEADLINE
                    if float(total_egress_deadline) == 20.0
                    else float(total_egress_deadline),
                    _cg.F08_V1_TOTAL_EGRESS_DEADLINE,
                )
            ),
            cast(_rb.F08V2Budget | None, None),
        )
        if f08_policy == _cg.DEFAULT_F08_POLICY and not isinstance(config, _cg.GuardedConfig):
            if (
                _cg.validate_f08_deadline(total_egress_deadline, f08_policy)
                and spec.total_egress_deadline != _cg.F08_V1_TOTAL_EGRESS_DEADLINE
            ):
                raise ConfigError(
                    f"retry_budget.total_egress_deadline: must be 40.0 for v1, got {spec.total_egress_deadline!r}"
                )
            try:
                spec.validate_f08()
            except ValueError as exc:
                raise ConfigError(f"retry_budget: {exc}") from None
        elif f08_policy == _rb.F08_V2_POLICY:
            v2_budget = _rb.validate_f08_v2(
                t_pre,
                t_idle,
                client_timeout_s,
                f08_v2_risk_ack,
                f08_v2_scope,
                non_stream_deadline=float(total_egress_deadline),
            )
        elif f08_policy != _cg.DEFAULT_F08_POLICY:
            raise ConfigError(f"server.f08_policy: unsupported policy {f08_policy!r}, expected 'v1' or 'v2'")
        self._pools, self._route_de_egress_id = (
            dict(pools) if pools is not None else {},
            config.route_de_egress_id if isinstance(config, _cg.GuardedConfig) else "",
        )
        self.max_inflight_per_backend, self.max_inflight_long_header, self.f08_policy, self.retry_budget, self.f08_v2_budget = per_b, long_h, str(f08_policy), spec, v2_budget  # fmt: skip
        self._inflight_by_egress, self._inflight_long_header, self._lock, self._local = (
            cast(dict[str, int], {}),
            0,
            threading.Lock(),
            threading.local(),
        )
        if admission_wait_s is _F08_UNSET:
            cfg_server = (
                config.bridge.server
                if isinstance(config, _cg.GuardedConfig)
                else (config.server if isinstance(config, BridgeConfig) else None)
            )
            admission_wait_s = getattr(cfg_server, "admission_wait_s", DEFAULT_ADMISSION_WAIT_S)
        if isinstance(admission_wait_s, bool) or not isinstance(admission_wait_s, int | float) or not math.isfinite(admission_wait_s) or admission_wait_s < 0:  # fmt: skip
            raise ConfigError("admission_wait_s: expected finite non-negative number")
        self.wait_timeout, self._cond = float(admission_wait_s), threading.Condition(self._lock)
        self.admission_wait_s = self.wait_timeout

    def resolve_egress_key(self, pool: str, backend: str | None = None) -> str:
        """Возвращает `pools[pool].egress_id`, либо fallback на `host:port` при отсутствии `egress_id`."""
        cfg = self._pools.get(pool)
        eg_id = (
            cast(Mapping[str, object], cfg).get("egress_id")
            if isinstance(cfg, Mapping)
            else getattr(cfg, "egress_id", None)
        )
        if (not isinstance(eg_id, str) or not eg_id.strip()) and pool == _cg.ROUTE_DE_POOL and self._route_de_egress_id:
            eg_id = self._route_de_egress_id
        if isinstance(eg_id, str) and eg_id.strip():
            return eg_id.strip()
        if backend and backend.strip():
            return _normalize_host_port(backend)
        if isinstance(cfg, PoolConfig) and cfg.proxies:
            return _normalize_host_port(cfg.proxies[0])
        if (
            isinstance(cfg, Mapping)
            and isinstance(raw_p := cast(Mapping[str, object], cfg).get("proxies"), list | tuple)
            and raw_p
            and isinstance(raw_p[0], str)
        ):
            return _normalize_host_port(raw_p[0])
        return _normalize_host_port(pool)

    def inflight_for(self, egress_key: str) -> int:
        """Текущее число активных запросов для заданного ключа egress."""
        with self._lock:
            return self._inflight_by_egress.get(egress_key, 0)

    @property
    def inflight_long_header(self) -> int:
        """Текущее число активных `streamGenerateContent` запросов."""
        with self._lock:
            return self._inflight_long_header

    def current(self, deadline: EgressDeadline | None = None, path: str = "", *, is_stream: bool = False) -> Ticket:
        """Возвращает активный билет текущего потока или расчётный билет по `deadline`."""
        if isinstance(t := getattr(self._local, "ticket", None), Ticket):
            return t
        long_hdr = is_streaming_request(path, stream_body=is_stream and self.f08_v2_budget is not None)
        dl, t_pre, s_idle = self.f08_v2_budget.resolve_for_path(path, deadline=deadline, is_stream=long_hdr) if self.f08_v2_budget else (deadline or EgressDeadline.from_timeout(self.retry_budget.total_egress_deadline), self.retry_budget.worst_case_pre_send_seconds, None)  # fmt: skip
        return Ticket("", path, "", max(0.0, dl.remaining()), dl, long_hdr, "", t_pre, s_idle)

    def enter_slot(self, pool: str, path: str, backend: str | None, deadline: EgressDeadline | None, started_at: float | None, now: float | None, is_stream: bool = False) -> tuple[object, Ticket]:  # fmt: skip
        """Занимает слот в счётчиках in-flight и сохраняет активный билет в TLS потока."""
        egress_key, long_hdr = (
            self.resolve_egress_key(pool, backend=backend),
            is_streaming_request(path, stream_body=is_stream and self.f08_v2_budget is not None),
        )
        eg_deadline, t_pre, s_idle = self.f08_v2_budget.resolve_for_path(path, deadline=deadline, started_at=started_at, is_stream=long_hdr) if self.f08_v2_budget else (deadline or EgressDeadline.from_timeout(self.retry_budget.total_egress_deadline, started_at=started_at), self.retry_budget.worst_case_pre_send_seconds, None)  # fmt: skip
        if (remaining_s := eg_deadline.remaining(now=now)) <= 0.0:
            raise EgressTimeoutError("egress deadline expired before admission", phase="admission")
        with self._cond:
            end_wait = time.monotonic() + min(self.wait_timeout, remaining_s)
            while True:
                cur_egress = self._inflight_by_egress.get(egress_key, 0)
                backend_full = cur_egress >= self.max_inflight_per_backend
                long_full = long_hdr and self._inflight_long_header >= self.max_inflight_long_header
                if not backend_full and not long_full:
                    break
                if (left := end_wait - time.monotonic()) <= 0:
                    if backend_full:
                        raise InflightLimitError(
                            "backend", pool=pool, egress_key=egress_key, limit=self.max_inflight_per_backend
                        )
                    raise InflightLimitError(
                        "long_header", pool=pool, egress_key=egress_key, limit=self.max_inflight_long_header
                    )
                _ = self._cond.wait(timeout=left)
            self._inflight_by_egress[egress_key] = cur_egress + 1
            self._inflight_long_header += 1 if long_hdr else 0
        prev, ticket = getattr(self._local, "ticket", None), Ticket(pool, path, egress_key, remaining_s, eg_deadline, long_hdr, backend or "", t_pre, s_idle)  # fmt: skip
        self._local.ticket = ticket
        return prev, ticket

    def exit_slot(self, ticket: Ticket | None, prev: object) -> None:
        """Освобождает слот в счётчиках in-flight и восстанавливает предыдущий билет."""
        self._local.ticket = prev
        if ticket is not None:
            with self._cond:
                left = self._inflight_by_egress.get(ticket.egress_key, 1) - 1
                _ = (
                    self._inflight_by_egress.pop(ticket.egress_key, None)
                    if left <= 0
                    else self._inflight_by_egress.__setitem__(ticket.egress_key, left)
                )
                self._inflight_long_header = max(0, self._inflight_long_header - (1 if ticket.is_long_header else 0))
                self._cond.notify_all()

    def admit(self, pool: str, path: str = "", *, backend: str | None = None, deadline: EgressDeadline | None = None, started_at: float | None = None, now: float | None = None, is_stream: bool = False) -> _AdmitCtx:  # fmt: skip
        """Немедленно выделяет слот или выбрасывает `InflightLimitError` (503, `inflight_limit`)."""
        return _AdmitCtx(self, pool, path, backend, deadline, started_at, now, is_stream)
