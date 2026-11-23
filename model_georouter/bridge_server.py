"""Слой 1: HTTP/1.1-сервер моста на 127.0.0.1 — приём, инспекция, маршрутизация, replay и relay."""

from __future__ import annotations

import select
import socket
import socketserver
import ssl
import threading
import time
from _thread import LockType
from dataclasses import dataclass, replace
from typing import cast
from urllib.parse import urlsplit

from . import SERVICE_NAME, SERVICE_NAME_ALIASES, __version__, config_guard, egress_admission, egress_hedge
from ._typing import override
from .adapters import AdapterRegistry, AdapterRoute, AdminAdapter
from .claude_code_routing import CatalogTicketStore, ClaudeCodeRouting
from .config import BridgeConfig, ConfigManager, PoolConfig, UpstreamConfig
from .error_handler import ErrorHandler, json_response_bytes
from .errors import (
    AllProxiesPenalizedError,
    BadRequestError,
    BridgeError,
    ClientTimeoutError,
    EgressConnectError,
    EgressTimeoutError,
    FramingError,
    NotFoundError,
    SlotsExhaustedError,
    UpstreamProtocolError,
    UpstreamTimeoutError,
)
from .geo_cache import GeoCache
from .http_wire import (
    BODY_METHODS,
    HOP_BY_HOP,
    Framing,
    HeadError,
    RequestHead,
    ResponseHead,
    build_client_head,
    connection_tokens,
    framing_for_request,
    framing_for_response,
    parse_request_head,
    parse_response_head,
    read_head_block,
)
from .logging_utils import get_logger
from .model_router import (
    THINKING_BINDING_BETA,
    BodyInspector,
    ModelRouter,
    RegionErrorClassifier,
    ensure_anthropic_thinking_drop_block,
    model_from_path,
    sanitize_google_v1beta_body,
    sanitize_openai_empty_messages,
)
from .proxy_pool import EgressSettings, ProxyEndpoint, ProxyPoolManager
from .relay import RELAY_CHUNK, SSERelay, close_quietly, configure_socket
from .retry_budget import (
    GEOROUTER_OWNER,
    AttemptEvent,
    AttemptState,
    EgressDeadline,
    RetryBudgetSpec,
    RouteCircuitBreaker,
    bridge_replay_allowed,
    consume_live_retry,
    evaluate_replay_for_endpoint,
    transition,
)
from .upstream_state import UpstreamStateWriter

LOGGER = get_logger("model_georouter.server")

DIAGNOSTIC_BODY_LIMIT = 16384
DRAIN_MAX_SECONDS = 2.0
DRAIN_MAX_BYTES = 64 * 1024 * 1024
OVERLOADED_503 = (
    b"HTTP/1.1 503 Service Unavailable\r\nRetry-After: 1\r\nContent-Type: text/plain\r\n"
    b"Connection: close\r\nContent-Length: 20\r\n\r\nService Unavailable\n"
)
SKIPPED_REQUEST_HEADERS = frozenset({"host", "accept-encoding", "content-length", "expect"})
_REPLAY_FLAGS = {"header_timeout": "timed_out", "protocol_error": "protocol_error"}


# ───────────────────────────── слоты и метрики ─────────────────────────────


class SlotPool:
    """Неблокирующий счётчик слотов с изменяемым лимитом (hot-reload)."""

    _limit: int
    _in_use: int
    _lock: LockType

    def __init__(self, limit: int):
        self._limit, self._in_use, self._lock = limit, 0, threading.Lock()

    def resize(self, limit: int) -> None:
        with self._lock: self._limit = limit

    def try_acquire(self) -> bool:
        with self._lock:
            if self._in_use >= self._limit:
                return False
            self._in_use += 1
            return True

    def release(self) -> None:
        with self._lock: self._in_use = max(0, self._in_use - 1)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"limit": self._limit, "in_use": self._in_use, "available": max(0, self._limit - self._in_use)}


class SlotLease:
    def __init__(self, pool: SlotPool):
        self._pool: SlotPool | None = pool

    def release(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None: pool.release()


class SplitSlots:
    """Раздельные бюджеты direct/proxy: зависшая прокси не может занять слоты direct-моделей."""

    _pools: dict[str, SlotPool]

    def __init__(self, direct_limit: int, proxy_limit: int):
        self._pools = {"direct": SlotPool(direct_limit), "proxy": SlotPool(proxy_limit)}

    def acquire(self, kind: str, limit: int) -> SlotLease | None:
        pool = self._pools[kind]
        pool.resize(limit)
        return SlotLease(pool) if pool.try_acquire() else None

    def stats(self) -> dict[str, dict[str, int]]:
        return {kind: pool.stats() for kind, pool in self._pools.items()}


class Metrics:
    _lock: LockType

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._lock = threading.Lock()

    def incr(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def snapshot(self) -> dict[str, int]:
        with self._lock: return dict(self._counters)


class BoundedCollector:
    """Копит payload диагностического тела не больше `limit` байт."""

    _limit: int
    data: bytearray
    truncated: bool

    def __init__(self, limit: int):
        self._limit, self.data, self.truncated = limit, bytearray(), False

    @property
    def full(self) -> bool:
        return len(self.data) >= self._limit

    def add(self, chunk: bytes) -> None:
        room = self._limit - len(self.data)
        if room > 0: self.data += chunk[:room]
        if len(chunk) > max(room, 0): self.truncated = True


@dataclass(frozen=True)
class _Runtime:
    """Производные от снимка конфигурации объекты (пересоздаются при смене снимка)."""

    registry: AdapterRegistry
    router: ModelRouter
    classifier: RegionErrorClassifier


class _ReplayRequested(Exception):
    """Внутренний сигнал: прямой маршрут получил подтверждённый RegionError."""


class _WorkerThreads(list[threading.Thread]):
    @override
    def append(self, thread: threading.Thread) -> None:
        self[:] = [t for t in self if t.is_alive()]
        super().append(thread)

    def join(self) -> None:
        while self:
            self.pop(0).join(0.5)


# ───────────────────────────── сервер ─────────────────────────────


class BridgeServer(socketserver.ThreadingTCPServer):
    allow_reuse_address: bool = True
    daemon_threads: bool = True
    block_on_close: bool = False
    request_queue_size: int = 64
    config_manager: ConfigManager
    proxy_manager: ProxyPoolManager
    geo_cache: GeoCache
    slots: SplitSlots
    ingress: SlotPool
    metrics: Metrics
    error_handler: ErrorHandler
    admin: AdminAdapter
    _threads: _WorkerThreads
    egress: egress_hedge.EgressHedgeRunner
    admission: egress_admission.EgressAdmission
    state_writer: UpstreamStateWriter | None
    _catalog_tickets: CatalogTicketStore

    def __init__(
        self,
        config_manager: ConfigManager,
        *,
        proxy_manager: ProxyPoolManager | None = None,
        geo_cache: GeoCache | None = None,
        ssl_context: ssl.SSLContext | None = None,
        state_writer: UpstreamStateWriter | None = None,
    ) -> None:
        cfg = config_manager.get().server
        self.config_manager, self.proxy_manager = config_manager, proxy_manager or ProxyPoolManager()
        self.geo_cache = geo_cache or GeoCache(cfg.geo_cache_file or None, cfg.geo_cache_ttl_seconds)
        self.slots, self.ingress = SplitSlots(cfg.direct_slots, cfg.proxy_slots), SlotPool(cfg.max_connections)
        self.metrics, self.error_handler, self.admin = Metrics(), ErrorHandler(), AdminAdapter()
        self._ssl_context: ssl.SSLContext | None = ssl_context
        self._runtime: tuple[BridgeConfig, _Runtime] | None = None
        self.retry_budget: RetryBudgetSpec = RetryBudgetSpec()
        self.started_at: float = time.monotonic()
        self._catalog_clock = time.monotonic
        self._catalog_tickets = CatalogTicketStore(lambda: self._catalog_clock())
        self._threads = _WorkerThreads()
        self.state_writer = (
            state_writer
            if state_writer is not None
            else (
                UpstreamStateWriter(config_manager.path.parent / "state" / "upstream-state.json")
                if config_manager.path is not None
                else None
            )
        )
        self.egress = egress_hedge.build_egress(
            self.proxy_manager, cfg.hedged_connect_enabled, recorder=self.state_writer
        )
        self.admission = egress_admission.EgressAdmission(config=config_guard.validate(config_manager.get()))
        super().__init__((cfg.listen, cfg.port), _Handler)

    # --- служебное ---

    @property
    def ssl_context(self) -> ssl.SSLContext:
        if self._ssl_context is None: self._ssl_context = ssl.create_default_context()
        return self._ssl_context

    def runtime_for(self, snapshot: BridgeConfig) -> _Runtime:
        if self._runtime is None or self._runtime[0] is not snapshot:
            reg, rtr = AdapterRegistry.from_config(snapshot), ModelRouter.from_config(snapshot)
            cls = RegionErrorClassifier(snapshot.server.geo_error_signatures)
            self._runtime = (snapshot, _Runtime(reg, rtr, cls))
        return self._runtime[1]

    @override
    def handle_error(self, request: socket.socket | tuple[bytes, socket.socket], client_address: object) -> None:
        LOGGER.exception("unhandled error in connection handler")

    def _check_reload(self) -> None:
        """Hot-reload: ёмкость входного лимита следует за `max_connections` актуального снимка."""
        self.ingress.resize(self.config_manager.get().server.max_connections)

    @override
    def process_request(self, request: socket.socket | tuple[bytes, socket.socket], client_address: object) -> None:
        """Лимит соединений проверяется до создания потока: при переполнении — inline 503 и закрытие в accept-цикле."""
        # ThreadingTCPServer передаёт сокет; вариант tuple нужен общей сигнатуре socketserver.
        request = cast(socket.socket, request)
        self._check_reload()
        if not self.ingress.try_acquire():
            self.metrics.incr("status_503")
            self.metrics.incr("rejected_503")
            try:
                request.settimeout(0.5); request.sendall(OVERLOADED_503); request.shutdown(socket.SHUT_WR)
                request.setblocking(False)  # дренаж без ожидания: accept-цикл не должен простаивать
                _ = request.recv(65536)  # вычитать пришедшее: закрытие с непрочитанным даёт RST
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.ingress.release()
            raise

    @override
    def process_request_thread(self, request: socket.socket | tuple[bytes, socket.socket], client_address: object) -> None:  # fmt: skip
        try: super().process_request_thread(request, client_address)
        finally: self.ingress.release()

    def start_background(self) -> threading.Thread:
        thread = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        thread.start()
        self._threads.append(thread)
        return thread

    def stop(self) -> None: self.shutdown(); self.server_close()

    # --- AdminProvider ---

    def health_payload(self) -> dict[str, object]:
        snapshot = self.config_manager.get()
        host, port = self.server_address[:2]
        return {"status": "ok", "service": SERVICE_NAME, "service_aliases": list(SERVICE_NAME_ALIASES),
                "version": __version__, "listen": f"{host}:{port}", "slots": self.slots.stats(),
                "pools": self.proxy_manager.stats(snapshot.pools.values()),
                "geo_cache": {"entries": len(self.geo_cache)}, "config_error": self.config_manager.last_error}

    def metrics_payload(self) -> dict[str, object]:
        return {"counters": self.metrics.snapshot(), "slots": self.slots.stats(),
                "geo_cache_entries": len(self.geo_cache), "uptime_seconds": round(time.monotonic() - self.started_at, 3)}

    def flush_cache(self) -> int: return self.geo_cache.flush()


# ───────────────────────────── обработчик соединения ─────────────────────────────


def _is_direct(pool: PoolConfig) -> bool:
    return pool.type == "direct"


_BREAKERS: dict[str, RouteCircuitBreaker] = {}
_BREAKER_LOCK = threading.Lock()


def _breaker(route_id: str, pool: str = "") -> RouteCircuitBreaker:
    """F-003: живой breaker на пару (upstream, pool); пустой pool — legacy-ключ route_id."""
    key = f"{pool}:{route_id}" if pool else route_id
    with _BREAKER_LOCK:
        return _BREAKERS.setdefault(key, RouteCircuitBreaker(key))


_CLAUDE_MODEL_PREFIXES = ("claude-", "anthropic/")


def _ensure_anthropic_beta(req: RequestHead, beta: str) -> None:
    """Дописать значение в `anthropic-beta` (все вхождения заголовка сливаются в одно, порядок сохранён)."""
    tokens = [t.strip() for k, v in req.headers if k.lower() == "anthropic-beta" for t in v.split(",") if t.strip()]
    if beta in tokens:
        return
    req.headers[:] = [(key, value) for key, value in req.headers if key.lower() != "anthropic-beta"]
    req.headers.append(("anthropic-beta", ", ".join([*tokens, beta])))


def build_upstream_request(
    req: RequestHead, path: str, query: str, upstream: UpstreamConfig, body_length: int
) -> bytes:
    """Запрос к upstream: hop-by-hop убраны, `Accept-Encoding: identity`, `Connection: close`."""
    default_port = 443 if upstream.use_tls else 80
    host_header = upstream.host if upstream.port == default_port else f"{upstream.host}:{upstream.port}"
    target = upstream.base_path + path + (f"?{query}" if query else "")
    _opencode_route = upstream.base_path.startswith(("/go/v1", "/zen/v1"))
    skip = SKIPPED_REQUEST_HEADERS | HOP_BY_HOP | connection_tokens(req.headers)
    if _opencode_route:
        skip = skip | {"user-agent"}
    lines = [f"{req.method} {target} HTTP/1.1", f"Host: {host_header}"]
    lines.extend(f"{key}: {value}" for key, value in req.headers if key.lower() not in skip)
    if _opencode_route:
        lines.append("User-Agent: OpenCode")
    lines.append("Accept-Encoding: identity")
    lines.append("Connection: close")
    if body_length or req.method in BODY_METHODS:
        lines.append(f"Content-Length: {body_length}")
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


class _Handler(ClaudeCodeRouting, socketserver.BaseRequestHandler):
    _last_proxy_endpoint: ProxyEndpoint | None = None
    committed: bool
    body_done: bool
    replayed: bool
    started: float
    status: int
    bytes_sent: int
    method: str
    path: str
    pool_name: str
    proxy_endpoint: str
    outcome: str

    def __init__(self, request: socket.socket, client_address: object, server: socketserver.BaseServer) -> None:
        self._context: (
            tuple[socket.socket, BridgeConfig, _Runtime, RequestHead, AdapterRoute, str, UpstreamConfig, BodyInspector]
            | None
        ) = None
        self.committed, self.body_done, self.replayed = False, False, False
        self.retry_allowed: bool = True
        self.started, self.status, self.bytes_sent = time.monotonic(), 0, 0
        self.method, self.path, self.pool_name, self.proxy_endpoint, self.outcome = "-", "-", "-", "-", "-"
        self.model: str | None = None
        self._attempt_ok: bool = False
        super().__init__(request, client_address, server)

    @property
    def _bridge(self) -> BridgeServer:
        return cast(BridgeServer, self.server)

    @property
    def _ctx(
        self,
    ) -> tuple[socket.socket, BridgeConfig, _Runtime, RequestHead, AdapterRoute, str, UpstreamConfig, BodyInspector]:
        assert self._context is not None
        return self._context

    @_ctx.setter
    def _ctx(
        self,
        value: tuple[
            socket.socket, BridgeConfig, _Runtime, RequestHead, AdapterRoute, str, UpstreamConfig, BodyInspector
        ],
    ) -> None:
        self._context = value

    @override
    def handle(self) -> None:
        sock = cast(socket.socket, self.request)
        configure_socket(sock)
        try:
            self._serve(sock)
        except BaseException as exc:  # noqa: BLE001 — любой сбой превращается в безопасный ответ или тихое закрытие
            self._fail(sock, exc)
        finally:
            self._log_request()

    # --- ответы ---

    def _send_bytes(self, sock: socket.socket, data: bytes, timeout: float = 5.0) -> None:
        try:
            sock.settimeout(timeout)
            sock.sendall(data)
        except OSError:
            pass

    def _respond_json(self, sock: socket.socket, status: int, payload: dict[str, object], headers: dict[str, str] | None = None, head_only: bool = False) -> None:  # fmt: skip
        self.status = status
        self.outcome = "local"
        self._bridge.metrics.incr(f"status_{status}")
        data = json_response_bytes(status, payload, headers)
        if head_only:  # HEAD: статус и заголовки (в т.ч. Content-Length) как у GET, без тела
            data = data.partition(b"\r\n\r\n")[0] + b"\r\n\r\n"
        self._send_bytes(sock, data)

    def _fail(self, sock: socket.socket, exc: BaseException) -> None:
        srv = self._bridge
        if self.committed:
            srv.metrics.incr("aborted_streams")
            self.outcome = self.outcome if self.outcome != "-" else "aborted"
            if not isinstance(exc, (BridgeError, OSError)):
                LOGGER.error("stream aborted by internal error: %s", type(exc).__name__)
            return
        if isinstance(exc, (ConnectionError, BrokenPipeError, KeyboardInterrupt)):
            self.outcome = "client_disconnected"
            return
        if not isinstance(exc, BridgeError):
            # LOG014 false positive: _fail() always runs inside `except` (see handle()),
            # so sys.exception() is set and exc_info carries the real traceback.
            LOGGER.error("internal error: %s", type(exc).__name__, exc_info=True)  # noqa: LOG014
        response = srv.error_handler.convert(exc)
        self.status = response.status
        self.outcome = "error:" + str(response.payload["error"]["type"])
        srv.metrics.incr(f"status_{response.status}")
        if response.status == 503: srv.metrics.incr("rejected_503")
        elif response.status == 413: srv.metrics.incr("rejected_413")
        self._send_bytes(sock, response.to_bytes())
        if not self.body_done:
            self._drain(sock)

    @staticmethod
    def _drain(sock: socket.socket) -> None:
        """Вычитать недочитанное тело, чтобы закрытие не превратилось в RST и не стёрло ответ."""
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            return
        end = time.monotonic() + DRAIN_MAX_SECONDS
        total = 0
        while total < DRAIN_MAX_BYTES:
            remaining = end - time.monotonic()
            if remaining <= 0:
                break
            try:
                readable, _, _ = select.select([sock], [], [], min(0.5, remaining))
                if not readable:
                    break
                data = sock.recv(65536)
            except (OSError, ValueError):
                break
            if not data:
                break
            total += len(data)

    def _log_request(self) -> None:
        LOGGER.info(
            "request method=%s path=%s model=%s pool=%s proxy=%s status=%s outcome=%s bytes=%d replayed=%s dur_ms=%d",
            self.method, self.path, self.model or "-", self.pool_name, self.proxy_endpoint,
            self.status or "-", self.outcome, self.bytes_sent, self.replayed,
            int((time.monotonic() - self.started) * 1000),
        )

    # --- основной поток ---

    def _sanitize_request_body(self, req: RequestHead, route: AdapterRoute, inspector: BodyInspector) -> None:
        if req.method != "POST" or inspector.size <= 0:
            return
        if "/v1beta/models/" in route.path:
            inspector.replace_body(sanitize_google_v1beta_body(inspector.body))
        if route.path.endswith("/chat/completions"):
            inspector.replace_body(sanitize_openai_empty_messages(inspector.body))
        if route.path.endswith("/v1/messages") and self.model and self.model.startswith(_CLAUDE_MODEL_PREFIXES):
            raw_body = inspector.body
            new_body, needs_beta = ensure_anthropic_thinking_drop_block(raw_body)
            if new_body is not raw_body:
                inspector.replace_body(new_body)
            if needs_beta:
                _ensure_anthropic_beta(req, THINKING_BINDING_BETA)

    def _serve(self, sock: socket.socket) -> None:
        srv = self._bridge
        snapshot = srv.config_manager.get()
        cfg = snapshot.server
        try:
            block, rest = read_head_block(sock, timeout=cfg.body_timeout)
        except HeadError as exc:
            if exc.kind == "eof":
                self.body_done, self.outcome = True, "client_closed"
                return
            if exc.kind == "timeout":
                raise ClientTimeoutError("request headers not received in time") from None
            raise BadRequestError("request headers are too large") from None
        req = parse_request_head(block)
        target = req.target
        if "://" in target.split("?", 1)[0]:
            parts = urlsplit(target)
            target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        path, _, query = target.partition("?")
        self.method, self.path = req.method, path
        self.retry_allowed = bridge_replay_allowed(dict(req.headers))
        srv.metrics.incr("requests_total")
        self.body_done = req.get("content-length") in (None, "0") and req.get("transfer-encoding") is None
        runtime = srv.runtime_for(snapshot)
        if srv.admin.matches(path):
            status, payload, headers = srv.admin.handle(req.method, path, srv)
            self._respond_json(sock, status, payload, headers, head_only=req.method == "HEAD")
            if not self.body_done:
                self._drain(sock)
            return
        route = runtime.registry.resolve(path)
        if route is None:
            raise NotFoundError(f"no adapter serves path {path}")
        inspector = self._read_request_body(sock, req, rest, cfg.body_buffer_max_bytes, cfg.body_timeout)
        self.model = inspector.model or model_from_path(path)
        if (catalog_ticket := self._catalog_callback_ticket(req, snapshot)) is not None:
            self._serve_catalog_callback(sock, route, catalog_ticket)
            return
        self._sanitize_request_body(req, route, inspector)
        decision = runtime.router.decide(self.model, route.upstream, srv.geo_cache, default_pool=route.pool or None)
        if decision.source == "geo_cache":
            srv.metrics.incr("geo_cache_hits")
        upstream, fallback = snapshot.upstreams[route.upstream], snapshot.geo_fallback_pool
        self._context = (sock, snapshot, runtime, req, route, query, upstream, inspector)
        if self._sidecar_matches(snapshot, route, inspector) and self._attempt_claude_sidecar(decision.pool):
            return
        now_tick = int(time.monotonic())
        replay_pool = fallback if (fallback and fallback != decision.pool and _is_direct(snapshot.pools[decision.pool]) and self.retry_allowed and snapshot.failover.enabled) else ""  # fmt: skip
        probe = decision.pool if _breaker(route.upstream, decision.pool).try_acquire(now_tick) else ""
        pool_name, replay_pool = (decision.pool, replay_pool) if probe or not replay_pool else (replay_pool, "")
        budget = srv.admission.f08_v2_budget
        long_hdr = egress_admission.is_streaming_request(path, stream_body=inspector.stream and budget is not None)
        deadline = EgressDeadline.from_timeout(budget.t_idle if budget is not None and long_hdr else cfg.total_egress_deadline)  # fmt: skip
        try:
            with srv.admission.admit(pool_name, path, deadline=deadline, is_stream=inspector.stream) as ticket:
                _ = srv.config_manager.retry_budget.mint_on_admitted_primary()
                self._attempt(pool_name, replay_pool, ticket.deadline, probe=probe)
        except _ReplayRequested:
            srv.geo_cache.mark_blocked(route.upstream, self.model, cfg.geo_cache_ttl_seconds)
            srv.metrics.incr("replays")
            self.replayed = True
            self._attempt(fallback, "", deadline, False, fallback if _breaker(route.upstream, fallback).try_acquire(now_tick) else "")  # fmt: skip

    def _read_request_body(
        self, sock: socket.socket, req: RequestHead, rest: bytes, limit: int, timeout: float
    ) -> BodyInspector:
        inspector = BodyInspector(limit)
        framing = framing_for_request(req.headers, inspector.feed, limit)
        if not framing.complete and (req.get("expect") or "").lower() == "100-continue":
            self._send_bytes(sock, b"HTTP/1.1 100 Continue\r\n\r\n")
        deadline = time.monotonic() + timeout  # единый дедлайн на всё тело: побайтовый trickle не продлевает его
        try:
            if rest:
                _ = framing.feed(rest)
            while not framing.complete:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ClientTimeoutError("request body not received in time")
                sock.settimeout(remaining)
                data = sock.recv(65536)
                if not data:
                    raise BadRequestError("request body is truncated")
                _ = framing.feed(data)
        except TimeoutError:
            raise ClientTimeoutError("request body not received in time") from None
        except FramingError as exc:
            raise BadRequestError(str(exc)) from exc
        self.body_done = True
        return inspector

    # --- одна попытка к upstream ---

    @staticmethod
    def _attempt_event(exc: Exception, deadline: EgressDeadline, endpoint: ProxyEndpoint | None) -> AttemptEvent:
        if isinstance(exc, UpstreamTimeoutError):
            kind = "header_timeout"
        elif isinstance(exc, UpstreamProtocolError):
            kind = "protocol_error"
        else:
            kind = "penalized" if isinstance(exc, AllProxiesPenalizedError) else "connect_error"
        return AttemptEvent(kind=kind, deadline_expired=deadline.expired(), has_endpoint=endpoint is not None)

    def _attempt(self, pool_name: str, replay: str, deadline: EgressDeadline, allow_fallback: bool = True, probe: str = "") -> None:  # fmt: skip
        _, snapshot, _, _, route, _, upstream, inspector = self._ctx
        long_header = egress_admission.is_streaming_request(route.path, stream_body=inspector.stream and self._bridge.admission.f08_v2_budget is not None)
        srv, cfg, pool = self._bridge, snapshot.server, snapshot.pools[pool_name]
        b, fb, direct = srv.retry_budget, snapshot.geo_fallback_pool, _is_direct(pool)
        single = allow_fallback and self.retry_allowed and snapshot.failover.enabled and bool(fb) and fb != pool_name and not direct and len(pool.proxies) == 1  # fmt: skip
        state = AttemptState(
            pool_name, fb if single else "", single, self.committed, b.penalize_on_header_timeout,
            has_alternative=single or len(pool.proxies) > 1, is_long_header=long_header,
            penalize_without_alternative=cfg.penalize_header_timeout_without_alternative)
        self.pool_name, self.proxy_endpoint, self._last_proxy_endpoint = pool_name, "-", None
        up: socket.socket | None = None
        lease: SlotLease | None = None
        self._attempt_ok = False
        try:
            if not direct:
                try:
                    srv.proxy_manager.ensure_available(pool)  # circuit breaker: до слота, за доли миллисекунды
                except AllProxiesPenalizedError as exc:
                    event = self._attempt_event(exc, deadline, None)
                    _breaker(route.upstream, pool_name).record_failure(int(time.monotonic()))
                    state = transition(state, event)
                    if state.next_pool and self._live_replay_allowed(0, False, event.kind):
                        return self._attempt(state.next_pool, "", deadline, False)
                    raise
            slot_kind = "direct" if direct else "proxy"
            lease = srv.slots.acquire(slot_kind, cfg.direct_slots if direct else cfg.proxy_slots)
            if lease is None:
                raise SlotsExhaustedError(slot_kind)
            try:
                up = self._connect_upstream(pool, upstream, deadline, single)
                self._send_upstream_request(up, cfg.body_timeout, deadline)
                hdr_wait = cfg.headers_timeout if long_header else min(cfg.headers_timeout, b.primary_first_header_timeout if single else srv.admission.current(deadline).header_deadline_s)  # fmt: skip
                head, rest = self._read_upstream_head(up, hdr_wait, None if long_header else deadline)
            except (AllProxiesPenalizedError, EgressConnectError, UpstreamTimeoutError, UpstreamProtocolError) as exc:
                event = self._attempt_event(exc, deadline, self._last_proxy_endpoint)
                _breaker(route.upstream, pool_name).record_failure(int(time.monotonic()))
                state = transition(state, event)
                if self._last_proxy_endpoint is not None and state.should_penalize:
                    srv.proxy_manager.penalize(self._last_proxy_endpoint, cfg.proxy_fail_penalty_seconds)
                if state.next_pool and self._live_replay_allowed(0, False, event.kind):
                    return self._attempt(state.next_pool, "", deadline, False)
                if replay and self._live_replay_allowed(0, False, self._attempt_event(exc, deadline, None).kind):
                    raise _ReplayRequested
                raise
            else:
                self._finish_upstream_response(up, head, rest, replay)
        finally:
            close_quietly(up)
            if lease is not None: lease.release()
            if probe: _breaker(route.upstream, probe).release_probe(self._attempt_ok, int(time.monotonic()))  # fmt: skip

    @staticmethod
    def _read_buffered_body(up: socket.socket, framing: Framing, rest: bytes, timeout: float) -> None:
        if rest:
            _ = framing.feed(rest)
        deadline = time.monotonic() + timeout
        while not framing.complete:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise UpstreamTimeoutError("upstream response body timeout")
            up.settimeout(remaining)
            chunk = up.recv(65536)
            if not chunk:
                if not framing.on_eof():
                    raise UpstreamProtocolError("upstream response body truncated")
                break
            _ = framing.feed(chunk)

    def _send_transformed_response(self, new_head: ResponseHead, new_body: bytes) -> None:
        sock, snapshot, _, req, *_ = self._ctx
        srv, cfg = self._bridge, snapshot.server
        norm = [(k, v) for k, v in new_head.headers if k.lower() not in {"transfer-encoding", "content-length"}]
        norm.append(("Content-Length", str(len(new_body))))
        final_head = ResponseHead(new_head.version, new_head.status, new_head.reason, norm)
        self.status, self.committed = final_head.status, True
        try:
            sock.settimeout(cfg.body_timeout)
            sock.sendall(build_client_head(final_head))
            if req.method != "HEAD" and new_body:
                sock.sendall(new_body)
            self.outcome, self.bytes_sent = "complete", len(new_body)
        except OSError:
            self.outcome = "client_cancelled"
            srv.metrics.incr("client_cancelled")

    def _live_replay_allowed(self, status: int, region: bool, kind: str = "") -> bool:
        """F-005: fences, breaker и retry-токен до коммита; `effort` пиннится константой (тело шлётся байт-в-байт)."""
        _, snapshot, _, req, route, _, _, _ = self._ctx
        if not (self.retry_allowed and snapshot.failover.enabled and (region or kind or 500 <= status < 600)):
            return False
        facts: dict[str, object] = {"endpoint": self.path, "status_code": status, "downstream_committed": False,
            "tool_side_effects": False, "canonical_model_id": self.model, "effort": "bridge-pinned",
            "retry_owner": GEOROUTER_OWNER, "control_headers": dict(req.headers)}
        if kind: facts[_REPLAY_FLAGS.get(kind, "connect_error")] = True  # fmt: skip
        if not evaluate_replay_for_endpoint(facts)["replay_allowed"]: return False
        if not _breaker(route.upstream, self.pool_name).allows(int(time.monotonic())): return False
        return consume_live_retry(dict(req.headers), self._bridge.config_manager.retry_budget, enabled=True,
                                  declared=GEOROUTER_OWNER if snapshot.geo_fallback_pool else "",
                                  configured=snapshot.failover.max_total_attempts)

    def _finish_upstream_response(self, up: socket.socket, head: ResponseHead, rest: bytes, replay: str) -> None:
        _, snapshot, runtime, req, route, *_ = self._ctx
        srv, cfg = self._bridge, snapshot.server
        adapter = runtime.registry.get(route.adapter)
        ok, status = 200 <= head.status < 300, head.status
        self._attempt_ok, breaker = ok, _breaker(route.upstream, self.pool_name)
        if ok: breaker.record_success()
        else: breaker.record_failure(int(time.monotonic()), client_validation=400 <= status < 500 and status not in (403, 429))  # fmt: skip
        if adapter and adapter.wants_transform(route, head):
            body = bytearray()
            framing = framing_for_response(req.method, status, head.headers, body.extend)
            self._read_buffered_body(up, framing, rest, cfg.body_timeout)
            return self._send_transformed_response(*adapter.transform_response(route, head, bytes(body)))
        framing = framing_for_response(req.method, status, head.headers)
        initial, initial_fed, region = rest, False, False
        if head.status == 403:
            collector = BoundedCollector(DIAGNOSTIC_BODY_LIMIT)
            framing = framing_for_response(req.method, head.status, head.headers, collector.add)
            initial, initial_fed = self._read_diagnostic(up, framing, collector, rest, cfg.body_timeout), True
            result = runtime.classifier.classify(403, bytes(collector.data), head.get("content-encoding"))
            srv.metrics.incr("classified_403_region" if result.is_region_error else "classified_403_plain")
            region = result.is_region_error
            if collector.truncated or (collector.full and not framing.complete):
                srv.metrics.incr("classified_403_truncated")
                LOGGER.warning("403 body exceeds %d bytes: signatures beyond it are not checked", DIAGNOSTIC_BODY_LIMIT)
        if replay and not ok and self._live_replay_allowed(head.status, region):
            raise _ReplayRequested
        self._relay(up, head, framing, initial, initial_fed)

    def _connect_upstream(self, pool: PoolConfig, up: UpstreamConfig, dl: EgressDeadline, single: bool = False) -> socket.socket:  # fmt: skip
        return self._bridge.egress.connect(pool, (up.host, up.port), replace(EgressSettings.from_server(self._ctx[1].server)), handler=(self, dl, EgressTimeoutError), up=up, single=single)  # fmt: skip

    def _send_upstream_request(self, up: socket.socket, timeout: float, dl: EgressDeadline | None = None) -> None:
        _, _, _, req, route, query, upstream, inspector = self._ctx
        body = inspector.view()
        try:
            up.settimeout(dl.clamp(timeout) if dl is not None else timeout)
            up.sendall(build_upstream_request(req, route.path, query, upstream, len(body)))
            if len(body):
                if dl is not None:
                    up.settimeout(dl.clamp(timeout))
                up.sendall(body)
        except TimeoutError as exc:
            if dl is not None and dl.expired():
                raise UpstreamTimeoutError(f"upstream request send exceeded {timeout:g}s deadline") from None
            raise UpstreamProtocolError(f"failed to send request to upstream: {type(exc).__name__}") from exc
        except OSError as exc:
            raise UpstreamProtocolError(f"failed to send request to upstream: {type(exc).__name__}") from exc
        finally:
            body.release()

    @staticmethod
    def _read_upstream_head(
        up: socket.socket, timeout: float, deadline: EgressDeadline | None = None
    ) -> tuple[ResponseHead, bytes]:
        initial, phase = b"", EgressDeadline.from_timeout(timeout)
        while True:
            try:
                step = deadline.clamp(phase.clamp(timeout)) if deadline else phase.clamp(timeout)
                block, rest = read_head_block(up, timeout=step, initial=initial)
            except TimeoutError:
                raise UpstreamTimeoutError(f"upstream sent no response headers within {timeout:g}s") from None
            except HeadError as exc:
                if exc.kind == "timeout":
                    raise UpstreamTimeoutError(f"upstream sent no response headers within {timeout:g}s") from None
                raise UpstreamProtocolError("upstream closed the connection before sending a response") from None
            except OSError as exc:
                raise UpstreamProtocolError(f"upstream connection failed: {type(exc).__name__}") from exc
            head = parse_response_head(block)
            if 100 <= head.status < 200 and head.status != 101:
                initial = rest
                continue
            return head, rest

    @staticmethod
    def _read_diagnostic(
        up: socket.socket, framing: Framing, collector: BoundedCollector, rest: bytes, timeout: float
    ) -> bytes:
        """Прочитать (≤ лимита) тело 403 для классификации; вернуть уже потреблённые wire-байты."""
        wire = bytearray(rest[: framing.feed(rest)] if rest else b"")
        deadline = time.monotonic() + timeout
        try:
            while not framing.complete and not collector.full:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                up.settimeout(remaining)
                data = up.recv(RELAY_CHUNK)
                if not data:
                    break
                wire += data[: framing.feed(data)]
        except (OSError, FramingError):  # включая таймаут: классифицируем то, что успели прочитать
            pass
        return bytes(wire)

    def _relay(self, up: socket.socket, head: ResponseHead, framing: Framing, initial: bytes, fed: bool) -> None:
        sock, snapshot, _runtime, _req, route, *_ = self._ctx
        cfg = snapshot.server
        self.status, self.committed = head.status, True
        try:
            sock.settimeout(cfg.body_timeout)
            sock.sendall(build_client_head(head))
        except OSError:
            self.outcome = "client_cancelled"
            self._bridge.metrics.incr("client_cancelled")
            return
        relay = SSERelay(inactivity_timeout=cfg.inactivity_timeout, write_timeout=cfg.body_timeout)
        ticket = self._bridge.admission.current(None, route.path)
        result = relay.relay(up, sock, framing, initial=initial, initial_fed=fed, on_chunk=ticket.on_chunk)
        self.outcome, self.bytes_sent = result.outcome, result.bytes_sent
        self._bridge.metrics.incr(f"status_{head.status}")
        self._bridge.metrics.incr(f"relay_{result.outcome}")
        if not result.ok:
            self._bridge.metrics.incr("aborted_streams")
