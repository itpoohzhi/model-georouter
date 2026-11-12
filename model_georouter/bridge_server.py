"""Слой 1: HTTP/1.1-сервер моста на 127.0.0.1 — приём, инспекция, маршрутизация, replay и relay."""

from __future__ import annotations

import select
import socket
import socketserver
import ssl
import threading
import time
from dataclasses import dataclass, replace
from urllib.parse import urlsplit

from . import SERVICE_NAME, SERVICE_NAME_ALIASES, __version__
from .adapters import AdapterRegistry, AdapterRoute, AdminAdapter
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
)
from .proxy_pool import EgressSettings, ProxyEndpoint, ProxyPoolManager
from .relay import RELAY_CHUNK, SSERelay, close_quietly, configure_socket
from .retry_budget import AttemptEvent, AttemptState, EgressDeadline, RetryBudgetSpec, transition

LOGGER = get_logger("model_georouter.server")
_RETRY_BUDGET = RetryBudgetSpec()

DIAGNOSTIC_BODY_LIMIT = 16384
DRAIN_MAX_SECONDS = 2.0
DRAIN_MAX_BYTES = 64 * 1024 * 1024
OVERLOADED_503 = (
    b"HTTP/1.1 503 Service Unavailable\r\nRetry-After: 1\r\nContent-Type: text/plain\r\n"
    b"Connection: close\r\nContent-Length: 20\r\n\r\nService Unavailable\n"
)
SKIPPED_REQUEST_HEADERS = frozenset({"host", "accept-encoding", "content-length", "expect"})


# ───────────────────────────── слоты и метрики ─────────────────────────────


class SlotPool:
    """Неблокирующий счётчик слотов с изменяемым лимитом (hot-reload)."""

    def __init__(self, limit: int):
        self._limit, self._in_use, self._lock = limit, 0, threading.Lock()

    def resize(self, limit: int) -> None:
        with self._lock:
            self._limit = limit

    def try_acquire(self) -> bool:
        with self._lock:
            if self._in_use >= self._limit:
                return False
            self._in_use += 1
            return True

    def release(self) -> None:
        with self._lock:
            self._in_use = max(0, self._in_use - 1)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"limit": self._limit, "in_use": self._in_use, "available": max(0, self._limit - self._in_use)}


class SlotLease:
    def __init__(self, pool: SlotPool):
        self._pool: SlotPool | None = pool

    def release(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.release()


class SplitSlots:
    """Раздельные бюджеты direct/proxy: зависшая прокси не может занять слоты direct-моделей."""

    def __init__(self, direct_limit: int, proxy_limit: int):
        self._pools = {"direct": SlotPool(direct_limit), "proxy": SlotPool(proxy_limit)}

    def acquire(self, kind: str, limit: int) -> SlotLease | None:
        pool = self._pools[kind]
        pool.resize(limit)
        return SlotLease(pool) if pool.try_acquire() else None

    def stats(self) -> dict[str, dict[str, int]]:
        return {kind: pool.stats() for kind, pool in self._pools.items()}


class Metrics:
    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._lock = threading.Lock()

    def incr(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)


class BoundedCollector:
    """Копит payload диагностического тела не больше `limit` байт."""

    def __init__(self, limit: int):
        self._limit, self.data, self.truncated = limit, bytearray(), False

    @property
    def full(self) -> bool:
        return len(self.data) >= self._limit

    def add(self, chunk: bytes) -> None:
        room = self._limit - len(self.data)
        if room > 0:
            self.data += chunk[:room]
        if len(chunk) > max(room, 0):
            self.truncated = True


@dataclass(frozen=True)
class _Runtime:
    """Производные от снимка конфигурации объекты (пересоздаются при смене снимка)."""

    registry: AdapterRegistry
    router: ModelRouter
    classifier: RegionErrorClassifier


class _ReplayRequested(Exception):
    """Внутренний сигнал: прямой маршрут получил подтверждённый RegionError."""


class _WorkerThreads(list[threading.Thread]):
    def append(self, thread: threading.Thread) -> None:
        self[:] = [t for t in self if t.is_alive()]
        super().append(thread)

    def join(self) -> None:
        while self:
            self.pop(0).join(0.5)


# ───────────────────────────── сервер ─────────────────────────────


class BridgeServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False
    request_queue_size = 64

    def __init__(
        self,
        config_manager: ConfigManager,
        *,
        proxy_manager: ProxyPoolManager | None = None,
        geo_cache: GeoCache | None = None,
        ssl_context: ssl.SSLContext | None = None,
    ):
        cfg = config_manager.get().server
        self.config_manager, self.proxy_manager = config_manager, proxy_manager or ProxyPoolManager()
        self.geo_cache = geo_cache or GeoCache(cfg.geo_cache_file or None, cfg.geo_cache_ttl_seconds)
        self.slots, self.ingress = SplitSlots(cfg.direct_slots, cfg.proxy_slots), SlotPool(cfg.max_connections)
        self.metrics, self.error_handler, self.admin = Metrics(), ErrorHandler(), AdminAdapter()
        self._ssl_context: ssl.SSLContext | None = ssl_context
        self._runtime: tuple[BridgeConfig, _Runtime] | None = None
        self.retry_budget: RetryBudgetSpec = RetryBudgetSpec()
        self.started_at: float = time.monotonic()
        self._threads = _WorkerThreads()
        super().__init__((cfg.listen, cfg.port), _Handler)

    # --- служебное ---

    @property
    def ssl_context(self) -> ssl.SSLContext:
        if self._ssl_context is None:
            self._ssl_context = ssl.create_default_context()
        return self._ssl_context

    def runtime_for(self, snapshot: BridgeConfig) -> _Runtime:
        if self._runtime is None or self._runtime[0] is not snapshot:
            reg, rtr = AdapterRegistry.from_config(snapshot), ModelRouter.from_config(snapshot)
            cls = RegionErrorClassifier(snapshot.server.geo_error_signatures)
            self._runtime = (snapshot, _Runtime(reg, rtr, cls))
        return self._runtime[1]

    def handle_error(self, request, client_address) -> None:
        LOGGER.exception("unhandled error in connection handler")

    def _check_reload(self) -> None:
        """Hot-reload: ёмкость входного лимита следует за `max_connections` актуального снимка."""
        self.ingress.resize(self.config_manager.get().server.max_connections)

    def process_request(self, request, client_address) -> None:
        """Лимит соединений проверяется до создания потока: при переполнении — inline 503 и закрытие в accept-цикле."""
        self._check_reload()
        if not self.ingress.try_acquire():
            self.metrics.incr("status_503")
            self.metrics.incr("rejected_503")
            try:
                request.settimeout(0.5)
                request.sendall(OVERLOADED_503)
                request.shutdown(socket.SHUT_WR)
                request.setblocking(False)  # дренаж без ожидания: accept-цикл не должен простаивать
                request.recv(65536)  # вычитать пришедшее без ожидания: закрытие с непрочитанным даёт RST
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.ingress.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.ingress.release()

    def start_background(self) -> threading.Thread:
        thread = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        thread.start()
        self._threads.append(thread)
        return thread

    def stop(self) -> None:
        self.shutdown()
        self.server_close()

    # --- AdminProvider ---

    def health_payload(self) -> dict:
        snapshot = self.config_manager.get()
        host, port = self.server_address[:2]
        return {
            "status": "ok",
            "service": SERVICE_NAME,
            "service_aliases": list(SERVICE_NAME_ALIASES),
            "version": __version__,
            "listen": f"{host}:{port}",
            "slots": self.slots.stats(),
            "pools": self.proxy_manager.stats(snapshot.pools.values()),
            "geo_cache": {"entries": len(self.geo_cache)},
            "config_error": self.config_manager.last_error,
        }

    def metrics_payload(self) -> dict:
        return {
            "counters": self.metrics.snapshot(),
            "slots": self.slots.stats(),
            "geo_cache_entries": len(self.geo_cache),
            "uptime_seconds": round(time.monotonic() - self.started_at, 3),
        }

    def flush_cache(self) -> int:
        return self.geo_cache.flush()


# ───────────────────────────── обработчик соединения ─────────────────────────────


def _is_direct(pool: PoolConfig) -> bool:
    return pool.type == "direct"


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
    skip = SKIPPED_REQUEST_HEADERS | HOP_BY_HOP | connection_tokens(req.headers)
    lines = [f"{req.method} {target} HTTP/1.1", f"Host: {host_header}"]
    lines.extend(f"{key}: {value}" for key, value in req.headers if key.lower() not in skip)
    lines.append("Accept-Encoding: identity")
    lines.append("Connection: close")
    if body_length or req.method in BODY_METHODS:
        lines.append(f"Content-Length: {body_length}")
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


class _Handler(socketserver.BaseRequestHandler):
    server: BridgeServer
    _last_proxy_endpoint: ProxyEndpoint | None = None
    _ctx: tuple[socket.socket, BridgeConfig, _Runtime, RequestHead, AdapterRoute, str, UpstreamConfig, BodyInspector]

    def setup(self) -> None:
        self.committed, self.body_done, self.replayed = False, False, False
        self.started, self.status, self.bytes_sent = time.monotonic(), 0, 0
        self.method, self.path, self.pool_name, self.proxy_endpoint, self.outcome = "-", "-", "-", "-", "-"
        self.model: str | None = None
        self._last_proxy_endpoint = None

    def handle(self) -> None:
        sock: socket.socket = self.request
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

    def _respond_json(
        self, sock: socket.socket, status: int, payload: dict, headers: dict | None = None, head_only: bool = False
    ) -> None:
        self.status = status
        self.outcome = "local"
        self.server.metrics.incr(f"status_{status}")
        data = json_response_bytes(status, payload, headers)
        if head_only:  # HEAD: статус и заголовки (в т.ч. Content-Length) как у GET, без тела
            data = data.partition(b"\r\n\r\n")[0] + b"\r\n\r\n"
        self._send_bytes(sock, data)

    def _fail(self, sock: socket.socket, exc: BaseException) -> None:
        srv = self.server
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
        self.outcome = "error:" + response.payload["error"]["type"]
        srv.metrics.incr(f"status_{response.status}")
        if response.status == 503:
            srv.metrics.incr("rejected_503")
        elif response.status == 413:
            srv.metrics.incr("rejected_413")
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
            self.method,
            self.path,
            self.model or "-",
            self.pool_name,
            self.proxy_endpoint,
            self.status or "-",
            self.outcome,
            self.bytes_sent,
            self.replayed,
            int((time.monotonic() - self.started) * 1000),
        )

    # --- основной поток ---

    def _sanitize_request_body(self, req: RequestHead, route: AdapterRoute, inspector: BodyInspector) -> None:
        if req.method != "POST" or inspector.size <= 0:
            return
        if "/v1beta/models/" in route.path:
            inspector.replace_body(sanitize_google_v1beta_body(inspector.body))
        if route.path.endswith("/v1/messages") and self.model and self.model.startswith(_CLAUDE_MODEL_PREFIXES):
            raw_body = inspector.body
            new_body, needs_beta = ensure_anthropic_thinking_drop_block(raw_body)
            if new_body is not raw_body:
                inspector.replace_body(new_body)
            if needs_beta:
                _ensure_anthropic_beta(req, THINKING_BINDING_BETA)

    def _serve(self, sock: socket.socket) -> None:
        srv = self.server
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
        self._sanitize_request_body(req, route, inspector)
        decision = runtime.router.decide(self.model, route.upstream, srv.geo_cache, default_pool=route.pool or None)
        if decision.source == "geo_cache":
            srv.metrics.incr("geo_cache_hits")
        upstream, fallback = snapshot.upstreams[route.upstream], snapshot.geo_fallback_pool
        replay_allowed = bool(fallback) and fallback != decision.pool and _is_direct(snapshot.pools[decision.pool])
        deadline = EgressDeadline.from_timeout(cfg.total_egress_deadline)
        self._ctx = (sock, snapshot, runtime, req, route, query, upstream, inspector)
        try:
            self._attempt(decision.pool, replay_allowed, deadline)
        except _ReplayRequested:
            srv.geo_cache.mark_blocked(route.upstream, self.model, cfg.geo_cache_ttl_seconds)
            srv.metrics.incr("replays")
            self.replayed = True
            self._attempt(fallback, False, deadline, False)

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
                framing.feed(rest)
            while not framing.complete:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ClientTimeoutError("request body not received in time")
                sock.settimeout(remaining)
                data = sock.recv(65536)
                if not data:
                    raise BadRequestError("request body is truncated")
                framing.feed(data)
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

    def _attempt(self, pool_name: str, replay: bool, deadline: EgressDeadline, allow_fallback: bool = True) -> None:
        _, snapshot, _, _, _, _, upstream, _ = self._ctx
        srv, cfg, pool = self.server, snapshot.server, snapshot.pools[pool_name]
        b, fb, direct = srv.retry_budget, snapshot.geo_fallback_pool, _is_direct(pool)
        single = allow_fallback and bool(fb) and fb != pool_name and not direct and len(pool.proxies) == 1
        state = AttemptState(
            pool_name,
            fb if single else "",
            single,
            self.committed,
            b.penalize_on_header_timeout,
            has_alternative=single or len(pool.proxies) > 1,
            penalize_without_alternative=cfg.penalize_header_timeout_without_alternative,
        )
        self.pool_name, self.proxy_endpoint, self._last_proxy_endpoint = pool_name, "-", None
        if not direct:
            try:
                srv.proxy_manager.ensure_available(pool)  # circuit breaker: до слота, за доли миллисекунды
            except AllProxiesPenalizedError as exc:
                state = transition(state, self._attempt_event(exc, deadline, None))
                if state.next_pool:
                    return self._attempt(state.next_pool, False, deadline, False)
                raise
        slot_kind = "direct" if direct else "proxy"
        lease = srv.slots.acquire(slot_kind, cfg.direct_slots if direct else cfg.proxy_slots)
        if lease is None:
            raise SlotsExhaustedError(slot_kind)
        up: socket.socket | None = None
        try:
            try:
                up = self._connect_upstream(pool, upstream, deadline, single)
                self._send_upstream_request(up, cfg.body_timeout, deadline)
                hdr_wait = min(cfg.headers_timeout, b.primary_first_header_timeout) if single else cfg.headers_timeout
                head, rest = self._read_upstream_head(up, hdr_wait, deadline)
            except (AllProxiesPenalizedError, EgressConnectError, UpstreamTimeoutError, UpstreamProtocolError) as exc:
                state = transition(state, self._attempt_event(exc, deadline, self._last_proxy_endpoint))
                if self._last_proxy_endpoint is not None and state.should_penalize:
                    srv.proxy_manager.penalize(self._last_proxy_endpoint, cfg.proxy_fail_penalty_seconds)
                if not state.next_pool:
                    raise
            else:
                self._finish_upstream_response(up, head, rest, replay)
        finally:
            close_quietly(up)
            lease.release()
        if state.next_pool:
            self._attempt(state.next_pool, False, deadline, False)

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
        srv, cfg = self.server, snapshot.server
        norm = [(k, v) for k, v in new_head.headers if k.lower() not in {"transfer-encoding", "content-length"}]
        norm.append(("Content-Length", str(len(new_body))))
        final_head = ResponseHead(new_head.version, new_head.status, new_head.reason, tuple(norm))
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

    def _finish_upstream_response(self, up: socket.socket, head: ResponseHead, rest: bytes, replay: bool) -> None:
        _, snapshot, runtime, req, route, *_ = self._ctx
        srv, cfg = self.server, snapshot.server
        adapter = runtime.registry.get(route.adapter)
        if adapter and adapter.wants_transform(route, head):
            body = bytearray()
            framing = framing_for_response(req.method, head.status, head.headers, body.extend)
            self._read_buffered_body(up, framing, rest, cfg.body_timeout)
            return self._send_transformed_response(*adapter.transform_response(route, head, bytes(body)))
        framing = framing_for_response(req.method, head.status, head.headers)
        initial, initial_fed = rest, False
        if head.status == 403 and replay:
            collector = BoundedCollector(DIAGNOSTIC_BODY_LIMIT)
            framing = framing_for_response(req.method, head.status, head.headers, collector.add)
            initial, initial_fed = self._read_diagnostic(up, framing, collector, rest, cfg.body_timeout), True
            result = runtime.classifier.classify(403, bytes(collector.data), head.get("content-encoding"))
            srv.metrics.incr("classified_403_region" if result.is_region_error else "classified_403_plain")
            if result.is_region_error:
                raise _ReplayRequested
            if collector.truncated or (collector.full and not framing.complete):
                srv.metrics.incr("classified_403_truncated")
                LOGGER.warning("403 body exceeds %d bytes: signatures beyond it are not checked", DIAGNOSTIC_BODY_LIMIT)
        self._relay(up, head, framing, initial, initial_fed)

    def _connect_upstream(
        self, pool: PoolConfig, up: UpstreamConfig, dl: EgressDeadline, single: bool = False
    ) -> socket.socket:
        _, snapshot, *_ = self._ctx
        srv, cfg = self.server, snapshot.server
        budget = srv.retry_budget
        base = EgressSettings.from_server(cfg)
        req_connect = min(base.connect_timeout, budget.connect_timeout) if single else base.connect_timeout
        try:
            max_retries = budget.single_pool_max_retries if single else base.single_pool_max_retries
            ct, td = dl.clamp(req_connect), dl.clamp(cfg.total_egress_deadline)
            settings = replace(base, connect_timeout=ct, total_deadline=td, single_pool_max_retries=max_retries)
        except TimeoutError:
            raise EgressTimeoutError(f"total egress deadline of {cfg.total_egress_deadline:g}s exceeded") from None

        def wrap(raw: socket.socket) -> socket.socket:
            return srv.ssl_context.wrap_socket(raw, server_hostname=up.host)

        def on_attempt(ep: ProxyEndpoint | None) -> None:
            # Только authority: ни URL, ни userinfo не сохраняются в состоянии запроса.
            self._last_proxy_endpoint, self.proxy_endpoint = ep, "-"
            if ep is not None:
                self.proxy_endpoint = f"[{ep.host}]:{ep.port}" if ":" in ep.host else f"{ep.host}:{ep.port}"

        return srv.proxy_manager._connect(pool, up.host, up.port, settings, wrap if up.use_tls else None, on_attempt)

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
        sock, snapshot, *_ = self._ctx
        cfg = snapshot.server
        self.status, self.committed = head.status, True
        try:
            sock.settimeout(cfg.body_timeout)
            sock.sendall(build_client_head(head))
        except OSError:
            self.outcome = "client_cancelled"
            self.server.metrics.incr("client_cancelled")
            return
        relay = SSERelay(inactivity_timeout=cfg.inactivity_timeout, write_timeout=cfg.body_timeout)
        result = relay.relay(up, sock, framing, initial=initial, initial_fed=fed)
        self.outcome, self.bytes_sent = result.outcome, result.bytes_sent
        self.server.metrics.incr(f"status_{head.status}")
        self.server.metrics.incr(f"relay_{result.outcome}")
        if not result.ok:
            self.server.metrics.incr("aborted_streams")
