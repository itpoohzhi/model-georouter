"""Слой 1: HTTP/1.1-сервер моста на 127.0.0.1 — приём, инспекция, маршрутизация, replay и relay."""

from __future__ import annotations

import select
import socket
import socketserver
import ssl
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

from . import SERVICE_NAME, __version__
from .adapters import AdapterRegistry, AdapterRoute, AdminAdapter
from .config import BridgeConfig, ConfigManager, PoolConfig, UpstreamConfig
from .error_handler import ErrorHandler, json_response_bytes
from .errors import (
    BadRequestError,
    BodyTooLargeError,
    BridgeError,
    ClientTimeoutError,
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
    header_get,
    parse_request_head,
    parse_response_head,
    read_head_block,
)
from .logging_utils import get_logger
from .model_router import BodyInspector, ModelRouter, RegionErrorClassifier, model_from_path
from .proxy_pool import EgressSettings, ProxyPoolManager
from .relay import RELAY_CHUNK, SSERelay, close_quietly, configure_socket

LOGGER = get_logger("universal_ai_bridge.server")

DIAGNOSTIC_BODY_LIMIT = 16384
DRAIN_MAX_SECONDS = 2.0
DRAIN_MAX_BYTES = 64 * 1024 * 1024
SKIPPED_REQUEST_HEADERS = frozenset({"host", "accept-encoding", "content-length", "expect"})


# ───────────────────────────── слоты и метрики ─────────────────────────────


class SlotPool:
    """Неблокирующий счётчик слотов с изменяемым лимитом (hot-reload)."""

    def __init__(self, limit: int):
        self._limit = limit
        self._in_use = 0
        self._lock = threading.Lock()

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
        self._limit = limit
        self.data = bytearray()

    @property
    def full(self) -> bool:
        return len(self.data) >= self._limit

    def add(self, chunk: bytes) -> None:
        room = self._limit - len(self.data)
        if room > 0:
            self.data += chunk[:room]


@dataclass(frozen=True)
class _Runtime:
    """Производные от снимка конфигурации объекты (пересоздаются при смене снимка)."""

    registry: AdapterRegistry
    router: ModelRouter
    classifier: RegionErrorClassifier


class _ReplayRequested(Exception):
    """Внутренний сигнал: прямой маршрут получил подтверждённый RegionError."""


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
        snapshot = config_manager.get()
        self.config_manager = config_manager
        self.proxy_manager = proxy_manager or ProxyPoolManager()
        self.geo_cache = geo_cache or GeoCache(
            snapshot.server.geo_cache_file or None, snapshot.server.geo_cache_ttl_seconds
        )
        self.slots = SplitSlots(snapshot.server.direct_slots, snapshot.server.proxy_slots)
        self.metrics = Metrics()
        self.error_handler = ErrorHandler()
        self.admin = AdminAdapter()
        self._ssl_context = ssl_context
        self._runtime: tuple[BridgeConfig, _Runtime] | None = None
        self.started_at = time.monotonic()
        super().__init__((snapshot.server.listen, snapshot.server.port), _Handler)

    # --- служебное ---

    @property
    def ssl_context(self) -> ssl.SSLContext:
        if self._ssl_context is None:
            self._ssl_context = ssl.create_default_context()
        return self._ssl_context

    def runtime_for(self, snapshot: BridgeConfig) -> _Runtime:
        cached = self._runtime
        if cached is not None and cached[0] is snapshot:
            return cached[1]
        runtime = _Runtime(
            AdapterRegistry.from_config(snapshot),
            ModelRouter.from_config(snapshot),
            RegionErrorClassifier(snapshot.server.geo_error_signatures),
        )
        self._runtime = (snapshot, runtime)
        return runtime

    def handle_error(self, request, client_address) -> None:  # noqa: ARG002
        LOGGER.exception("unhandled error in connection handler")

    def start_background(self) -> threading.Thread:
        thread = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        thread.start()
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

    def setup(self) -> None:
        self.committed = False
        self.body_done = False
        self.started = time.monotonic()
        self.method = "-"
        self.path = "-"
        self.model: str | None = None
        self.pool_name = "-"
        self.status = 0
        self.outcome = "-"
        self.replayed = False
        self.bytes_sent = 0

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

    def _respond_json(self, sock: socket.socket, status: int, payload: dict, headers: dict | None = None) -> None:
        self.status = status
        self.outcome = "local"
        self.server.metrics.incr(f"status_{status}")
        self._send_bytes(sock, json_response_bytes(status, payload, headers))

    def _fail(self, sock: socket.socket, exc: BaseException) -> None:
        srv = self.server
        if self.committed:
            srv.metrics.incr("aborted_streams")
            self.outcome = self.outcome if self.outcome != "-" else "aborted"
            if not isinstance(exc, (BridgeError, OSError)):
                LOGGER.error("stream aborted by internal error: %s", type(exc).__name__)
            return
        if isinstance(exc, (ConnectionError, BrokenPipeError)) or isinstance(exc, KeyboardInterrupt):
            self.outcome = "client_disconnected"
            return
        if not isinstance(exc, BridgeError):
            LOGGER.error("internal error: %s", type(exc).__name__, exc_info=True)
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
            "request method=%s path=%s model=%s pool=%s status=%s outcome=%s bytes=%d replayed=%s dur_ms=%d",
            self.method,
            self.path,
            self.model or "-",
            self.pool_name,
            self.status or "-",
            self.outcome,
            self.bytes_sent,
            self.replayed,
            int((time.monotonic() - self.started) * 1000),
        )

    # --- основной поток ---

    def _serve(self, sock: socket.socket) -> None:
        srv = self.server
        snapshot = srv.config_manager.get()
        cfg = snapshot.server
        try:
            block, rest = read_head_block(sock, timeout=cfg.body_timeout)
        except HeadError as exc:
            if exc.kind == "eof":
                self.body_done = True
                self.outcome = "client_closed"
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
            self._respond_json(sock, status, payload, headers)
            return
        route = runtime.registry.resolve(path)
        if route is None:
            raise NotFoundError(f"no adapter serves path {path}")

        inspector = self._read_request_body(sock, req, rest, cfg.body_buffer_max_bytes, cfg.body_timeout)
        self.model = inspector.model or model_from_path(path)
        decision = runtime.router.decide(self.model, route.upstream, srv.geo_cache)
        if decision.source == "geo_cache":
            srv.metrics.incr("geo_cache_hits")
        upstream = snapshot.upstreams[route.upstream]
        fallback = snapshot.geo_fallback_pool
        first_pool = snapshot.pools[decision.pool]
        replay_allowed = bool(fallback) and fallback != decision.pool and _is_direct(first_pool)
        try:
            self._attempt(sock, snapshot, runtime, req, route, query, upstream, inspector, decision.pool, replay_allowed)
        except _ReplayRequested:
            srv.geo_cache.mark_blocked(route.upstream, self.model, cfg.geo_cache_ttl_seconds)
            srv.metrics.incr("replays")
            self.replayed = True
            self._attempt(sock, snapshot, runtime, req, route, query, upstream, inspector, fallback, False)

    def _read_request_body(
        self, sock: socket.socket, req: RequestHead, rest: bytes, limit: int, timeout: float
    ) -> BodyInspector:
        inspector = BodyInspector(limit)
        framing = framing_for_request(req.headers, inspector.feed, limit)
        if not framing.complete and (req.get("expect") or "").lower() == "100-continue":
            self._send_bytes(sock, b"HTTP/1.1 100 Continue\r\n\r\n")
        try:
            if rest:
                framing.feed(rest)
            while not framing.complete:
                sock.settimeout(timeout)
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

    def _attempt(
        self,
        sock: socket.socket,
        snapshot: BridgeConfig,
        runtime: _Runtime,
        req: RequestHead,
        route: AdapterRoute,
        query: str,
        upstream: UpstreamConfig,
        inspector: BodyInspector,
        pool_name: str,
        replay_allowed: bool,
    ) -> None:
        srv = self.server
        cfg = snapshot.server
        pool = snapshot.pools[pool_name]
        self.pool_name = pool_name
        direct = _is_direct(pool)
        if not direct:
            srv.proxy_manager.ensure_available(pool)  # circuit breaker: до слота, за доли миллисекунды
        lease = srv.slots.acquire("direct" if direct else "proxy", cfg.direct_slots if direct else cfg.proxy_slots)
        if lease is None:
            raise SlotsExhaustedError("direct" if direct else "proxy")
        up: socket.socket | None = None
        try:
            up = self._connect_upstream(snapshot, pool, upstream)
            self._send_upstream_request(up, req, route, query, upstream, inspector, cfg.body_timeout)
            head, rest = self._read_upstream_head(up, cfg.headers_timeout)
            framing = framing_for_response(req.method, head.status, head.headers)
            initial, initial_fed = rest, False
            if head.status == 403 and replay_allowed:
                collector = BoundedCollector(DIAGNOSTIC_BODY_LIMIT)
                framing = framing_for_response(req.method, head.status, head.headers, collector.add)
                initial = self._read_diagnostic(up, framing, collector, rest, cfg.body_timeout)
                initial_fed = True
                result = runtime.classifier.classify(403, bytes(collector.data), head.get("content-encoding"))
                srv.metrics.incr("classified_403_region" if result.is_region_error else "classified_403_plain")
                if result.is_region_error:
                    raise _ReplayRequested
            self._relay(sock, up, head, framing, initial, initial_fed, cfg.inactivity_timeout, cfg.body_timeout)
        finally:
            close_quietly(up)
            lease.release()

    def _connect_upstream(self, snapshot: BridgeConfig, pool: PoolConfig, upstream: UpstreamConfig) -> socket.socket:
        srv = self.server
        wrap = None
        if upstream.use_tls:
            context = srv.ssl_context

            def wrap(raw: socket.socket) -> socket.socket:
                return context.wrap_socket(raw, server_hostname=upstream.host)

        settings = EgressSettings.from_server(snapshot.server)
        return srv.proxy_manager.connect(pool, upstream.host, upstream.port, settings, wrap)

    @staticmethod
    def _send_upstream_request(
        up: socket.socket,
        req: RequestHead,
        route: AdapterRoute,
        query: str,
        upstream: UpstreamConfig,
        inspector: BodyInspector,
        timeout: float,
    ) -> None:
        body = inspector.view()
        try:
            up.settimeout(timeout)
            up.sendall(build_upstream_request(req, route.path, query, upstream, len(body)))
            if len(body):
                up.sendall(body)
        except OSError as exc:
            raise UpstreamProtocolError(f"failed to send request to upstream: {type(exc).__name__}") from exc
        finally:
            body.release()

    @staticmethod
    def _read_upstream_head(up: socket.socket, timeout: float) -> tuple[ResponseHead, bytes]:
        initial = b""
        while True:
            try:
                block, rest = read_head_block(up, timeout=timeout, initial=initial)
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
        except OSError:  # включая таймаут: классифицируем то, что успели прочитать
            pass
        return bytes(wire)

    def _relay(
        self,
        sock: socket.socket,
        up: socket.socket,
        head: ResponseHead,
        framing: Framing,
        initial: bytes,
        initial_fed: bool,
        inactivity_timeout: float,
        write_timeout: float,
    ) -> None:
        self.status = head.status
        self.committed = True
        try:
            sock.settimeout(write_timeout)
            sock.sendall(build_client_head(head))
        except OSError:
            self.outcome = "client_cancelled"
            self.server.metrics.incr("client_cancelled")
            return
        relay = SSERelay(inactivity_timeout=inactivity_timeout, write_timeout=write_timeout)
        result = relay.relay(up, sock, framing, initial=initial, initial_fed=initial_fed)
        self.outcome = result.outcome
        self.bytes_sent = result.bytes_sent
        self.server.metrics.incr(f"status_{head.status}")
        self.server.metrics.incr(f"relay_{result.outcome}")
        if not result.ok:
            self.server.metrics.incr("aborted_streams")
