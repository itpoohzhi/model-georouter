"""Opt-in маршрутизация Claude Code через loopback-sidecar с fail-open до первого коммита."""

from __future__ import annotations

import secrets
import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .adapters import AdapterRoute
from .config import BridgeConfig, UpstreamConfig
from .errors import EgressConnectError, SlotsExhaustedError, UpstreamProtocolError, UpstreamTimeoutError
from .http_wire import RequestHead, ResponseHead, framing_for_response
from .model_router import BodyInspector
from .relay import close_quietly
from .retry_budget import EgressDeadline

if TYPE_CHECKING:
    from .bridge_server import _Runtime

_CATALOG_ROUTE_PATH = "/v1internal:fetchAvailableModels"
_CATALOG_CALLBACK_PATH = "/cloudcode" + _CATALOG_ROUTE_PATH
_CATALOG_TICKET_HEADER = "X-Model-Georouter-Catalog-Ticket"
_CATALOG_CALLBACK_HEADER = "X-Model-Georouter-Catalog-Callback-Path"
_INTERNAL_HEADER_PREFIX = "x-model-georouter-"
_SIDECAR_STRIPPED_HEADERS = frozenset({"authorization", "proxy-authorization", "cookie", "x-api-key", "x-goog-api-key"})


@dataclass(frozen=True)
class _CatalogGrant:
    """Одноразовый контекст catalog callback; Google Authorization остаётся только в памяти роутера."""

    snapshot: BridgeConfig
    runtime: _Runtime
    req: RequestHead
    route: AdapterRoute
    query: str
    upstream: UpstreamConfig
    inspector: BodyInspector
    pool_name: str
    expected_route_path: str
    expires_at: float


class CatalogTicketStore:
    """Потокобезопасное хранилище одноразовых catalog-ticket с TTL 10 секунд."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._tickets: dict[str, _CatalogGrant] = {}
        self._lock = threading.Lock()

    def issue(
        self,
        snapshot: BridgeConfig,
        runtime: _Runtime,
        req: RequestHead,
        route: AdapterRoute,
        query: str,
        upstream: UpstreamConfig,
        inspector: BodyInspector,
        pool_name: str,
    ) -> str:
        """Сохранить исходный catalog-запрос под случайным single-use ticket."""
        now = self._clock()
        token = secrets.token_urlsafe(32)
        grant = _CatalogGrant(
            snapshot,
            runtime,
            req,
            route,
            query,
            upstream,
            inspector,
            pool_name,
            _CATALOG_ROUTE_PATH,
            now + 10.0,
        )
        with self._lock:
            expired = [key for key, item in self._tickets.items() if item.expires_at <= now]
            for key in expired:
                self._tickets.pop(key, None)
            self._tickets[token] = grant
        return token

    def exists(self, token: str) -> bool:
        with self._lock:
            return token in self._tickets

    def consume(self, token: str) -> _CatalogGrant | None:
        """Атомарно изъять ticket: повтор и параллельный callback никогда не получают контекст."""
        with self._lock:
            grant = self._tickets.pop(token, None)
        if grant is None or grant.expires_at <= self._clock():
            return None
        return grant


class ClaudeCodeRouting:
    """Mixin HTTP-обработчика: exact-match hook, sanitize, callback и pre-commit fallback."""

    def _catalog_callback_ticket(self, req: RequestHead, snapshot: BridgeConfig) -> str | None:
        token = req.get(_CATALOG_TICKET_HEADER)
        if token is not None and (
            snapshot.cloudcode_claude_code.enabled or self._bridge._catalog_tickets.exists(token)
        ):
            return token
        return None

    @staticmethod
    def _sidecar_matches(snapshot: BridgeConfig, route: AdapterRoute, inspector: BodyInspector) -> bool:
        claude = snapshot.cloudcode_claude_code
        if not claude.enabled or route.adapter != "cloudcode" or route.path not in claude.intercept_paths:
            return False
        return route.path == _CATALOG_ROUTE_PATH or inspector.model in claude.intercept_models

    @staticmethod
    def _loopback_peer(client_address: object) -> bool:
        return isinstance(client_address, tuple) and bool(client_address) and client_address[0] in ("127.0.0.1", "::1")

    def _serve_catalog_callback(self, sock: socket.socket, route: AdapterRoute, token: str) -> None:
        """Вернуть sidecar исходный Google catalog без рекурсии и без выдачи ему Authorization."""
        srv = self._bridge
        grant = srv._catalog_tickets.consume(token)
        if (
            grant is None
            or not self._loopback_peer(self.client_address)
            or route.adapter != "cloudcode"
            or route.path != grant.expected_route_path
        ):
            self._respond_json(
                sock,
                403,
                {
                    "error": {
                        "type": "catalog_callback_forbidden",
                        "message": "catalog callback rejected",
                        "retryable": False,
                    }
                },
            )
            return
        self.model = grant.inspector.model
        self._ctx = (
            sock,
            grant.snapshot,
            grant.runtime,
            grant.req,
            grant.route,
            grant.query,
            grant.upstream,
            grant.inspector,
        )
        deadline = EgressDeadline.from_timeout(grant.snapshot.server.total_egress_deadline)
        with srv.admission.admit(grant.pool_name, self.path, deadline=deadline, is_stream=False) as ticket:
            self._attempt(grant.pool_name, "", ticket.deadline, allow_fallback=False)

    def _sidecar_fallback_allowed(self, snapshot: BridgeConfig, route: AdapterRoute, inspector: BodyInspector) -> bool:
        claude = snapshot.cloudcode_claude_code
        return claude.pre_commit_fallback and (
            route.path == _CATALOG_ROUTE_PATH or inspector.model in claude.google_fallback_models
        )

    def _sidecar_status_requires_fallback(
        self, head: ResponseHead, snapshot: BridgeConfig, route: AdapterRoute, inspector: BodyInspector
    ) -> bool:
        if not self._sidecar_fallback_allowed(snapshot, route, inspector):
            return False
        if route.path == _CATALOG_ROUTE_PATH:
            return not 200 <= head.status < 300
        if head.status in (429, 502, 503, 504):
            return True
        return head.status in (400, 422, 501) and head.get("X-Sidecar-Pre-Dispatch") == "1"

    def _attempt_claude_sidecar(self, original_pool: str) -> bool:
        """Сделать один pre-upstream attempt; `False` означает ровно один возврат на исходный Google route."""
        original = self._ctx
        sock, snapshot, runtime, req, route, query, _upstream, inspector = original
        srv, cfg, claude = self._bridge, snapshot.server, snapshot.cloudcode_claude_code
        sidecar_upstream = snapshot.upstreams[claude.sidecar_upstream]
        headers = [
            (key, value)
            for key, value in req.headers
            if key.lower() not in _SIDECAR_STRIPPED_HEADERS and not key.lower().startswith(_INTERNAL_HEADER_PREFIX)
        ]
        headers.append(("X-Model-Georouter-Request-Id", secrets.token_hex(16)))
        if route.path == _CATALOG_ROUTE_PATH:
            callback_req = RequestHead(
                req.method,
                req.target,
                req.version,
                [(key, value) for key, value in req.headers if not key.lower().startswith(_INTERNAL_HEADER_PREFIX)],
            )
            ticket = srv._catalog_tickets.issue(
                snapshot, runtime, callback_req, route, query, original[6], inspector, original_pool
            )
            headers.extend(((_CATALOG_CALLBACK_HEADER, _CATALOG_CALLBACK_PATH), (_CATALOG_TICKET_HEADER, ticket)))
        sidecar_req = RequestHead(req.method, req.target, req.version, headers)
        self._ctx = (sock, snapshot, runtime, sidecar_req, route, query, sidecar_upstream, inspector)
        self.pool_name, self.proxy_endpoint, self._last_proxy_endpoint = "direct", "-", None
        srv.metrics.incr("claude_sidecar_attempts")
        up: socket.socket | None = None
        lease = None
        try:
            lease = srv.slots.acquire("direct", cfg.direct_slots)
            if lease is None:
                raise SlotsExhaustedError("direct")
            deadline = EgressDeadline.from_timeout(claude.sidecar_header_timeout_seconds)
            try:
                up = self._connect_upstream(snapshot.pools["direct"], sidecar_upstream, deadline)
                self._send_upstream_request(up, cfg.body_timeout, deadline)
                head, rest = self._read_upstream_head(up, claude.sidecar_header_timeout_seconds)
            except (EgressConnectError, UpstreamTimeoutError, UpstreamProtocolError):
                srv.metrics.incr("claude_sidecar_precommit_errors")
                if self._sidecar_fallback_allowed(snapshot, route, inspector):
                    srv.metrics.incr("claude_sidecar_fallbacks")
                    if route.path == _CATALOG_ROUTE_PATH:
                        srv.metrics.incr("claude_sidecar_catalog_fallbacks")
                    return False
                raise
            if self._sidecar_status_requires_fallback(head, snapshot, route, inspector):
                srv.metrics.incr("claude_sidecar_precommit_errors")
                srv.metrics.incr("claude_sidecar_fallbacks")
                if route.path == _CATALOG_ROUTE_PATH:
                    srv.metrics.incr("claude_sidecar_catalog_fallbacks")
                return False
            framing = framing_for_response(req.method, head.status, head.headers)
            self._relay(up, head, framing, rest, False)
            return True
        finally:
            close_quietly(up)
            if lease is not None:
                lease.release()
            self._ctx = original
