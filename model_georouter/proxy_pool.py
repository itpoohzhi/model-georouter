"""Слой 3: egress-пулы — direct, HTTP CONNECT, SOCKS5 (удалённый DNS), штрафы и circuit breaker."""

from __future__ import annotations

import base64
import ipaddress
import random
import socket
import struct
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit

from .errors import AllProxiesPenalizedError, EgressConnectError, EgressTimeoutError
from .logging_utils import get_logger

if TYPE_CHECKING:
    from .config import PoolConfig, ServerConfig

logger = get_logger(__name__)

DEFAULT_PORTS = {"http": 8080, "socks5": 1080, "socks5h": 1080}
SOCKS_REPLIES = {
    1: "general SOCKS server failure",
    2: "connection not allowed by ruleset",
    3: "network unreachable",
    4: "host unreachable",
    5: "connection refused",
    6: "TTL expired",
    7: "command not supported",
    8: "address type not supported",
}
CONNECT_RESPONSE_LIMIT = 16384
DNS_MAX_THREADS = 32
_DNS_SEMAPHORE = threading.BoundedSemaphore(DNS_MAX_THREADS)


@dataclass(frozen=True)
class ProxyEndpoint:
    """Разобранный URL прокси; пароль не попадает в repr/label."""

    scheme: str
    host: str
    port: int
    username: str | None = None
    password: str | None = field(default=None, repr=False)
    raw: str = field(default="", repr=False)

    @property
    def label(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"


def parse_proxy_url(url: str) -> ProxyEndpoint:
    """Разобрать `http://user:pass@host:port` / `socks5[h]://host:port`; ValueError при ошибке."""
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in DEFAULT_PORTS:
        raise ValueError(f"unsupported proxy scheme {scheme!r}")
    if not parts.hostname:
        raise ValueError("proxy URL has no host")
    port = parts.port or DEFAULT_PORTS[scheme]
    username = unquote(parts.username) if parts.username is not None else None
    password = unquote(parts.password) if parts.password is not None else None
    return ProxyEndpoint(scheme, parts.hostname, port, username, password, url)


@dataclass(frozen=True)
class EgressSettings:
    """Параметры установления соединения — всё из конфигурации."""

    connect_timeout: float
    total_deadline: float
    retries: int
    penalty_seconds: float
    retry_sleep_seconds: float = 0.5
    retry_backoff_factor: float = 2.0
    retry_max_sleep_seconds: float = 3.0
    single_pool_max_retries: int | None = None

    @classmethod
    def from_server(cls, server: ServerConfig) -> EgressSettings:
        return cls(
            connect_timeout=server.connect_timeout,
            total_deadline=server.total_egress_deadline,
            retries=server.pre_send_retries,
            penalty_seconds=server.proxy_fail_penalty_seconds,
            retry_sleep_seconds=server.retry_sleep_seconds,
            retry_backoff_factor=server.retry_backoff_factor,
            retry_max_sleep_seconds=server.retry_max_sleep_seconds,
            single_pool_max_retries=server.single_pool_max_retries,
        )


# ───────────────────────────── транспорты ─────────────────────────────


def _arm(sock: socket.socket, deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("egress deadline exceeded")
    sock.settimeout(remaining)


def _recv_exact(sock: socket.socket, size: int, deadline: float, phase: str | None = None) -> bytes:
    buf = bytearray()
    while len(buf) < size:
        _arm(sock, deadline)
        chunk = sock.recv(size - len(buf))
        if not chunk:
            raise EgressConnectError("proxy closed the connection during handshake", phase=phase)
        buf += chunk
    return bytes(buf)


def _send_all(sock: socket.socket, data: bytes, deadline: float) -> None:
    _arm(sock, deadline)
    sock.sendall(data)


def _resolve(host: str, port: int, deadline: float) -> list[tuple]:
    """`getaddrinfo` в пределах бюджета: сам вызов таймаута не принимает, поэтому ждём его в потоке до `deadline`."""
    try:
        ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        pass
    else:
        return socket.getaddrinfo(host.strip("[]"), port, type=socket.SOCK_STREAM)  # литерал: DNS не нужен
    if not _DNS_SEMAPHORE.acquire(timeout=max(deadline - time.monotonic(), 0)):
        raise EgressTimeoutError("DNS resolver concurrency limit reached")
    outcome: list[list[tuple] | OSError] = []

    def work() -> None:
        try:
            outcome.append(socket.getaddrinfo(host, port, type=socket.SOCK_STREAM))
        except OSError as exc:
            outcome.append(exc)
        finally:
            _DNS_SEMAPHORE.release()

    resolver = threading.Thread(target=work, daemon=True)
    try:
        resolver.start()
    except BaseException:
        _DNS_SEMAPHORE.release()
        raise
    resolver.join(max(deadline - time.monotonic(), 0))
    if not outcome:
        raise TimeoutError("DNS resolution exceeded egress deadline")
    if isinstance(outcome[0], OSError):
        raise outcome[0]
    return outcome[0]


def _dial(host: str, port: int, deadline: float) -> socket.socket:
    last: OSError | None = None
    for family, kind, proto, _, address in _resolve(host, port, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("egress deadline exceeded")
        sock = socket.socket(family, kind, proto)
        try:
            sock.settimeout(remaining)
            sock.connect(address)
        except OSError as exc:
            sock.close()
            last = exc
            continue
        break
    else:
        raise last or OSError(f"no addresses resolved for {host}")
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass
    return sock


def _authority(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host and not host.startswith("[") else f"{host}:{port}"


def open_direct(host: str, port: int, deadline: float) -> socket.socket:
    """Прямое TCP-соединение без прокси."""
    return _dial(host, port, deadline)


def _read_connect_head(sock: socket.socket, endpoint: ProxyEndpoint, deadline: float) -> bytearray:
    """Прочитать голову ответа прокси до `\\r\\n\\r\\n` включительно, не забирая ни байта сверх неё.

    `MSG_PEEK` показывает, что уже пришло; без разделителя забираем подсмотренное и блокируемся
    на следующем peek до новых данных (без spin-wait), с разделителем — ровно до его конца.
    Начало туннельного потока за разделителем остаётся в сокете.
    """
    head = bytearray()
    while True:
        _arm(sock, deadline)
        peek_limit = min(4096, CONNECT_RESPONSE_LIMIT - len(head) + 4)
        peeked = sock.recv(peek_limit, socket.MSG_PEEK)
        if not peeked:
            raise EgressConnectError(f"proxy {endpoint.label} closed connection during CONNECT", phase="connect_read")
        prefix = bytes(head[-3:])  # разделитель может быть разорван между чтениями
        idx = (prefix + peeked).find(b"\r\n\r\n")
        found = idx != -1
        take = idx + 4 - len(prefix) if found else len(peeked)
        head += _recv_exact(sock, take, deadline, phase="connect_read")
        if len(head) > CONNECT_RESPONSE_LIMIT:
            raise EgressConnectError(f"proxy {endpoint.label} sent an oversized CONNECT response", phase="connect_read")
        if found:
            return head


def open_http_connect(endpoint: ProxyEndpoint, host: str, port: int, deadline: float) -> socket.socket:
    """HTTP CONNECT-туннель (с Basic Proxy-Authorization при наличии учётных данных)."""
    sock = _dial(endpoint.host, endpoint.port, deadline)
    try:
        authority = _authority(host, port)
        lines = [f"CONNECT {authority} HTTP/1.1", f"Host: {authority}"]
        if endpoint.username is not None:
            token = base64.b64encode(f"{endpoint.username}:{endpoint.password or ''}".encode()).decode("ascii")
            lines.append(f"Proxy-Authorization: Basic {token}")
        _send_all(sock, ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1"), deadline)
        head = _read_connect_head(sock, endpoint, deadline)
        status_line = bytes(head).split(b"\r\n", 1)[0].decode("latin-1")
        fields = status_line.split(None, 2)
        if len(fields) < 2 or not fields[1].isdigit():
            raise EgressConnectError(f"proxy {endpoint.label} sent a malformed CONNECT response", phase="connect_read")
        code = int(fields[1])
        if not 200 <= code < 300:
            raise EgressConnectError(f"proxy {endpoint.label} refused CONNECT: HTTP {code}", phase="connect_read")
        return sock
    except BaseException:
        sock.close()
        raise


def _socks_address(host: str, port: int, remote_dns: bool, deadline: float) -> bytes:
    """ATYP + адрес + порт; при `remote_dns` доменное имя всегда уходит прокси (ATYP 0x03)."""
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        ip = None
    if ip is None and not remote_dns:
        sockaddr = _resolve(host, port, deadline)[0][4]
        ip = ipaddress.ip_address(sockaddr[0])
    if ip is not None:
        atyp = 0x01 if ip.version == 4 else 0x04
        return bytes([atyp]) + ip.packed + struct.pack(">H", port)
    encoded = host.encode("idna")
    if not 0 < len(encoded) <= 255:
        raise EgressConnectError("target host name is not representable in SOCKS5")
    return bytes([0x03, len(encoded)]) + encoded + struct.pack(">H", port)


def open_socks5(
    endpoint: ProxyEndpoint, host: str, port: int, deadline: float, remote_dns: bool = True
) -> socket.socket:
    """SOCKS5 (RFC 1928, RFC 1929); имя хоста резолвит прокси (socks5h)."""
    sock = _dial(endpoint.host, endpoint.port, deadline)
    try:
        authenticated = endpoint.username is not None
        methods = [0x00, 0x02] if authenticated else [0x00]
        _send_all(sock, bytes([0x05, len(methods), *methods]), deadline)
        version, method = _recv_exact(sock, 2, deadline)
        if version != 0x05 or method == 0xFF:
            raise EgressConnectError(f"proxy {endpoint.label} rejected all SOCKS5 auth methods")
        if method == 0x02:
            user = (endpoint.username or "").encode()
            password = (endpoint.password or "").encode()
            if len(user) > 255 or len(password) > 255:
                raise EgressConnectError("SOCKS5 credentials are too long")
            _send_all(sock, bytes([0x01, len(user)]) + user + bytes([len(password)]) + password, deadline)
            _, status = _recv_exact(sock, 2, deadline)
            if status != 0x00:
                raise EgressConnectError(f"proxy {endpoint.label} rejected SOCKS5 credentials")
        elif method != 0x00:
            raise EgressConnectError(f"proxy {endpoint.label} chose unsupported SOCKS5 method {method:#x}")
        request = bytes([0x05, 0x01, 0x00]) + _socks_address(host, port, remote_dns, deadline)
        _send_all(sock, request, deadline)
        version, reply, _, atyp = _recv_exact(sock, 4, deadline)
        if version != 0x05:
            raise EgressConnectError(f"proxy {endpoint.label} sent a malformed SOCKS5 reply")
        if reply != 0x00:
            reason = SOCKS_REPLIES.get(reply, f"code {reply}")
            raise EgressConnectError(f"proxy {endpoint.label} failed SOCKS5 CONNECT: {reason}")
        if atyp == 0x01:
            _recv_exact(sock, 4 + 2, deadline)
        elif atyp == 0x04:
            _recv_exact(sock, 16 + 2, deadline)
        elif atyp == 0x03:
            (length,) = _recv_exact(sock, 1, deadline)
            _recv_exact(sock, length + 2, deadline)
        else:
            raise EgressConnectError(f"proxy {endpoint.label} sent an unknown SOCKS5 address type")
        return sock
    except BaseException:
        sock.close()
        raise


# ───────────────────────────── менеджер пулов ─────────────────────────────


class ProxyPoolManager:
    """Выбор прокси, штрафы и circuit breaker; состояние штрафов переживает hot-reload конфигурации."""

    def __init__(self, *, clock: Callable[[], float] = time.time, rng: random.Random | None = None):
        self._clock = clock
        self._rng = rng or random.Random()
        self._penalties: dict[str, float] = {}
        self._endpoints: dict[str, ProxyEndpoint] = {}
        self._lock = threading.Lock()

    def endpoints(self, pool: PoolConfig) -> list[ProxyEndpoint]:
        with self._lock:
            result = []
            for url in pool.proxies:
                endpoint = self._endpoints.get(url)
                if endpoint is None:
                    endpoint = self._endpoints[url] = parse_proxy_url(url)
                result.append(endpoint)
            return result

    def penalize(self, endpoint: ProxyEndpoint, seconds: float) -> None:
        """`penalty_until = time.time() + penalty_seconds`."""
        with self._lock:
            self._penalties[endpoint.raw] = self._clock() + seconds

    def is_penalized(self, endpoint: ProxyEndpoint) -> bool:
        with self._lock:
            return self._penalties.get(endpoint.raw, 0.0) > self._clock()

    def available(self, pool: PoolConfig, exclude: Iterable[str] = ()) -> list[ProxyEndpoint]:
        excluded = set(exclude)
        now = self._clock()
        endpoints = self.endpoints(pool)
        with self._lock:
            return [e for e in endpoints if self._penalties.get(e.raw, 0.0) <= now and e.raw not in excluded]

    def ensure_available(self, pool: PoolConfig) -> None:
        """Circuit breaker: все прокси в штрафе — мгновенный отказ без удержания слота."""
        if pool.type != "direct" and not self.available(pool):
            raise AllProxiesPenalizedError(pool.name)

    def select(self, pool: PoolConfig, exclude: Iterable[str] = ()) -> ProxyEndpoint | None:
        """Выбрать прокси по стратегии пула; None — нет кандидатов (или пул direct)."""
        if pool.type == "direct":
            return None
        candidates = self.available(pool, exclude)
        if not candidates:
            return None
        if pool.strategy == "random":
            return self._rng.choice(candidates)
        return candidates[0]

    def stats(self, pools: Iterable[PoolConfig]) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for pool in pools:
            if pool.type == "direct":
                result[pool.name] = {"type": pool.type, "proxies_total": 0, "proxies_penalized": 0}
                continue
            total = len(pool.proxies)
            result[pool.name] = {
                "type": pool.type,
                "proxies_total": total,
                "proxies_penalized": total - len(self.available(pool)),
            }
        return result

    def _open(
        self, pool: PoolConfig, endpoint: ProxyEndpoint | None, host: str, port: int, deadline: float
    ) -> socket.socket:
        if endpoint is None:
            return open_direct(host, port, deadline)
        if pool.type == "http_connect":
            return open_http_connect(endpoint, host, port, deadline)
        remote_dns = pool.remote_dns or endpoint.scheme == "socks5h"
        return open_socks5(endpoint, host, port, deadline, remote_dns)

    @staticmethod
    def _normalize(exc: Exception, endpoint: ProxyEndpoint | None, host: str, port: int) -> EgressConnectError:
        if isinstance(exc, EgressConnectError):
            return exc
        via = f" via {endpoint.label}" if endpoint else ""
        if isinstance(exc, TimeoutError):
            return EgressTimeoutError(f"connect to {host}:{port}{via} timed out")
        return EgressConnectError(f"cannot connect to {host}:{port}{via}: {type(exc).__name__}")

    @staticmethod
    def _backoff_pause(pool: PoolConfig, settings: EgressSettings, attempt: int, single: bool, overall: float) -> None:
        """Экспоненциальная пауза перед повтором к той же цели (direct или единственный прокси).

        Пул из нескольких прокси не спит: непробованный кандидат берётся мгновенно, а без
        кандидатов цикл завершается — пауза перед выходом бессмысленна. После последней
        попытки и когда пауза не укладывается в остаток общего дедлайна — тоже без сна.
        """
        if attempt >= settings.retries or (pool.type != "direct" and not single):
            return
        retry_number = attempt + 1  # attempt считается с 0: после первой неудачи идёт повтор №1
        exponent = retry_number - 1
        try:
            raw_pause = settings.retry_sleep_seconds * (settings.retry_backoff_factor**exponent)
        except OverflowError:  # float ** int сверх ~1.8e308 — пауза упирается в потолок
            raw_pause = settings.retry_max_sleep_seconds
        pause = min(raw_pause, settings.retry_max_sleep_seconds)
        if pause > 0 and overall - time.monotonic() > pause:
            time.sleep(pause)

    def connect(
        self,
        pool: PoolConfig,
        host: str,
        port: int,
        settings: EgressSettings,
        wrap: Callable[[socket.socket], socket.socket] | None = None,
    ) -> socket.socket:
        """Открыть соединение к host:port через пул: ретраи до отправки, штраф упавших прокси, общий дедлайн."""
        return self._connect(pool, host, port, settings, wrap)

    def _connect(
        self,
        pool: PoolConfig,
        host: str,
        port: int,
        settings: EgressSettings,
        wrap: Callable[[socket.socket], socket.socket] | None = None,
        on_attempt: Callable[[ProxyEndpoint | None], None] | None = None,
    ) -> socket.socket:
        """Внутренний вариант connect с наблюдением за фактически начатой попыткой."""
        overall = time.monotonic() + settings.total_deadline
        tried: set[str] = set()
        last: EgressConnectError | None = None
        single = len(pool.proxies) == 1  # единственную прокси перебирать нечем: ретраи идут в неё же
        if single and settings.single_pool_max_retries is not None:
            settings = replace(settings, retries=min(settings.retries, settings.single_pool_max_retries))
        for attempt in range(settings.retries + 1):
            endpoint = None
            if pool.type != "direct":
                endpoint = self.select(pool, set() if single else tried)
                if endpoint is None:
                    if last is None:
                        raise AllProxiesPenalizedError(pool.name)
                    break
                tried.add(endpoint.raw)
            remaining = overall - time.monotonic()
            if remaining <= 0:
                last = EgressTimeoutError(f"total egress deadline of {settings.total_deadline:g}s exceeded")
                break
            attempt_start = time.monotonic()
            attempt_deadline = attempt_start + min(settings.connect_timeout, remaining)
            target = endpoint.label if endpoint is not None else f"{host}:{port}"  # без учётных данных прокси
            penalize = endpoint is not None and (not single or attempt == settings.retries)
            if on_attempt is not None:
                on_attempt(endpoint)
            try:
                sock = self._open(pool, endpoint, host, port, attempt_deadline)
            except (OSError, EgressConnectError) as exc:
                last = self._normalize(exc, endpoint, host, port)
                err_phase = last.phase or "connect"
                if endpoint is not None and penalize:
                    self.penalize(endpoint, settings.penalty_seconds)
                logger.warning(
                    "proxy %s failed during %s: %s (penalized=%s, elapsed=%.1fms)",
                    target,
                    err_phase,
                    type(exc).__name__,
                    penalize,
                    (time.monotonic() - attempt_start) * 1000,
                )
                self._backoff_pause(pool, settings, attempt, single, overall)
                continue
            if wrap is None:
                sock.settimeout(None)
                return sock
            if time.monotonic() >= attempt_deadline:
                # бюджет попытки ушёл на установление туннеля: прокси не виноват, штрафа нет
                sock.close()
                last = EgressTimeoutError(
                    f"connect timeout of {settings.connect_timeout:g}s exceeded before TLS handshake",
                    phase="tls_handshake",
                )
                logger.warning("connect timeout before TLS for %s (penalized=False)", target)
                self._backoff_pause(pool, settings, attempt, single, overall)
                continue
            try:
                _arm(sock, attempt_deadline)
                wrapped = wrap(sock)
                wrapped.settimeout(None)
                return wrapped
            except (OSError, ValueError) as exc:
                sock.close()
                last = EgressConnectError(
                    f"TLS handshake with {host} failed: {type(exc).__name__}", phase="tls_handshake"
                )
                if endpoint is not None and penalize:
                    self.penalize(endpoint, settings.penalty_seconds)
                logger.warning(
                    "proxy %s failed during tls_handshake: %s (penalized=%s, elapsed=%.1fms)",
                    target,
                    type(exc).__name__,
                    penalize,
                    (time.monotonic() - attempt_start) * 1000,
                )
                self._backoff_pause(pool, settings, attempt, single, overall)
        raise last or EgressConnectError(f"cannot connect to {host}:{port}")
