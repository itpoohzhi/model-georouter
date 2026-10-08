"""Тестовая инфраструктура: фейковые upstream/прокси на сокетах и сырой HTTP-клиент."""

from __future__ import annotations

import base64
import select
import socket
import struct
import threading
import time
from dataclasses import dataclass, field

from universal_ai_bridge.bridge_server import BridgeServer
from universal_ai_bridge.config import ConfigManager, parse_config
from universal_ai_bridge.proxy_pool import ProxyPoolManager

LOCALHOST = "127.0.0.1"


def free_port() -> int:
    """Порт, на котором гарантированно никто не слушает (connect → ECONNREFUSED)."""
    with socket.socket() as sock:
        sock.bind((LOCALHOST, 0))
        return sock.getsockname()[1]


def recv_head(conn: socket.socket, limit: int = 65536) -> tuple[bytes, bytes]:
    buf = b""
    conn.settimeout(5)
    while b"\r\n\r\n" not in buf:
        data = conn.recv(65536)
        if not data:
            return buf, b""
        buf += data
        if len(buf) > limit:
            break
    head, _, rest = buf.partition(b"\r\n\r\n")
    return head + b"\r\n\r\n", rest


def pipe(a: socket.socket, b: socket.socket, stop: threading.Event | None = None) -> None:
    """Двунаправленная перекачка до закрытия любой стороны."""
    try:
        while not (stop and stop.is_set()):
            readable, _, _ = select.select([a, b], [], [], 0.2)
            for sock in readable:
                data = sock.recv(65536)
                if not data:
                    return
                (b if sock is a else a).sendall(data)
    except (OSError, ValueError):
        return
    finally:
        for sock in (a, b):
            try:
                sock.close()
            except OSError:
                pass


class FakeServer:
    """TCP-сервер: каждое соединение обслуживает `handler(conn)` в своём потоке."""

    def __init__(self, handler):
        self._handler = handler
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((LOCALHOST, 0))
        self._sock.listen(64)
        self._sock.settimeout(0.1)
        self.port = self._sock.getsockname()[1]
        self._closed = threading.Event()
        self.connections: list[socket.socket] = []
        self.stop = threading.Event()
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self) -> None:
        while not self._closed.is_set():
            try:
                conn, _ = self._sock.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            self.connections.append(conn)
            threading.Thread(target=self._run, args=(conn,), daemon=True).start()

    def _run(self, conn: socket.socket) -> None:
        try:
            self._handler(conn)
        except (OSError, ValueError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def close(self) -> None:
        self._closed.set()
        self.stop.set()
        try:
            self._sock.close()
        except OSError:
            pass
        for conn in self.connections:
            try:
                conn.close()
            except OSError:
                pass


@dataclass
class Req:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes


def read_http_request(conn: socket.socket) -> Req | None:
    head, rest = recv_head(conn)
    if not head.strip():
        return None
    lines = head.decode("latin-1").split("\r\n")
    method, path, _ = lines[0].split(" ", 2)
    headers = {}
    for line in lines[1:]:
        if ":" in line:
            key, _, value = line.partition(":")
            headers[key.strip().lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    body = rest
    while len(body) < length:
        data = conn.recv(65536)
        if not data:
            break
        body += data
    return Req(method, path, headers, body[:length])


class HttpUpstream(FakeServer):
    """Фейковый upstream: читает запрос, записывает его и вызывает `responder(conn, req)`."""

    def __init__(self, responder):
        self.requests: list[Req] = []
        self.responder = responder
        super().__init__(self._serve)

    def _serve(self, conn: socket.socket) -> None:
        req = read_http_request(conn)
        if req is None:
            return
        self.requests.append(req)
        self.responder(conn, req)


def respond(conn: socket.socket, status: int = 200, body: bytes = b"", headers: dict | None = None) -> None:
    lines = [f"HTTP/1.1 {status} X", f"Content-Length: {len(body)}", "Connection: close"]
    lines.extend(f"{k}: {v}" for k, v in (headers or {}).items())
    conn.sendall(("\r\n".join(lines) + "\r\n\r\n").encode() + body)


def chunk(data: bytes) -> bytes:
    return b"%x\r\n" % len(data) + data + b"\r\n"


def send_sse_head(conn: socket.socket) -> None:
    conn.sendall(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
    )


class FakeConnectProxy(FakeServer):
    """HTTP CONNECT-прокси; `target` подменяет адрес туннеля (чтобы вести на фейковый upstream)."""

    def __init__(self, target: tuple[str, int] | None = None, status: int = 200):
        self.target = target
        self.status = status
        self.connects: list[dict] = []
        super().__init__(self._serve)

    def _serve(self, conn: socket.socket) -> None:
        head, rest = recv_head(conn)
        lines = head.decode("latin-1").split("\r\n")
        authority = lines[0].split(" ")[1]
        headers = {k.strip().lower(): v.strip() for k, _, v in (line.partition(":") for line in lines[1:] if line)}
        self.connects.append({"authority": authority, "headers": headers})
        if self.status != 200:
            conn.sendall(f"HTTP/1.1 {self.status} Denied\r\nContent-Length: 0\r\n\r\n".encode())
            return
        host, _, port = authority.rpartition(":")
        remote = socket.create_connection(self.target or (host, int(port)), timeout=5)
        conn.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
        if rest:
            remote.sendall(rest)
        pipe(conn, remote, self.stop)


class FakeSocks5Proxy(FakeServer):
    """SOCKS5-прокси; фиксирует ATYP/адрес/порт и учётные данные."""

    def __init__(self, target: tuple[str, int] | None = None, reply: int = 0, require_auth: bool = False):
        self.target = target
        self.reply = reply
        self.require_auth = require_auth
        self.requests: list[dict] = []
        super().__init__(self._serve)

    @staticmethod
    def _exact(conn: socket.socket, size: int) -> bytes:
        buf = b""
        while len(buf) < size:
            data = conn.recv(size - len(buf))
            if not data:
                raise OSError("closed")
            buf += data
        return buf

    def _serve(self, conn: socket.socket) -> None:
        conn.settimeout(5)
        _, nmethods = self._exact(conn, 2)
        methods = list(self._exact(conn, nmethods))
        record: dict = {"methods": methods, "auth": None}
        if self.require_auth:
            conn.sendall(b"\x05\x02")
            _, ulen = self._exact(conn, 2)
            user = self._exact(conn, ulen).decode()
            (plen,) = self._exact(conn, 1)
            record["auth"] = (user, self._exact(conn, plen).decode())
            conn.sendall(b"\x01\x00")
        else:
            conn.sendall(b"\x05\x00")
        _, cmd, _, atyp = self._exact(conn, 4)
        if atyp == 3:
            (length,) = self._exact(conn, 1)
            host = self._exact(conn, length).decode()
        elif atyp == 1:
            host = socket.inet_ntoa(self._exact(conn, 4))
        else:
            host = socket.inet_ntop(socket.AF_INET6, self._exact(conn, 16))
        (port,) = struct.unpack(">H", self._exact(conn, 2))
        record.update(cmd=cmd, atyp=atyp, host=host, port=port)
        self.requests.append(record)
        if self.reply != 0:
            conn.sendall(bytes([5, self.reply, 0, 1]) + b"\x00" * 6)
            return
        remote = socket.create_connection(self.target or (host, port), timeout=5)
        conn.sendall(b"\x05\x00\x00\x01" + b"\x00" * 6)
        pipe(conn, remote, self.stop)


def echo_handler(conn: socket.socket) -> None:
    conn.settimeout(5)
    while True:
        data = conn.recv(4096)
        if not data:
            return
        conn.sendall(data)


def basic_token(user: str, password: str) -> str:
    return base64.b64encode(f"{user}:{password}".encode()).decode()


# ───────────────────────────── клиент ─────────────────────────────


def dechunk(data: bytes) -> tuple[bytes, bool]:
    """Декодировать chunked-тело; второй элемент — найден ли терминирующий chunk."""
    out = b""
    pos = 0
    while True:
        end = data.find(b"\r\n", pos)
        if end < 0:
            return out, False
        size = int(data[pos:end].split(b";")[0], 16)
        if size == 0:
            return out, True
        if len(data) < end + 2 + size:
            return out + data[end + 2 :], False
        out += data[end + 2 : end + 2 + size]
        pos = end + 2 + size + 2


@dataclass
class RawResponse:
    raw: bytes
    status: int
    headers: dict[str, str]
    body_raw: bytes
    body: bytes = b""
    terminated: bool = True
    json: dict = field(default_factory=dict)


def parse_response(raw: bytes) -> RawResponse:
    import json as _json

    head, _, body_raw = raw.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    status = int(lines[0].split(" ")[1]) if lines and lines[0] else 0
    headers = {}
    for line in lines[1:]:
        key, _, value = line.partition(":")
        headers[key.strip().lower()] = value.strip()
    body, terminated = body_raw, True
    if headers.get("transfer-encoding", "").lower() == "chunked":
        body, terminated = dechunk(body_raw)
    parsed: dict = {}
    if headers.get("content-type", "").startswith("application/json"):
        try:
            parsed = _json.loads(body)
        except ValueError:
            parsed = {}
    return RawResponse(raw, status, headers, body_raw, body, terminated, parsed)


def request_bytes(method: str, path: str, body: bytes = b"", headers: dict | None = None) -> bytes:
    lines = [f"{method} {path} HTTP/1.1", f"Host: {LOCALHOST}"]
    merged = dict(headers or {})
    if body and not any(k.lower() == "transfer-encoding" for k in merged):
        merged.setdefault("Content-Length", str(len(body)))
    lines.extend(f"{k}: {v}" for k, v in merged.items())
    return ("\r\n".join(lines) + "\r\n\r\n").encode() + body


def open_request(port: int, method: str, path: str, body: bytes = b"", headers: dict | None = None) -> socket.socket:
    sock = socket.create_connection((LOCALHOST, port), timeout=10)
    sock.sendall(request_bytes(method, path, body, headers))
    return sock


def read_until(sock: socket.socket, marker: bytes, timeout: float = 5.0) -> bytes:
    buf = b""
    deadline = time.monotonic() + timeout
    while marker not in buf:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"marker {marker!r} not received; got {buf!r}")
        sock.settimeout(remaining)
        data = sock.recv(65536)
        if not data:
            break
        buf += data
    return buf


def read_all(sock: socket.socket, timeout: float = 10.0) -> bytes:
    buf = b""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"connection not closed; got {buf!r}")
        sock.settimeout(remaining)
        try:
            data = sock.recv(65536)
        except ConnectionResetError:
            return buf
        if not data:
            return buf
        buf += data


def http_request(port: int, method: str, path: str, body: bytes = b"", headers: dict | None = None) -> RawResponse:
    sock = open_request(port, method, path, body, headers)
    try:
        return parse_response(read_all(sock))
    finally:
        sock.close()


def wait_for(predicate, timeout: float = 5.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# ───────────────────────────── мост ─────────────────────────────


def make_config(upstream_port: int, pools: dict | None = None, rules: list | None = None, **extra) -> dict:
    """Конфигурация моста для тестов: все три upstream указывают на один фейковый порт без TLS."""
    server = {
        "listen": LOCALHOST,
        "port": 0,
        "connect_timeout": 2.0,
        "total_egress_deadline": 4.0,
        "headers_timeout": 5.0,
        "inactivity_timeout": 5.0,
        "body_timeout": 5.0,
    }
    server.update(extra.pop("server", {}))
    upstreams = {
        name: {"host": LOCALHOST, "port": upstream_port, "use_tls": False}
        for name in ("opencode-ai", "cordis-ai", "openrouter-ai")
    }
    upstreams.update(extra.pop("upstreams", {}))
    config = {
        "server": server,
        "upstreams": upstreams,
        "pools": {"direct": {"type": "direct"}, **(pools or {})},
        "rules": rules or [],
    }
    config.update(extra)
    return config


class BridgeHarness:
    def __init__(self, config: dict, **kwargs):
        self.server = BridgeServer(ConfigManager.from_dict(config), **kwargs)
        self.thread = self.server.start_background()
        self.port = self.server.server_address[1]

    def close(self) -> None:
        self.server.stop()

    def health(self) -> dict:
        return http_request(self.port, "GET", "/health").json

    def slots_idle(self) -> bool:
        slots = self.health()["slots"]
        return all(s["in_use"] == 0 for s in slots.values())


def new_proxy_manager() -> ProxyPoolManager:
    return ProxyPoolManager()


def config_from(data: dict):
    return parse_config(data)
