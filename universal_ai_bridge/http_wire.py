"""Низкоуровневый HTTP/1.1 wire-слой: чтение голов, парсинг, трекеры обрамления тела."""

from __future__ import annotations

import re
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass

from .errors import BadRequestError, BodyTooLargeError, FramingError, UpstreamProtocolError

HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})
HEAD_LIMIT = 65536
RECV_SIZE = 65536

Headers = list[tuple[str, str]]
Sink = Callable[[bytes], None]


class HeadError(Exception):
    """Не удалось прочитать блок заголовков: `kind` ∈ {timeout, eof, too_large}."""

    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind


def read_head_block(
    sock: socket.socket, *, timeout: float, limit: int = HEAD_LIMIT, initial: bytes = b""
) -> tuple[bytes, bytes]:
    """Прочитать до `\\r\\n\\r\\n`; вернуть (блок заголовков, уже прочитанный хвост тела)."""
    buf = bytearray(initial)
    deadline = time.monotonic() + timeout
    scanned = 0
    while True:
        idx = buf.find(b"\r\n\r\n", max(0, scanned - 3))
        if idx >= 0:
            end = idx + 4
            return bytes(buf[:end]), bytes(buf[end:])
        if len(buf) > limit:
            raise HeadError("too_large")
        scanned = len(buf)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise HeadError("timeout")
        sock.settimeout(remaining)
        try:
            data = sock.recv(RECV_SIZE)
        except TimeoutError:
            raise HeadError("timeout") from None
        if not data:
            raise HeadError("eof")
        buf += data


def header_get(headers: Headers, name: str, default: str | None = None) -> str | None:
    lowered = name.lower()
    for key, value in headers:
        if key.lower() == lowered:
            return value
    return default


def header_values(headers: Headers, name: str) -> list[str]:
    lowered = name.lower()
    return [value for key, value in headers if key.lower() == lowered]


def connection_tokens(headers: Headers) -> set[str]:
    return {t.strip().lower() for v in header_values(headers, "connection") for t in v.split(",") if t.strip()}


def _parse_header_lines(lines: list[str], error: type[Exception]) -> Headers:
    headers: Headers = []
    for line in lines:
        if not line:
            break
        name, sep, value = line.partition(":")
        if not sep or not name or name != name.strip():
            raise error("malformed header line")
        headers.append((name, value.strip()))
    return headers


@dataclass(frozen=True)
class RequestHead:
    method: str
    target: str
    version: str
    headers: Headers

    def get(self, name: str, default: str | None = None) -> str | None:
        return header_get(self.headers, name, default)


@dataclass(frozen=True)
class ResponseHead:
    version: str
    status: int
    reason: str
    headers: Headers

    def get(self, name: str, default: str | None = None) -> str | None:
        return header_get(self.headers, name, default)


def parse_request_head(block: bytes) -> RequestHead:
    lines = block.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) != 3 or not parts[0] or not parts[1] or not parts[2].startswith("HTTP/1."):
        raise BadRequestError("malformed request line")
    return RequestHead(parts[0].upper(), parts[1], parts[2], _parse_header_lines(lines[1:], BadRequestError))


def parse_response_head(block: bytes) -> ResponseHead:
    lines = block.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ", 2)
    if len(parts) < 2 or not parts[0].startswith("HTTP/1.") or not parts[1].isdigit():
        raise UpstreamProtocolError("malformed upstream status line")
    reason = parts[2] if len(parts) > 2 else ""
    return ResponseHead(parts[0], int(parts[1]), reason, _parse_header_lines(lines[1:], UpstreamProtocolError))


def build_client_head(head: ResponseHead) -> bytes:
    """Голова ответа клиенту: hop-by-hop убраны, кроме `transfer-encoding` (тело идёт как есть)."""
    skip = (HOP_BY_HOP - {"transfer-encoding"}) | connection_tokens(head.headers)
    lines = [f"HTTP/1.1 {head.status} {head.reason}"]
    lines.extend(f"{key}: {value}" for key, value in head.headers if key.lower() not in skip)
    lines.append("Connection: close")
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


# ───────────────────────────── обрамление тела ─────────────────────────────


class Framing:
    """Трекер границы тела: `feed` возвращает число потреблённых wire-байтов."""

    complete: bool = False

    def __init__(self, sink: Sink | None = None):
        self._sink = sink

    def _emit(self, data: bytes) -> None:
        if self._sink is not None and data:
            self._sink(data)

    def feed(self, data: bytes) -> int:
        raise NotImplementedError

    def on_eof(self) -> bool:
        """Законно ли завершение тела по закрытию соединения."""
        return self.complete


class NoBodyFraming(Framing):
    complete = True

    def feed(self, data: bytes) -> int:
        return 0


class EofFraming(Framing):
    """Тело ограничено закрытием соединения."""

    def feed(self, data: bytes) -> int:
        self._emit(data)
        return len(data)

    def on_eof(self) -> bool:
        return True


class LengthFraming(Framing):
    def __init__(self, length: int, sink: Sink | None = None):
        super().__init__(sink)
        self.remaining = length
        self.complete = length == 0

    def feed(self, data: bytes) -> int:
        take = min(self.remaining, len(data))
        self._emit(data[:take])
        self.remaining -= take
        self.complete = self.remaining == 0
        return take


class ChunkedFraming(Framing):
    """Инкрементальный разбор `Transfer-Encoding: chunked`; wire-байты сохраняются как есть."""

    _SIZE, _DATA, _DATA_END, _TRAILER = range(4)
    LINE_LIMIT = 4096

    def __init__(self, sink: Sink | None = None):
        super().__init__(sink)
        self._state = self._SIZE
        self._remaining = 0
        self._line = bytearray()

    def _take_line(self, data: bytes, i: int) -> tuple[bytes | None, int]:
        j = data.find(b"\n", i)
        if j == -1:
            self._line += data[i:]
            if len(self._line) > self.LINE_LIMIT:
                raise FramingError("chunk line too long")
            return None, len(data)
        self._line += data[i:j]
        line = bytes(self._line)
        self._line.clear()
        return line, j + 1

    def feed(self, data: bytes) -> int:
        i, n = 0, len(data)
        while i < n and not self.complete:
            if self._state == self._SIZE:
                line, i = self._take_line(data, i)
                if line is None:
                    break
                token = line.split(b";", 1)[0].strip()
                if not re.fullmatch(rb"[0-9a-fA-F]{1,16}", token):
                    raise FramingError("invalid chunk size")
                self._remaining = int(token, 16)
                self._state = self._TRAILER if self._remaining == 0 else self._DATA
            elif self._state == self._DATA:
                take = min(self._remaining, n - i)
                self._emit(data[i : i + take])
                self._remaining -= take
                i += take
                if self._remaining == 0:
                    self._state = self._DATA_END
            elif self._state == self._DATA_END:
                j = data.find(b"\n", i)
                if j == -1:
                    i = n
                else:
                    i = j + 1
                    self._state = self._SIZE
            else:
                line, i = self._take_line(data, i)
                if line is None:
                    break
                if line.strip() == b"":
                    self.complete = True
        return i

    def on_eof(self) -> bool:
        return self.complete


_CONTENT_LENGTH_RE = re.compile(r"[0-9]{1,18}")


def framing_for_request(headers: Headers, sink: Sink | None, limit: int) -> Framing:
    """Обрамление тела запроса клиента; нарушения — BadRequestError, превышение лимита — BodyTooLargeError."""
    encodings = header_values(headers, "transfer-encoding")
    lengths = header_values(headers, "content-length")
    if encodings and lengths:
        raise BadRequestError("ambiguous body framing")
    if encodings:
        codings = [c.strip().lower() for v in encodings for c in v.split(",") if c.strip()]
        if codings != ["chunked"]:
            raise BadRequestError("unsupported transfer-encoding")
        return ChunkedFraming(sink)
    if lengths:
        if len(lengths) > 1 or not _CONTENT_LENGTH_RE.fullmatch(lengths[0].strip()):
            raise BadRequestError("invalid content-length")
        size = int(lengths[0].strip())
        if size > limit:
            raise BodyTooLargeError(limit)
        return LengthFraming(size, sink)
    return NoBodyFraming()


def framing_for_response(method: str, status: int, headers: Headers, sink: Sink | None = None) -> Framing:
    """Обрамление тела ответа upstream."""
    if method == "HEAD" or status in (204, 304) or 100 <= status < 200:
        return NoBodyFraming()
    encodings = [c.strip().lower() for v in header_values(headers, "transfer-encoding") for c in v.split(",")]
    if encodings and encodings[-1] == "chunked":
        return ChunkedFraming(sink)
    lengths = header_values(headers, "content-length")
    if lengths:
        value = lengths[0].strip()
        if not _CONTENT_LENGTH_RE.fullmatch(value):
            raise UpstreamProtocolError("invalid upstream content-length")
        return LengthFraming(int(value), sink)
    return EofFraming(sink)
