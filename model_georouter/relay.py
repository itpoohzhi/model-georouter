"""Слой 4: потоковый relay upstream → клиент (SSE) с backpressure и детектированием обрыва клиента."""

from __future__ import annotations

import select
import socket
import ssl
import time
from dataclasses import dataclass

from .errors import FramingError
from .http_wire import Framing

RELAY_CHUNK = 16384
CLIENT_PROBE_SIZE = 4096

COMPLETE = "complete"
CLIENT_CANCELLED = "client_cancelled"
CLIENT_WRITE_TIMEOUT = "client_write_timeout"
UPSTREAM_TRUNCATED = "upstream_truncated"
UPSTREAM_ERROR = "upstream_error"
INACTIVITY_TIMEOUT = "inactivity_timeout"


@dataclass(frozen=True)
class RelayResult:
    outcome: str
    bytes_sent: int

    @property
    def ok(self) -> bool:
        return self.outcome == COMPLETE


def configure_socket(sock: socket.socket) -> None:
    """TCP_NODELAY: каждый SSE-чанк уходит немедленно."""
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass


class _ClientGone(Exception):
    pass


class _ClientTimeout(Exception):
    pass


def close_quietly(sock: socket.socket | None) -> None:
    if sock is None:
        return
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass


class SSERelay:
    """Гоняет байты upstream → клиент через `select`, сохраняя обрамление тела как есть.

    * Читает upstream только когда предыдущая запись клиенту завершена (backpressure).
    * При обрыве клиента немедленно закрывает upstream.
    * При обрыве upstream посреди тела ничего не дописывает (никакого ложного `data: [DONE]`
      и терминирующего chunk): вызывающий закрывает клиентский сокет «как есть».
    """

    def __init__(
        self,
        *,
        inactivity_timeout: float,
        write_timeout: float,
        chunk_size: int = RELAY_CHUNK,
        poll_interval: float = 0.25,
    ):
        self._inactivity = inactivity_timeout
        self._write_timeout = write_timeout
        self._chunk = chunk_size
        self._poll = poll_interval

    def _send(self, client: socket.socket, data: bytes) -> None:
        view = memoryview(data)
        client.settimeout(self._write_timeout)
        while view:
            try:
                sent = client.send(view)
            except TimeoutError:
                raise _ClientTimeout from None
            except OSError:
                raise _ClientGone from None
            view = view[sent:]

    @staticmethod
    def _client_gone(client: socket.socket) -> bool:
        """Клиент читаем: либо EOF/сброс (ушёл), либо прислал лишние байты (отбрасываем)."""
        try:
            client.settimeout(0)
            data = client.recv(CLIENT_PROBE_SIZE)
        except (BlockingIOError, InterruptedError):
            return False
        except OSError:
            return True
        return data == b""

    def relay(
        self,
        upstream: socket.socket,
        client: socket.socket,
        framing: Framing,
        *,
        initial: bytes = b"",
        initial_fed: bool = False,
    ) -> RelayResult:
        configure_socket(upstream)
        configure_socket(client)
        sent = 0
        try:
            if initial:
                if not initial_fed:
                    initial = initial[: framing.feed(initial)]
                self._send(client, initial)
                sent += len(initial)
            last_data = time.monotonic()
            while not framing.complete:
                pending = isinstance(upstream, ssl.SSLSocket) and upstream.pending() > 0
                try:
                    readable, _, _ = select.select([upstream, client], [], [], 0 if pending else self._poll)
                except (OSError, ValueError):
                    return RelayResult(UPSTREAM_ERROR, sent)
                if client in readable and self._client_gone(client):
                    return RelayResult(CLIENT_CANCELLED, sent)
                if upstream in readable or pending:
                    try:
                        upstream.settimeout(0)
                        data = upstream.recv(self._chunk)
                    except (BlockingIOError, InterruptedError, ssl.SSLWantReadError):
                        data = None
                    except OSError:
                        return RelayResult(UPSTREAM_ERROR, sent)
                    if data == b"":
                        if framing.on_eof():
                            break
                        return RelayResult(UPSTREAM_TRUNCATED, sent)
                    if data:
                        last_data = time.monotonic()
                        consumed = framing.feed(data)
                        self._send(client, data[:consumed])
                        sent += consumed
                if time.monotonic() - last_data > self._inactivity:
                    return RelayResult(INACTIVITY_TIMEOUT, sent)
            return RelayResult(COMPLETE, sent)
        except _ClientGone:
            return RelayResult(CLIENT_CANCELLED, sent)
        except _ClientTimeout:
            return RelayResult(CLIENT_WRITE_TIMEOUT, sent)
        except FramingError:
            return RelayResult(UPSTREAM_ERROR, sent)
        finally:
            close_quietly(upstream)
