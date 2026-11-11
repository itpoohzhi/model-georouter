"""Минимальный управляемый TCP-сервер для тестов на реальных сокетах."""

from __future__ import annotations

import socket
import threading
from collections.abc import Callable

LOCALHOST = "127.0.0.1"


class FakeServer:
    """TCP-сервер, который при закрытии дожидается accept- и worker-потоков."""

    def __init__(self, handler: Callable[[socket.socket], object]):
        self._handler = handler
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((LOCALHOST, 0))
        self._sock.listen(64)
        self._sock.settimeout(0.02)
        self.port = self._sock.getsockname()[1]
        self._closed = threading.Event()
        self.connections: list[socket.socket] = []
        self._threads: list[threading.Thread] = []
        self.stop = threading.Event()
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        while not self._closed.is_set():
            try:
                conn, _ = self._sock.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            self.connections.append(conn)
            worker = threading.Thread(target=self._run, args=(conn,), daemon=True)
            self._threads.append(worker)
            worker.start()

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
        self._accept_thread.join(0.2)
        for worker in list(self._threads):
            worker.join(0.2)
