"""Регрессионные тесты жизненного цикла потоков и остановки BridgeServer."""

from __future__ import annotations

import socket
import socketserver
import threading

from model_georouter.bridge_server import BridgeServer, _WorkerThreads
from model_georouter.config import ConfigManager
from tests.helpers import make_config


def test_bridge_server_stop_joins_worker_threads_and_handles_no_threads() -> None:
    joined: list[tuple[str, float | None]] = []

    class _SpyThread:
        def __init__(self, name: str, alive: bool) -> None:
            self.name, self._alive = name, alive

        def is_alive(self) -> bool:
            return self._alive

        def join(self, timeout: float | None = None) -> None:
            joined.append((self.name, timeout))

    wt = _WorkerThreads()
    t_dead, t_live1, t_live2 = _SpyThread("dead", False), _SpyThread("t1", True), _SpyThread("t2", True)
    wt.append(t_dead)  # type: ignore[arg-type]
    wt.append(t_live1)  # type: ignore[arg-type]
    assert list(wt) == [t_live1]
    wt.append(t_live2)  # type: ignore[arg-type]
    assert list(wt) == [t_live1, t_live2]
    wt.join()
    assert list(wt) == [] and joined == [("t1", 0.5), ("t2", 0.5)]

    server = BridgeServer(ConfigManager.from_dict(make_config(1, server={"max_connections": 2})))
    entered = threading.Event()
    orig_process = server.process_request_thread

    def tracked_process(request, client_address) -> None:
        entered.set()
        orig_process(request, client_address)

    server.process_request_thread = tracked_process  # type: ignore[method-assign]
    bg = server.start_background()
    assert bg in server._threads
    port = server.server_address[1]
    holder = socket.create_connection(("127.0.0.1", port), timeout=5)
    try:
        holder.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: h\r\n")
        assert entered.wait(2.0)
        workers = list(server._threads)
        assert len(workers) >= 2
    finally:
        holder.close()
        server.stop()
    assert list(server._threads) == [] and not bg.is_alive() and all(not t.is_alive() for t in workers)
    server.stop()
    server._threads = socketserver._NoThreads()  # type: ignore[assignment]
    server.stop()
