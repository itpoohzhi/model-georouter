"""Injectable I/O pre-relay client disconnect watcher for model-georouter (W14)."""

from __future__ import annotations

import json
import os
import selectors
import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, Protocol

from .errors import UpstreamTimeoutError
from .prerelay_policy import (
    ClientWatchAction,
    PreRelayClientResetError,
    classify_client_oserror,
    classify_client_peek,
    resolve_prerelay_flag,
)

DEFAULT_FLAGS_PATH_ENV: Final[str] = "GEOROUTER_DYNAMIC_FLAGS_PATH"
_DEFAULT_FLAGS_FILE: Final[Path] = Path.home() / ".config/model-georouter/dynamic-flags.json"
_MSG_PEEK_NONBLOCK: Final[int] = socket.MSG_PEEK | getattr(socket, "MSG_DONTWAIT", 0)

LegacyHeadReader = Callable[[socket.socket, float, Any], tuple[Any, bytes]]


class PreRelayWatcher(Protocol):
    """Interface allowing future shared-selector evolution without changing policy."""

    def read_head(
        self,
        client_sock: socket.socket | None,
        up_sock: socket.socket,
        timeout: float,
        deadline: Any,
        legacy_reader: LegacyHeadReader,
    ) -> tuple[Any, bytes]: ...


class DynamicFlagCache:
    """(path, mtime_ns, size, st_ino)-cached JSON > env > False flag reader."""

    def __init__(self, flags_path: Path | None = None) -> None:
        self._explicit_path = flags_path
        self._lock = threading.Lock()
        self._cache_key: tuple[str, int, int, int] | None = None
        self._json_snap: dict[str, object] | None = None
        self._json_invalid = False

    def _resolve_path(self) -> Path:
        if self._explicit_path is not None:
            return self._explicit_path
        raw = os.environ.get(DEFAULT_FLAGS_PATH_ENV, "").strip()
        return Path(raw).expanduser() if raw else _DEFAULT_FLAGS_FILE

    def _load_json_cached(self) -> tuple[dict[str, object] | None, bool]:
        path = self._resolve_path()
        try:
            st = path.stat()
        except OSError:
            with self._lock:
                self._cache_key, self._json_snap, self._json_invalid = None, None, False
            return None, False
        key = (str(path), st.st_mtime_ns, st.st_size, st.st_ino)
        with self._lock:
            if self._cache_key == key:
                return self._json_snap, self._json_invalid
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            snap = data if isinstance(data, dict) else None
            invalid = snap is None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            snap, invalid = None, True
        with self._lock:
            self._cache_key, self._json_snap, self._json_invalid = key, snap, invalid
        return snap, invalid

    def is_enabled(self) -> bool:
        snap, invalid = self._load_json_cached()
        return resolve_prerelay_flag(snap, os.environ, json_invalid=invalid)


_DEFAULT_FLAG_CACHE = DynamicFlagCache()


def _remaining_wait(timeout: float, deadline: Any, started_at: float) -> float:
    phase_left = max(0.0, timeout - (time.monotonic() - started_at))
    if phase_left <= 0.0:
        return 0.0
    if deadline is not None and hasattr(deadline, "clamp"):
        try:
            return max(0.0, float(deadline.clamp(phase_left)))
        except TimeoutError:
            raise UpstreamTimeoutError(f"upstream sent no response headers within {timeout:g}s") from None
    return phase_left


def _probe_client_socket(client_sock: socket.socket) -> ClientWatchAction:
    try:
        so_error = int(client_sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR))
    except OSError as exc:
        return classify_client_oserror(exc.errno)
    if so_error != 0:
        return classify_client_oserror(so_error)
    try:
        peek = client_sock.recv(1, _MSG_PEEK_NONBLOCK)
    except OSError as exc:
        return classify_client_oserror(exc.errno)
    return classify_client_peek(peek, 0)


def _upstream_has_buffered_tls(up_sock: socket.socket) -> bool:
    pending_fn = getattr(up_sock, "pending", None)
    if callable(pending_fn):
        try:
            return int(pending_fn()) > 0
        except OSError:
            return False
    return False


def _handle_client_ready(sel: selectors.BaseSelector, client_sock: socket.socket) -> bool:
    action = _probe_client_socket(client_sock)
    if action is ClientWatchAction.RST_CANCEL:
        raise PreRelayClientResetError("client_rst_prerelay")
    if action in (ClientWatchAction.FIN_UNWATCH, ClientWatchAction.PEEK_DATA_UNWATCH):
        sel.unregister(client_sock)
        return False
    return True


def _register_client(sel: selectors.BaseSelector, client_sock: socket.socket) -> bool:
    try:
        sel.register(client_sock, selectors.EVENT_READ, data="client")
        return True
    except (OSError, ValueError):
        return False


def _wait_upstream_or_client_reset(
    client_sock: socket.socket,
    up_sock: socket.socket,
    timeout: float,
    deadline: Any,
    started_at: float,
) -> bool:
    if _upstream_has_buffered_tls(up_sock):
        return True
    with selectors.DefaultSelector() as sel:
        sel.register(up_sock, selectors.EVENT_READ, data="up")
        watch_client = _register_client(sel, client_sock)
        while True:
            rem = _remaining_wait(timeout, deadline, started_at)
            if rem <= 0.0:
                return False
            events = sel.select(timeout=rem)
            if not events:
                return False
            ready = {key.data for key, _ in events}
            if watch_client and "client" in ready:
                watch_client = _handle_client_ready(sel, client_sock)
            if "up" in ready:
                return True


class SelectorPreRelayWatcher:
    """Default per-request DefaultSelector watcher (kqueue/epoll, no FD_SETSIZE limit)."""

    def __init__(self, flag_cache: DynamicFlagCache | None = None) -> None:
        self._flag_cache = flag_cache or _DEFAULT_FLAG_CACHE

    def read_head(
        self,
        client_sock: socket.socket | None,
        up_sock: socket.socket,
        timeout: float,
        deadline: Any,
        legacy_reader: LegacyHeadReader,
    ) -> tuple[Any, bytes]:
        if client_sock is None or not self._flag_cache.is_enabled():
            return legacy_reader(up_sock, timeout, deadline)
        started_at = time.monotonic()
        up_ready = _wait_upstream_or_client_reset(client_sock, up_sock, timeout, deadline, started_at)
        if not up_ready:
            raise UpstreamTimeoutError(f"upstream sent no response headers within {timeout:g}s")
        rem = _remaining_wait(timeout, deadline, started_at)
        if rem <= 0.0:
            rem = min(max(timeout, 1e-3), 0.05)
        return legacy_reader(up_sock, rem, deadline)


_DEFAULT_WATCHER = SelectorPreRelayWatcher()


def read_upstream_head_watched(
    client_sock: socket.socket | None,
    up_sock: socket.socket,
    timeout: float,
    deadline: Any,
    legacy_reader: LegacyHeadReader,
) -> tuple[Any, bytes]:
    """Module-level delegation seam for bridge_server._attempt (<=5 LOC call-site)."""
    return _DEFAULT_WATCHER.read_head(client_sock, up_sock, timeout, deadline, legacy_reader)
