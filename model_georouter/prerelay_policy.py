"""Pure Zero-I/O policy for model-georouter pre-relay client disconnect (W14)."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Final

FLAG_GEO_PRERELAY_DISCONNECT: Final[str] = "GEOROUTER_PRERELAY_DISCONNECT_ENABLED"

_TRUE_STRINGS: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_FALSE_STRINGS: Final[frozenset[str]] = frozenset({"0", "false", "no", "off", ""})

# POSIX errno numbers (macOS & Linux) for client-side hard disconnect vs transient wakeup
_TRANSIENT_ERRNOS: Final[frozenset[int]] = frozenset({4, 11, 35})  # EINTR, EAGAIN(linux), EAGAIN(darwin)
_RESET_ERRNOS: Final[frozenset[int]] = frozenset({
    32,   # EPIPE
    53,   # ECONNABORTED (darwin)
    54,   # ECONNRESET (darwin)
    60,   # ETIMEDOUT (darwin)
    103,  # ECONNABORTED (linux)
    104,  # ECONNRESET (linux)
    110,  # ETIMEDOUT (linux)
})


class PreRelayClientResetError(Exception):
    """Domain exception for confirmed pre-relay client disconnect.

    Intentionally inherits bare Exception (NOT OSError/ConnectionError) so
    upstream proxy penalty/failover handlers in bridge_server._attempt never
    misclassify a client reset as an upstream/proxy failure.
    """

    def __init__(self, reason: str, errno_code: int = 0) -> None:
        super().__init__(reason)
        self.reason = reason
        self.errno_code = errno_code


class ClientWatchAction(Enum):
    """Action for pre-relay client socket observation during header wait."""

    KEEP = "keep"
    FIN_UNWATCH = "fin_unwatch"
    PEEK_DATA_UNWATCH = "peek_data_unwatch"
    RST_CANCEL = "rst_cancel"


def parse_bool_strict(raw: object) -> bool | None:
    """Parse boolean token fail-closed (identical golden vectors with bridge)."""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        return bool(raw) if raw in (0, 1) else None
    if isinstance(raw, str):
        norm = raw.strip().lower()
        if norm in _TRUE_STRINGS:
            return True
        if norm in _FALSE_STRINGS:
            return False
    return None


def resolve_prerelay_flag(
    json_snapshot: Mapping[str, object] | None,
    env_snapshot: Mapping[str, str] | None,
    *,
    json_invalid: bool = False,
) -> bool:
    """Resolve GEOROUTER_PRERELAY_DISCONNECT_ENABLED: dynamic JSON > env > False."""
    if json_invalid:
        return False
    if json_snapshot is not None:
        for key in (FLAG_GEO_PRERELAY_DISCONNECT, FLAG_GEO_PRERELAY_DISCONNECT.lower()):
            if key in json_snapshot:
                parsed = parse_bool_strict(json_snapshot[key])
                return bool(parsed) if parsed is not None else False
    if env_snapshot is not None and FLAG_GEO_PRERELAY_DISCONNECT in env_snapshot:
        env_parsed = parse_bool_strict(env_snapshot[FLAG_GEO_PRERELAY_DISCONNECT])
        return bool(env_parsed) if env_parsed is not None else False
    return False


def classify_client_oserror(errno_code: int | None) -> ClientWatchAction:
    """Classify an OSError raised exclusively on client_sock operations."""
    if errno_code in _TRANSIENT_ERRNOS:
        return ClientWatchAction.KEEP
    if errno_code in _RESET_ERRNOS:
        return ClientWatchAction.RST_CANCEL
    return ClientWatchAction.FIN_UNWATCH


def classify_client_peek(peek_bytes: bytes, so_error: int) -> ClientWatchAction:
    """Classify non-destructive MSG_PEEK result + SO_ERROR on client_sock."""
    if so_error != 0:
        return classify_client_oserror(so_error)
    if peek_bytes == b"":
        return ClientWatchAction.FIN_UNWATCH
    return ClientWatchAction.PEEK_DATA_UNWATCH
