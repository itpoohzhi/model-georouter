#!/usr/bin/env python3
"""Fail-closed Claude Code sidecar skeleton.

Generation is deliberately unavailable until the Phase-0 protocol fixtures are
verified. This module is standalone and uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import getpass
import http.server
import json
import logging
import os
import re
import signal
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

LOGGER = logging.getLogger("claude_code_sidecar")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 10835
MAX_REQUEST_BYTES = 4 * 1024 * 1024
GENERATION_PATHS = frozenset(
    {
        "/v1internal:streamGenerateContent",
        "/v1internal:generateContent",
        "/v1internal:countTokens",
    }
)
SUPPORTED_POST_PATHS = GENERATION_PATHS | {"/v1internal:fetchAvailableModels"}
ISOLATION_ENV = {
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "DISABLE_AUTOUPDATER": "1",
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "DISABLE_FEEDBACK_COMMAND": "1",
    "CLAUDE_CODE_DISABLE_TERMINAL_TITLE": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
}
SECRET_ENV_NAMES = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_FOUNDRY_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_OAUTH_ACCESS_TOKEN",
        "CLOUDSDK_AUTH_ACCESS_TOKEN",
    }
)


class SidecarConfigError(ValueError):
    """Invalid sidecar configuration."""


class PinnedVersionError(RuntimeError):
    """The configured Claude binary does not match the pinned version."""


class CliTimeoutError(TimeoutError):
    """A Claude process exceeded its execution deadline."""


@dataclass(frozen=True)
class SidecarConfig:
    model_mapping: Mapping[str, str] = MappingProxyType({})
    claude_bin: str = "~/.local/bin/claude"
    claude_bin_pinned_version: str = "2.1.322"
    sidecar_keepalive_seconds: float = 39.0
    sidecar_non_stream_timeout_seconds: float = 40.0
    sidecar_header_timeout_seconds: float = 45.0
    sidecar_idle_timeout_seconds: float = 120.0
    sidecar_max_execution_timeout_seconds: float = 300.0

    @property
    def resolved_claude_bin(self) -> str:
        return str(Path(self.claude_bin).expanduser().resolve())


def _parse_model_mapping(value: object) -> Mapping[str, str]:
    if not isinstance(value, dict):
        raise SidecarConfigError("cloudcode_claude_code.model_mapping must be an object")
    mapping: dict[str, str] = {}
    for source, target in value.items():
        if not isinstance(source, str) or not source:
            raise SidecarConfigError("model_mapping keys must be non-empty strings")
        if (
            not isinstance(target, str)
            or not target
            or not target.isascii()
            or target.startswith("-")
            or any(character.isspace() for character in target)
        ):
            raise SidecarConfigError(f"invalid exact Claude model mapping for {source!r}")
        mapping[source] = target
    return MappingProxyType(mapping)


def _positive_seconds(section: Mapping[str, Any], name: str, default: float) -> float:
    value = section.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise SidecarConfigError(f"cloudcode_claude_code.{name} must be positive")
    return float(value)


def _validate_config(config: SidecarConfig, port: int = DEFAULT_PORT) -> None:
    try:
        absolute_claude_bin = isinstance(config.claude_bin, str) and Path(config.claude_bin).expanduser().is_absolute()
    except RuntimeError:
        absolute_claude_bin = False
    if not absolute_claude_bin:
        raise SidecarConfigError("claude_bin must be an absolute or expandable tilde path")

    if (
        not isinstance(config.claude_bin_pinned_version, str)
        or re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z._-]*", config.claude_bin_pinned_version) is None
    ):
        raise SidecarConfigError(
            "claude_bin_pinned_version must be non-empty and contain only ASCII letters, digits, '.', '_' or '-'"
        )

    if not (
        config.sidecar_keepalive_seconds
        < config.sidecar_non_stream_timeout_seconds
        < config.sidecar_header_timeout_seconds
        < config.sidecar_idle_timeout_seconds
        <= config.sidecar_max_execution_timeout_seconds
    ):
        raise SidecarConfigError("invalid sidecar timeout chain")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise SidecarConfigError("port must be between 1 and 65535")


def load_config(path: str | os.PathLike[str]) -> SidecarConfig:
    try:
        with Path(path).expanduser().open(encoding="utf-8") as config_file:
            root = json.load(config_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise SidecarConfigError("cannot load sidecar JSON configuration") from exc
    if not isinstance(root, dict):
        raise SidecarConfigError("configuration root must be an object")
    section = root.get("cloudcode_claude_code", {})
    if not isinstance(section, dict):
        raise SidecarConfigError("cloudcode_claude_code must be an object")

    config = SidecarConfig(
        model_mapping=_parse_model_mapping(section.get("model_mapping", {})),
        claude_bin=section.get("claude_bin", SidecarConfig.claude_bin),
        claude_bin_pinned_version=section.get("claude_bin_pinned_version", SidecarConfig.claude_bin_pinned_version),
        sidecar_keepalive_seconds=_positive_seconds(section, "sidecar_keepalive_seconds", 39.0),
        sidecar_non_stream_timeout_seconds=_positive_seconds(section, "sidecar_non_stream_timeout_seconds", 40.0),
        sidecar_header_timeout_seconds=_positive_seconds(section, "sidecar_header_timeout_seconds", 45.0),
        sidecar_idle_timeout_seconds=_positive_seconds(section, "sidecar_idle_timeout_seconds", 120.0),
        sidecar_max_execution_timeout_seconds=_positive_seconds(
            section, "sidecar_max_execution_timeout_seconds", 300.0
        ),
    )
    _validate_config(config)
    return config


def llm_fixtures_verified() -> bool:
    return os.environ.get("SIDECAR_FIXTURES_VERIFIED", "0") == "1"


def sanitized_cli_env(*, base_url: str | None = None) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in SECRET_ENV_NAMES and not key.startswith(("GOOGLE_", "CLOUDSDK_"))
    }
    env.update(ISOLATION_ENV)
    user = getpass.getuser()
    env.setdefault("USER", user)
    env.setdefault("LOGNAME", user)
    no_proxy = {entry.strip() for entry in env.get("NO_PROXY", "").split(",") if entry.strip()}
    no_proxy.update({"127.0.0.1", "localhost"})
    env["NO_PROXY"] = ",".join(sorted(no_proxy))
    if base_url is not None:
        env["ANTHROPIC_BASE_URL"] = base_url
    else:
        env.pop("ANTHROPIC_BASE_URL", None)
    return env


def verify_pinned_version(config: SidecarConfig) -> None:
    try:
        completed = subprocess.run(
            [config.resolved_claude_bin, "--version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            shell=False,
            env=sanitized_cli_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PinnedVersionError("pinned Claude version check failed") from exc
    output = f"{completed.stdout}\n{completed.stderr}"
    version_pattern = rf"(?<![0-9A-Za-z.]){re.escape(config.claude_bin_pinned_version)}(?![0-9A-Za-z.])"
    if completed.returncode != 0 or re.search(version_pattern, output) is None:
        raise PinnedVersionError("Claude binary does not match claude_bin_pinned_version")


def build_cli_argv(config: SidecarConfig, claude_model_id: str) -> list[str]:
    return [
        config.resolved_claude_bin,
        "-p",
        "--model",
        claude_model_id,
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--no-session-persistence",
        "--max-turns",
        "1",
        "--tools",
        "",
        "--permission-mode",
        "dontAsk",
        "--setting-sources",
        "",
    ]


def spawn_cli(
    config: SidecarConfig,
    claude_model_id: str,
    *,
    base_url: str | None = None,
) -> subprocess.Popen[bytes]:
    """Spawn one isolated CLI process; the HTTP skeleton does not call this yet."""
    return subprocess.Popen(
        build_cli_argv(config, claude_model_id),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        start_new_session=True,
        env=sanitized_cli_env(base_url=base_url),
    )


def terminate_process_group(process: subprocess.Popen[bytes], grace_seconds: float = 3.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def wait_for_cli(
    process: subprocess.Popen[bytes],
    timeout_seconds: float,
    *,
    grace_seconds: float = 3.0,
) -> int:
    try:
        return process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        terminate_process_group(process, grace_seconds)
        raise CliTimeoutError("Claude CLI execution deadline exceeded") from None


@dataclass(frozen=True)
class SseFrame:
    payload: Mapping[str, Any] | None = None
    keepalive: bool = False


def parse_sse_frames(source: str | bytes) -> list[SseFrame]:
    """Parse complete v1internal SSE frames without changing their JSON envelope."""
    if isinstance(source, bytes):
        source = source.decode("utf-8")
    normalized = source.replace("\r\n", "\n").replace("\r", "\n")
    frames: list[SseFrame] = []
    for block in normalized.split("\n\n"):
        if not block.strip():
            continue
        data_lines: list[str] = []
        keepalive = False
        for line in block.split("\n"):
            if line.startswith(":"):
                keepalive = keepalive or line[1:].strip() == "keepalive"
            elif line.startswith("data:"):
                data_lines.append(line[5:].removeprefix(" "))
            elif line:
                raise ValueError("unsupported SSE field")
        if data_lines:
            payload = json.loads("\n".join(data_lines))
            if not isinstance(payload, dict):
                raise ValueError("SSE data must contain a JSON object")
            frames.append(SseFrame(payload=payload))
        elif keepalive:
            frames.append(SseFrame(keepalive=True))
    return frames


def reframe_sse_frame(frame: SseFrame) -> bytes:
    if frame.keepalive:
        if frame.payload is not None:
            raise ValueError("keepalive cannot contain a JSON payload")
        return b": keepalive\n\n"
    if frame.payload is None:
        raise ValueError("v1internal SSE payload is required")
    if not isinstance(frame.payload, dict):
        raise TypeError("v1internal SSE payload must be an object")
    if "response" not in frame.payload:
        raise ValueError("v1internal SSE payload must preserve the response envelope")
    if not isinstance(frame.payload["response"], dict):
        raise TypeError("v1internal response envelope must contain an object")
    data = json.dumps(frame.payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return b"data: " + data + b"\n\n"


class SidecarHTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], config: SidecarConfig):
        super().__init__(address, SidecarHandler)
        self.config = config


class SidecarHandler(http.server.BaseHTTPRequestHandler):
    server: SidecarHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *args: object) -> None:
        """Suppress BaseHTTPRequestHandler request-line/header logging."""

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        LOGGER.info("method=%s status=%s", self.command, code)

    def _send_json(self, status: int, payload: Mapping[str, Any], *, pre_dispatch: bool = False) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if pre_dispatch:
            self.send_header("X-Sidecar-Pre-Dispatch", "1")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message}, pre_dispatch=True)

    def do_GET(self) -> None:
        if urlsplit(self.path).path != "/healthz":
            self._error(404, "not found")
            return
        self._send_json(
            200,
            {"ok": True, "mode": "skeleton", "fixtures_verified": llm_fixtures_verified()},
        )

    def _read_body(self) -> bytes | None:
        if self.headers.get("Transfer-Encoding"):
            self._error(400, "chunked requests are not supported")
            return None
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._error(400, "invalid Content-Length")
            return None
        if length < 0 or length > MAX_REQUEST_BYTES:
            self._error(413, "request body too large")
            return None
        return self.rfile.read(length)

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path not in SUPPORTED_POST_PATHS:
            self._error(404, "not found")
            return
        body = self._read_body()
        if body is None:
            return

        if path in GENERATION_PATHS:
            try:
                payload = json.loads(body or b"{}")
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._error(400, "invalid JSON request")
                return
            if not isinstance(payload, dict):
                self._error(400, "JSON request must be an object")
                return
            model_id = payload.get("model")
            if not isinstance(model_id, str) or model_id not in self.server.config.model_mapping:
                self._error(422, "unknown model_id")
                return

        if not llm_fixtures_verified():
            self._error(503, "LLM protocol fixtures are not verified")
            return
        self._error(501, "generation path is not implemented in the skeleton")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="model-georouter JSON configuration")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--claude-bin")
    parser.add_argument("--claude-bin-pinned-version")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    try:
        config = load_config(args.config)
        if args.claude_bin is not None:
            config = replace(config, claude_bin=args.claude_bin)
        if args.claude_bin_pinned_version is not None:
            config = replace(config, claude_bin_pinned_version=args.claude_bin_pinned_version)
        _validate_config(config, args.port)
        verify_pinned_version(config)
    except (SidecarConfigError, PinnedVersionError) as exc:
        LOGGER.error("startup refused: %s", exc)
        return 2

    try:
        server = SidecarHTTPServer((DEFAULT_HOST, args.port), config)
    except OSError:
        LOGGER.error("startup refused: cannot bind loopback listener")
        return 2
    LOGGER.info("sidecar skeleton listening on %s:%d", DEFAULT_HOST, server.server_port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
