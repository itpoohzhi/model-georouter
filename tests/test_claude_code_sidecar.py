from __future__ import annotations

import http.client
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from sidecar.claude_code_sidecar import (
    CliTimeoutError,
    SidecarConfig,
    SidecarHTTPServer,
    parse_sse_frames,
    reframe_sse_frame,
    spawn_cli,
    wait_for_cli,
)

ROOT = Path(__file__).resolve().parents[1]
SIDECAR_SCRIPT = ROOT / "sidecar" / "claude_code_sidecar.py"
FIXTURE = ROOT / ".ai-team/runs/antigravity-claude-code-02/raw/cli-fixtures/open-01-response.txt"
MODEL = "claude-synthetic"
TARGET_MODEL = "claude-sonnet-exact"


def _write_fake_cli(tmp_path: Path) -> Path:
    fake_cli = tmp_path / "claude"
    fake_cli.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
import time

if "--version" in sys.argv:
    print(os.environ.get("FAKE_CLI_VERSION", "2.1.322"))
    raise SystemExit(0)
output = Path(os.environ["FAKE_CLI_DIR"])
(output / "spawned").write_text("1", encoding="utf-8")
(output / "pid").write_text(str(os.getpid()), encoding="utf-8")
(output / "argv.json").write_text(json.dumps(sys.argv[1:]), encoding="utf-8")
secret_names = [
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_FOUNDRY_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_OAUTH_ACCESS_TOKEN",
    "CLOUDSDK_AUTH_ACCESS_TOKEN",
]
(output / "leaked-env.json").write_text(
    json.dumps([name for name in secret_names if name in os.environ]), encoding="utf-8"
)
(output / "base-url.txt").write_text(os.environ.get("ANTHROPIC_BASE_URL", ""), encoding="utf-8")
time.sleep(60)
""",
        encoding="utf-8",
    )
    fake_cli.chmod(0o755)
    return fake_cli


def _config(fake_cli: Path) -> SidecarConfig:
    return SidecarConfig(
        model_mapping=MappingProxyType({MODEL: TARGET_MODEL}),
        claude_bin=str(fake_cli),
        claude_bin_pinned_version="2.1.322",
    )


def _request(
    port: int,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    body = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"} if body is not None else {}
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    response_body = response.read()
    response_headers = {name.lower(): value for name, value in response.getheaders()}
    status = response.status
    connection.close()
    return status, response_headers, response_body


@pytest.fixture
def serve() -> Iterator[Any]:
    running: list[tuple[SidecarHTTPServer, threading.Thread]] = []

    def start(config: SidecarConfig) -> int:
        server = SidecarHTTPServer(("127.0.0.1", 0), config)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        running.append((server, thread))
        return server.server_port

    yield start
    for server, thread in reversed(running):
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_healthz(serve: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIDECAR_FIXTURES_VERIFIED", raising=False)
    port = serve(_config(_write_fake_cli(tmp_path)))

    status, _headers, body = _request(port, "GET", "/healthz")

    assert status == 200
    assert json.loads(body) == {"ok": True, "mode": "skeleton", "fixtures_verified": False}


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/v1internal:streamGenerateContent", {"model": MODEL, "request": {"contents": []}}),
        ("/v1internal:generateContent", {"model": MODEL, "request": {"contents": []}}),
        ("/v1internal:countTokens", {"model": MODEL, "request": {"contents": []}}),
        ("/v1internal:fetchAvailableModels", {}),
    ],
)
def test_unverified_fixtures_fail_before_cli_dispatch(
    serve: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    payload: dict[str, Any],
) -> None:
    monkeypatch.delenv("SIDECAR_FIXTURES_VERIFIED", raising=False)
    monkeypatch.setenv("FAKE_CLI_DIR", str(tmp_path))
    port = serve(_config(_write_fake_cli(tmp_path)))

    status, headers, body = _request(port, "POST", path, payload)

    assert status == 503
    assert headers["x-sidecar-pre-dispatch"] == "1"
    assert json.loads(body)["error"]
    assert not (tmp_path / "spawned").exists()


def test_unknown_model_is_rejected_before_cli_dispatch(
    serve: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SIDECAR_FIXTURES_VERIFIED", raising=False)
    monkeypatch.setenv("FAKE_CLI_DIR", str(tmp_path))
    port = serve(_config(_write_fake_cli(tmp_path)))

    status, headers, _body = _request(
        port,
        "POST",
        "/v1internal:streamGenerateContent",
        {"model": "unknown-model", "request": {"contents": []}},
    )

    assert status == 422
    assert headers["x-sidecar-pre-dispatch"] == "1"
    assert not (tmp_path / "spawned").exists()


def test_pinned_version_mismatch_exits_two(tmp_path: Path) -> None:
    fake_cli = _write_fake_cli(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "cloudcode_claude_code": {
                    "claude_bin": str(fake_cli),
                    "claude_bin_pinned_version": "2.1.322",
                    "model_mapping": {MODEL: TARGET_MODEL},
                }
            }
        ),
        encoding="utf-8",
    )
    env = dict(os.environ, FAKE_CLI_VERSION="2.1.293", FAKE_CLI_DIR=str(tmp_path))

    completed = subprocess.run(
        [sys.executable, str(SIDECAR_SCRIPT), "--config", str(config_path), "--port", "0"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
        env=env,
    )

    assert completed.returncode == 2
    assert not (tmp_path / "spawned").exists()
    assert "2.1.293" not in completed.stderr


def test_verified_env_reaches_only_skeleton_501_boundary(
    serve: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SIDECAR_FIXTURES_VERIFIED", "1")
    monkeypatch.setenv("FAKE_CLI_DIR", str(tmp_path))
    port = serve(_config(_write_fake_cli(tmp_path)))

    status, headers, _body = _request(
        port,
        "POST",
        "/v1internal:streamGenerateContent",
        {"model": MODEL, "request": {"contents": []}},
    )

    assert status == 501
    assert headers["x-sidecar-pre-dispatch"] == "1"
    assert not (tmp_path / "spawned").exists()


def _decode_chunked(body: bytes) -> bytes:
    decoded = bytearray()
    while body:
        size_line, body = body.split(b"\r\n", 1)
        size = int(size_line.split(b";", 1)[0], 16)
        if size == 0:
            break
        decoded.extend(body[:size])
        if body[size:] == b"\r":  # The partial capture ends mid chunk delimiter.
            break
        assert body[size : size + 2] == b"\r\n"
        body = body[size + 2 :]
    return bytes(decoded)


def test_open01_sse_golden_and_keepalive_reframing() -> None:
    # The sanitized capture stores carriage returns as the two visible characters "\\r".
    raw = FIXTURE.read_bytes().replace(b"\\r\n", b"\r\n").replace(b"\\r", b"\r")
    head, chunked_body = raw.split(b"\r\n\r\n", 1)
    assert head.startswith(b"HTTP/1.1 200 OK\r\n")
    sse = _decode_chunked(chunked_body)

    frames = parse_sse_frames(sse + b": keepalive\r\n\r\n")

    assert len(frames) == 2
    assert frames[0].payload == {
        "response": {
            "candidates": [{"content": {"role": "model", "parts": [{"text": "PONG"}]}}],
            "usageMetadata": {
                "promptTokenCount": 15442,
                "candidatesTokenCount": 2,
                "totalTokenCount": 15509,
                "thoughtsTokenCount": 65,
            },
            "modelVersion": "gemini-4-neon",
            "responseId": "jpEDa8jTC7nPnsEPyr790AI",
        },
        "traceId": "e4cf30273ec3d9f7",
        "metadata": {},
    }
    reframed = reframe_sse_frame(frames[0])
    assert reframed.startswith(b'data: {"response":')
    assert reframed.endswith(b"\n\n")
    assert json.loads(reframed.removeprefix(b"data: ")) == frames[0].payload
    assert frames[1].keepalive
    assert reframe_sse_frame(frames[1]) == b": keepalive\n\n"


def test_cli_isolation_timeout_and_process_group_kill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cli = _write_fake_cli(tmp_path)
    config = _config(fake_cli)
    monkeypatch.setenv("FAKE_CLI_DIR", str(tmp_path))
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_FOUNDRY_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_OAUTH_ACCESS_TOKEN",
        "CLOUDSDK_AUTH_ACCESS_TOKEN",
    ):
        monkeypatch.setenv(name, "must-not-reach-cli")

    process = spawn_cli(config, TARGET_MODEL, base_url="http://127.0.0.1:45678/test-token")
    deadline = time.monotonic() + 3
    while not (tmp_path / "pid").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert (tmp_path / "pid").exists()
    pid = int((tmp_path / "pid").read_text(encoding="utf-8"))
    argv = json.loads((tmp_path / "argv.json").read_text(encoding="utf-8"))
    assert argv[argv.index("--model") + 1] == TARGET_MODEL
    for flag, value in (
        ("--tools", ""),
        ("--permission-mode", "dontAsk"),
        ("--max-turns", "1"),
        ("--setting-sources", ""),
    ):
        assert argv[argv.index(flag) + 1] == value
    for flag in ("-p", "--no-session-persistence", "--output-format", "--include-partial-messages"):
        assert flag in argv
    assert json.loads((tmp_path / "leaked-env.json").read_text(encoding="utf-8")) == []
    assert (tmp_path / "base-url.txt").read_text(encoding="utf-8") == "http://127.0.0.1:45678/test-token"

    with pytest.raises(CliTimeoutError):
        wait_for_cli(process, 0.05, grace_seconds=0.1)

    assert process.poll() is not None
    assert process.returncode in {-signal.SIGTERM, -signal.SIGKILL}
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
