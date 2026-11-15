"""Регрессии F-001/F-006 второго Совета: содержимое сообщений и клиентский таймаут."""

from __future__ import annotations

import json

import pytest

from model_georouter.egress_admission import EgressAdmission
from model_georouter.errors import ConfigError
from model_georouter.model_router import sanitize_openai_empty_messages


@pytest.mark.parametrize(
    "part",
    [
        {"type": "refusal", "refusal": "denied"},
        {"type": "thinking", "thinking": "reason"},
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "audio", "id": "audio-test"},
        {"type": "image_url", "image_url": {"url": "https://example.invalid/image"}},
        {"type": "video_url", "video_url": "https://example.invalid/video"},
        {"type": "document", "data": "opaque"},
        {"type": "future_block"},
        {},
        "opaque",
    ],
)
def test_council2_nontext_content_is_preserved(part: object) -> None:
    raw = json.dumps({"messages": [{"role": "assistant", "content": [part]}]}).encode()
    assert sanitize_openai_empty_messages(raw) is raw


@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "content": None, "audio": {"id": "audio-test"}},
        {"role": "assistant", "content": None, "audio": {}},
        {"role": "assistant", "content": None, "audio": None},
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "assistant", "content": None, "function_call": {}},
        {"role": "function", "name": "lookup", "content": ""},
        {"role": "tool", "content": ""},
    ],
)
def test_council2_nontext_message_is_preserved(message: dict[str, object]) -> None:
    raw = json.dumps({"messages": [message]}).encode()
    assert sanitize_openai_empty_messages(raw) is raw


def test_council2_empty_text_parts_are_still_empty() -> None:
    for content in ([], [{"type": "text", "text": "  "}, {"type": "input_text", "text": None}]):
        raw = json.dumps({"messages": [{"role": "user", "content": content}]}).encode()
        assert sanitize_openai_empty_messages(raw) == b'{"messages":[{"role":"user","content":" "}]}'
    raw = b'{"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}'
    assert sanitize_openai_empty_messages(raw) is raw


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), -1, 0, True, "180"])
@pytest.mark.parametrize("policy", ["v1", "v2"])
def test_council2_invalid_client_timeout_rejected(timeout: object, policy: str) -> None:
    with pytest.raises(ConfigError, match="client_timeout_s"):
        _ = EgressAdmission(client_timeout_s=timeout, f08_policy=policy, f08_v2_risk_ack="AD-003/2026-11-14")
