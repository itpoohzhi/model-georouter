"""Регрессии скользящего таймаута, burst-допуска и увеличенных лимитов."""

from __future__ import annotations

import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import Mock, call

import pytest

from model_georouter import config_guard
from model_georouter import relay as relay_module
from model_georouter.config import BridgeConfig, ServerConfig
from model_georouter.egress_admission import EgressAdmission
from model_georouter.http_wire import LengthFraming
from model_georouter.relay import COMPLETE, INACTIVITY_TIMEOUT, SSERelay
from model_georouter.retry_budget import AttemptEvent, AttemptState, _should_penalize


def test_sliding_timeout_on_stream_relay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Каждый чанк сбрасывает простой и вызывает callback после передачи."""
    upstream, client = Mock(), Mock()
    chunks = [b"data: first\n\n", b"data: second\n\n", b"data: [DONE]\n\n"]
    clock = [0.0]
    forwarded: list[bytes] = []
    callback_snapshots: list[tuple[bytes, ...]] = []
    upstream.recv.side_effect = chunks

    def readable(*args: object) -> tuple[list[Mock], list[object], list[object]]:
        clock[0] += 4.0
        return [upstream], [], []

    def send(data: memoryview) -> int:
        forwarded.append(bytes(data))
        return len(data)

    client.send.side_effect = send
    monkeypatch.setattr(relay_module.select, "select", readable)
    monkeypatch.setattr(relay_module.time, "monotonic", lambda: clock[0])
    on_chunk = Mock(side_effect=lambda: callback_snapshots.append(tuple(forwarded)))
    relay = SSERelay(inactivity_timeout=5.0, write_timeout=5.0)

    result = relay.relay(upstream, client, LengthFraming(sum(map(len, chunks))), on_chunk=on_chunk)

    assert result.outcome == COMPLETE
    assert result.ok
    assert result.bytes_sent == sum(map(len, chunks))
    assert forwarded == chunks
    assert callback_snapshots == [tuple(chunks[:i]) for i in range(1, len(chunks) + 1)]
    assert on_chunk.call_args_list == [call()] * len(chunks)
    assert clock[0] == 12.0  # Общая длительность превышает таймаут, каждый простой — нет.
    assert client.settimeout.call_args_list == [call(5.0)] * len(chunks)
    upstream.shutdown.assert_called_once_with(socket.SHUT_RDWR)
    upstream.close.assert_called_once_with()


def test_stream_relay_times_out_without_further_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Скользящий таймаут не отключает отказ при реальном простое."""
    upstream, client = Mock(), Mock()
    clock = [0.0]

    def readable(*args: object) -> tuple[list[object], list[object], list[object]]:
        clock[0] += 6.0
        return [], [], []

    monkeypatch.setattr(relay_module.select, "select", readable)
    monkeypatch.setattr(relay_module.time, "monotonic", lambda: clock[0])
    on_chunk = Mock()
    result = SSERelay(inactivity_timeout=5.0, write_timeout=5.0).relay(
        upstream, client, LengthFraming(1), on_chunk=on_chunk
    )
    assert result.outcome == INACTIVITY_TIMEOUT
    assert result.bytes_sent == 0
    on_chunk.assert_not_called()
    client.send.assert_not_called()
    upstream.close.assert_called_once_with()


def test_no_proxy_penalize_on_long_header_timeout() -> None:
    """Длинная голова не штрафует multi-proxy, но сохраняет остальные правила."""
    state = AttemptState(pool="route-test", single_proxy=False, is_long_header=True)
    event = AttemptEvent("header_timeout", has_endpoint=True)
    assert _should_penalize(state, event) is False
    assert _should_penalize(replace(state, is_long_header=False), event) is True
    assert _should_penalize(replace(state, single_proxy=True), event) is True
    assert _should_penalize(state, replace(event, has_endpoint=False)) is False


@pytest.mark.parametrize("limit_kind", ["backend", "long_header"])
def test_burst_admission_wait_and_release(monkeypatch: pytest.MonkeyPatch, limit_kind: str) -> None:
    """Ожидающий поток получает слот после exit_slot для обоих видов квот."""
    admission = EgressAdmission(
        proxy_slots=4,
        max_inflight_per_backend=1 if limit_kind == "backend" else 2,
        max_inflight_long_header=1,
        admission_wait_s=0.5,
    )
    path = "/streamGenerateContent"
    waiting = threading.Event()
    acquired = threading.Event()
    original_wait = admission._cond.wait

    def observed_wait(timeout: float | None = None) -> bool:
        waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(admission._cond, "wait", observed_wait)
    prev, ticket = admission.enter_slot("route-test", path, None, None, None, None)
    released = False

    def worker() -> str:
        with admission.admit("route-test", path) as admitted:
            acquired.set()
            assert admission.inflight_for(admitted.egress_key) == 1
            assert admission.inflight_long_header == 1
            return admitted.egress_key

    try:
        assert admission.admission_wait_s == 0.5
        assert admission.inflight_for(ticket.egress_key) == 1
        assert admission.inflight_long_header == 1
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(worker)
            try:
                assert waiting.wait(timeout=2.0), "worker did not enter the admission wait"
                assert not acquired.is_set()
                assert not future.done()
            finally:
                admission.exit_slot(ticket, prev)
                released = True
            assert future.result(timeout=2.0) == ticket.egress_key
            assert acquired.is_set()
    finally:
        if not released:
            admission.exit_slot(ticket, prev)
    assert admission.inflight_for(ticket.egress_key) == 0
    assert admission.inflight_long_header == 0
    assert admission.current().pool == ""


def test_sizing_configuration_m5_max() -> None:
    """Профиль M5 Max проходит проверку снимка и полный разбор конфигурации."""
    server = ServerConfig(
        listen="127.0.0.1",
        port=10830,
        proxy_slots=256,
        direct_slots=256,
        max_connections=1024,
        max_inflight_per_backend=128,
        max_inflight_long_header=64,
        headers_timeout=300.0,
        body_buffer_max_bytes=67108864,
        total_egress_deadline=40.0,
    )
    bridge = BridgeConfig(server, {}, {}, {}, ())
    guarded = config_guard.validate(bridge)
    assert guarded.bridge is bridge
    assert guarded.max_inflight_per_backend == 128
    assert guarded.max_inflight_long_header == 64

    sizing_fields = (
        "proxy_slots",
        "direct_slots",
        "max_connections",
        "max_inflight_per_backend",
        "max_inflight_long_header",
        "headers_timeout",
        "body_buffer_max_bytes",
    )
    raw_server = {field: getattr(server, field) for field in sizing_fields}
    raw_server.update(listen=server.listen, port=server.port, total_egress_deadline=40.0)
    parsed = config_guard.validate_config({"server": raw_server, "pools": {"direct": {"type": "direct"}}, "rules": []})
    for field in sizing_fields:
        assert getattr(parsed.bridge.server, field) == getattr(server, field)
