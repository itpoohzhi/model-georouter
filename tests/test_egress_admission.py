"""Тесты контроля входа egress-запросов (W1-C)."""

from __future__ import annotations

import asyncio
import json
import math
import time
from contextlib import AsyncExitStack

import pytest

from model_georouter import egress_admission as _ea
from model_georouter.config import BridgeConfig, ServerConfig
from model_georouter.config_guard import validate_config
from model_georouter.egress_admission import EgressAdmission, InflightLimitError, Ticket
from model_georouter.errors import ConfigError, EgressTimeoutError, SlotsExhaustedError
from model_georouter.model_router import sanitize_openai_empty_messages
from model_georouter.retry_budget import F08_V2_RISK_ACK, EgressDeadline, RetryBudgetSpec, validate_f08_v2

DE_EGRESS_ID = "de-213.165.42.220"
DE_PORTS = ("127.0.0.1:10827", "127.0.0.1:10828", "127.0.0.1:10829")
GENERATE_PATH = "/v1beta/models/gemini-2.5-pro:generateContent"
STREAM_PATH = "/v1beta/models/gemini-2.5-pro:streamGenerateContent"


def _source(**fields: object) -> BridgeConfig:
    """Создаёт снимок с независимыми полями источника допуска."""
    return BridgeConfig(
        ServerConfig(
            listen="127.0.0.1",
            port=0,
            proxy_slots=80,
            total_egress_deadline=40,
            f08_policy=fields.get("f08_policy", "v2"),
            f08_v2_risk_ack=fields.get("f08_v2_risk_ack"),
            client_timeout_s=fields.get("client_timeout_s"),
        ),
        {},
        {},
        {},
        (),
    )


def test_explicit_none_ack_overrides_source() -> None:
    """Явный None отменяет подтверждение риска в источнике."""
    with pytest.raises(ConfigError):
        _ = EgressAdmission(config=_source(f08_v2_risk_ack=F08_V2_RISK_ACK), f08_v2_risk_ack=None)


def test_explicit_none_policy_overrides_source() -> None:
    """Явный None политики не подменяется значением источника."""
    with pytest.raises(ConfigError, match="f08_policy"):
        _ = EgressAdmission(config=_source(client_timeout_s=180), f08_policy=None)
    assert EgressAdmission(config=_source(f08_policy=None)).f08_policy == "v1"


def test_explicit_none_timeout_overrides_source() -> None:
    """Явный None отменяет клиентский таймаут в источнике."""
    with pytest.raises(ConfigError):
        _ = EgressAdmission(config=_source(client_timeout_s=180), client_timeout_s=None)


@pytest.mark.parametrize("policy", ["", "bad", 42])
def test_source_invalid_policy_rejected(policy: object) -> None:
    """Некорректная политика источника не подменяется дефолтом."""
    with pytest.raises(ConfigError):
        _ = EgressAdmission(config=_source(f08_policy=policy))


@pytest.mark.parametrize("timeout", ["bad", True])
def test_source_invalid_timeout_rejected(timeout: object) -> None:
    """Таймаут проверяется даже при действующем подтверждении риска."""
    with pytest.raises(ConfigError, match="client_timeout_s"):
        _ = EgressAdmission(config=_source(f08_v2_risk_ack=F08_V2_RISK_ACK, client_timeout_s=timeout))


def test_source_invalid_ack_rejected() -> None:
    """Число вместо подтверждения риска отклоняется и при достаточном таймауте."""
    with pytest.raises(ConfigError, match="f08_v2_risk_ack"):
        _ = EgressAdmission(config=_source(f08_v2_risk_ack=42, client_timeout_s=180))


def test_hedging_omission_is_fail_safe() -> None:
    """Пропуск ключа не включает hedging ни в снимке, ни в конфиге v2."""
    assert ServerConfig("127.0.0.1", 0).hedged_connect_enabled is False
    cfg: dict[str, object] = {
        "server": {
            "listen": "127.0.0.1",
            "port": 0,
            "proxy_slots": 80,
            "total_egress_deadline": 40,
            "f08_policy": "v2",
            "client_timeout_s": 180,
        },
        "pools": {},
        "rules": [],
    }
    assert validate_config(cfg).bridge.server.hedged_connect_enabled is False


@pytest.mark.parametrize("flag", ["false", 1, None, True])
def test_hedging_v2_rejects_unsafe_or_non_boolean_flag(flag: object) -> None:
    """В v2 допускается только явно булев выключенный hedging."""
    with pytest.raises(ConfigError, match="hedged_connect_enabled"):
        _ = validate_config(
            {
                "server": {
                    "listen": "127.0.0.1",
                    "port": 0,
                    "proxy_slots": 80,
                    "total_egress_deadline": 40,
                    "f08_policy": "v2",
                    "client_timeout_s": 180,
                    "hedged_connect_enabled": flag,
                },
                "pools": {},
                "rules": [],
            }
        )


@pytest.mark.parametrize("content", ["", None])
def test_openai_empty_user_normalized(content: object) -> None:
    """Обычное пустое сообщение заменяется пробелом."""
    raw = json.dumps({"messages": [{"role": "user", "content": content}]}).encode()
    assert sanitize_openai_empty_messages(raw) == b'{"messages":[{"role":"user","content":" "}]}'


@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-test", "type": "function"}]},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-test", "type": "function"}]},
        {"role": "assistant", "content": None, "function_call": {"name": "example", "arguments": "{}"}},
        {"role": "assistant", "content": "", "function_call": {"name": "example", "arguments": "{}"}},
        {"role": "tool", "content": "", "tool_call_id": "call-test", "extra": {"keep": True}},
        {"role": "tool", "content": None, "tool_call_id": "call-test"},
    ],
)
def test_openai_tool_structure_preserved(message: dict[str, object]) -> None:
    """Вызовы и ответы инструментов сохраняют исходные байты и структуру."""
    raw = json.dumps({"messages": [message]}).encode()
    assert sanitize_openai_empty_messages(raw) is raw


def test_openai_tool_structure_preserved_beside_normalized_user() -> None:
    """Изменение соседнего сообщения не затрагивает поля вызова и ответа инструмента."""
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-test", "type": "function"}]},
        {"role": "tool", "content": "", "tool_call_id": "call-test", "extra": {"keep": True}},
        {"role": "user", "content": ""},
    ]
    raw = json.dumps({"messages": messages}).encode()
    expected = json.dumps(
        {"messages": [*messages[:2], {"role": "user", "content": " "}]}, separators=(",", ":")
    ).encode()
    assert sanitize_openai_empty_messages(raw) == expected


def _make_de_pools() -> dict[str, object]:
    return {
        "route-de": {"type": "socks5", "egress_id": DE_EGRESS_ID, "proxies": [f"socks5://{hp}" for hp in DE_PORTS]},
        "route-us": {"type": "socks5", "proxies": ["socks5://127.0.0.1:10810"]},
    }


async def _expect_inflight_limit(
    admission: EgressAdmission, pool: str, path: str, backend: str
) -> tuple[InflightLimitError, float]:
    t0 = time.monotonic()
    try:
        async with admission.admit(pool, path, backend=backend):
            pass
    except InflightLimitError as err:
        return err, (time.monotonic() - t0) * 1000.0
    raise AssertionError("expected InflightLimitError")


def test_route_de_three_ports_share_one_egress_counter() -> None:
    """Все 3 локальных порта DE делят один счётчик по `egress_id` (<= 40), 41-й сразу получает 503."""

    async def _run() -> None:
        admission = EgressAdmission(_make_de_pools())
        assert admission.max_inflight_per_backend == _ea.DEFAULT_MAX_INFLIGHT_PER_BACKEND == 40
        async with AsyncExitStack() as stack:
            tickets: list[Ticket] = []
            for idx in range(40):
                port_backend = DE_PORTS[idx % len(DE_PORTS)]
                ticket = await stack.enter_async_context(
                    admission.admit("route-de", GENERATE_PATH, backend=port_backend)
                )
                assert ticket.egress_key == DE_EGRESS_ID and ticket.egress_id == DE_EGRESS_ID
                tickets.append(ticket)
            assert len(tickets) == 40 and admission.inflight_for(DE_EGRESS_ID) == 40

            for candidate_backend in DE_PORTS:
                err, elapsed_ms = await _expect_inflight_limit(admission, "route-de", GENERATE_PATH, candidate_backend)
                assert isinstance(err, SlotsExhaustedError)
                assert err.status_code == 503 and err.outcome == "inflight_limit" and err.egress_key == DE_EGRESS_ID
                assert elapsed_ms < 50.0
        assert admission.inflight_for(DE_EGRESS_ID) == 0

    asyncio.run(_run())


def test_fallback_to_host_port_when_egress_id_missing() -> None:
    """При отсутствии `egress_id` ключ счётчика берётся по `host:port`."""

    async def _run() -> None:
        admission = EgressAdmission(_make_de_pools(), max_inflight_per_backend=2)
        async with AsyncExitStack() as stack:
            t1 = await stack.enter_async_context(
                admission.admit("route-us", GENERATE_PATH, backend="socks5://127.0.0.1:10810")
            )
            t2 = await stack.enter_async_context(admission.admit("route-us", GENERATE_PATH, backend="127.0.0.1:10810"))
            assert t1.egress_key == "127.0.0.1:10810" and t2.egress_key == "127.0.0.1:10810"
            err, _ = await _expect_inflight_limit(admission, "route-us", GENERATE_PATH, "127.0.0.1:10810")
            assert err.status_code == 503 and err.outcome == "inflight_limit"
            async with admission.admit("route-us", GENERATE_PATH, backend="127.0.0.1:10811") as other:
                assert other.egress_key == "127.0.0.1:10811"

    asyncio.run(_run())


def test_max_inflight_long_header_limit() -> None:
    """Лимит `streamGenerateContent` (по умолчанию 24) срабатывает независимо от запаса backend-слотов."""

    async def _run() -> None:
        admission = EgressAdmission(_make_de_pools())
        assert admission.max_inflight_long_header == _ea.DEFAULT_MAX_INFLIGHT_LONG_HEADER == 24
        async with AsyncExitStack() as stack:
            for idx in range(24):
                ticket = await stack.enter_async_context(
                    admission.admit("route-de", STREAM_PATH, backend=DE_PORTS[idx % 3])
                )
                assert ticket.is_long_header is True
            assert admission.inflight_long_header == 24 and admission.inflight_for(DE_EGRESS_ID) == 24
            err, _ = await _expect_inflight_limit(admission, "route-de", STREAM_PATH, DE_PORTS[0])
            assert err.status_code == 503 and err.outcome == "inflight_limit"
            assert admission.inflight_for(DE_EGRESS_ID) == 24
            async with admission.admit("route-de", GENERATE_PATH, backend=DE_PORTS[0]) as short_ticket:
                assert short_ticket.is_long_header is False and admission.inflight_for(DE_EGRESS_ID) == 25
        assert admission.inflight_long_header == 0 and admission.inflight_for(DE_EGRESS_ID) == 0

    asyncio.run(_run())


def test_slot_released_on_exception_and_cancelled_error() -> None:
    """Слот гарантированно освобождается при исключении и при `asyncio.CancelledError`."""

    async def _raise_inside(admission: EgressAdmission) -> None:
        async with admission.admit("route-de", STREAM_PATH, backend=DE_PORTS[0]):
            assert admission.inflight_for(DE_EGRESS_ID) == 1 and admission.inflight_long_header == 1
            raise RuntimeError("boom")

    async def _run() -> None:
        admission = EgressAdmission(_make_de_pools(), max_inflight_per_backend=1, max_inflight_long_header=1)
        caught = False
        try:
            await _raise_inside(admission)
        except RuntimeError:
            caught = True
        assert caught and admission.inflight_for(DE_EGRESS_ID) == 0 and admission.inflight_long_header == 0

        entered, hold = asyncio.Event(), asyncio.Event()

        async def _worker() -> None:
            async with admission.admit("route-de", STREAM_PATH, backend=DE_PORTS[1]):
                entered.set()
                _ = await hold.wait()

        task = asyncio.create_task(_worker())
        _ = await entered.wait()
        assert admission.inflight_for(DE_EGRESS_ID) == 1 and admission.inflight_long_header == 1
        _ = task.cancel()
        cancelled = False
        try:
            await task
        except asyncio.CancelledError:
            cancelled = True
        assert cancelled and admission.inflight_for(DE_EGRESS_ID) == 0 and admission.inflight_long_header == 0

    asyncio.run(_run())


def test_header_deadline_from_retry_budget_v1_and_policy_guard() -> None:
    """`Ticket.header_deadline_s` берёт остаток от 40s `retry_budget`, а `f08_policy != 'v1'` вызывает `ConfigError`."""

    async def _run() -> None:
        admission = EgressAdmission(_make_de_pools())
        deadline = EgressDeadline.from_timeout(40.0, started_at=100.0)
        async with admission.admit("route-de", GENERATE_PATH, deadline=deadline, now=107.5) as ticket:
            assert math.isclose(ticket.header_deadline_s, 32.5)
        async with admission.admit("route-de", GENERATE_PATH, started_at=200.0, now=200.0) as fresh:
            assert math.isclose(fresh.header_deadline_s, 40.0)
        expired_caught = False
        try:
            async with admission.admit("route-de", GENERATE_PATH, deadline=deadline, now=140.0):
                pass
        except EgressTimeoutError:
            expired_caught = True
        assert expired_caught is True
        for invalid_policy in ("v2", "legacy", ""):
            policy_caught = False
            try:
                _ = EgressAdmission(_make_de_pools(), f08_policy=invalid_policy)
            except ConfigError:
                policy_caught = True
            assert policy_caught is True
        budget_caught = False
        try:
            _ = EgressAdmission(_make_de_pools(), retry_budget=RetryBudgetSpec(total_egress_deadline=35.0))
        except ConfigError:
            budget_caught = True
        assert budget_caught is True

    asyncio.run(_run())


def test_f08_v2_validation_and_sliding_stream_budget() -> None:
    """Политика `v2` требует `validate_f08_v2` (client_timeout_s >= 180 или AD-003/2026-11-14) и даёт 150s sliding для stream."""

    async def _run() -> None:
        for bad_kwargs in (
            {"T_pre": 20.0, "f08_v2_risk_ack": F08_V2_RISK_ACK},
            {"T_idle": 120.0, "f08_v2_risk_ack": F08_V2_RISK_ACK},
            {"scope": "generateContent", "f08_v2_risk_ack": F08_V2_RISK_ACK},
            {"client_timeout_s": 179.0},
            {"client_timeout_s": None, "f08_v2_risk_ack": "wrong-ack"},
        ):
            caught = False
            try:
                _ = validate_f08_v2(**bad_kwargs)
            except ConfigError:
                caught = True
            assert caught is True

        v2_ct = validate_f08_v2(client_timeout_s=180.0)
        assert v2_ct.T_pre == 19.0 and v2_ct.T_idle == 150.0
        adm = EgressAdmission(_make_de_pools(), f08_policy="v2", f08_v2_risk_ack=F08_V2_RISK_ACK)
        async with adm.admit("route-de", STREAM_PATH, started_at=100.0, now=100.0) as st:
            assert st.t_pre_s == 19.0 and st.sliding_idle_s == 150.0 and math.isclose(st.header_deadline_s, 150.0)
            assert math.isclose(st.deadline.remaining(now=220.0), 30.0)
            _ = st.on_chunk(now=220.0)
            assert math.isclose(st.deadline.remaining(now=220.0), 150.0)
        async with adm.admit("route-de", GENERATE_PATH, started_at=100.0, now=100.0) as gt:
            assert gt.t_pre_s == 19.0 and gt.sliding_idle_s is None and math.isclose(gt.header_deadline_s, 40.0)
            _ = gt.on_chunk(now=120.0)
            assert math.isclose(gt.deadline.remaining(now=120.0), 20.0)

    asyncio.run(_run())
