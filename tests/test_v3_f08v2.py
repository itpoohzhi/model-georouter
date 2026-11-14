"""Интеграционные тесты потокового бюджета F-08v2 и лимита long-header (W3-F)."""

from __future__ import annotations

import asyncio

import pytest

from model_georouter.egress_admission import (
    DEFAULT_MAX_INFLIGHT_LONG_HEADER,
    EgressAdmission,
    InflightLimitError,
    Ticket,
)
from model_georouter.errors import EgressTimeoutError

STREAM_PATH = "/v1beta/models/gemini-2.5-pro:streamGenerateContent"
NON_STREAM_PATH = "/v1beta/models/gemini-2.5-pro:generateContent"
POOLS = {"de": {"egress_id": "eg-de-1", "proxies": ["http://127.0.0.1:8081"]}}


def _step_stream_chunk(ticket: Ticket, *, now: float) -> str:
    """Пробрасывает очередной чанк потока или возвращает/выбрасывает upstream_timeout при паузе > 150s."""
    if ticket.deadline.expired(now=now):
        raise EgressTimeoutError("upstream_timeout: stream idle timeout exceeded", phase="upstream_timeout")
    ticket.on_chunk(now=now)
    return "chunk_ok"


def test_sliding_timer_resets_on_each_chunk_over_169s() -> None:
    """Поток с паузами < 150s и суммарным временем > 169s успешно пробрасывается под v2."""
    adm = EgressAdmission(POOLS, f08_policy="v2", client_timeout_s=180.0)
    t0 = 1000.0

    # Паузы между событиями: 18s, 110s, 120s, 115s (каждая < 150s, сумма = 363s > 169s)
    chunk_timestamps = [t0 + 18.0, t0 + 128.0, t0 + 248.0, t0 + 363.0]

    with adm.admit("de", STREAM_PATH, started_at=t0, now=t0) as ticket:
        assert ticket.is_long_header is True
        assert ticket.t_pre_s == pytest.approx(19.0)
        assert ticket.sliding_idle_s == pytest.approx(150.0)

        for now in chunk_timestamps:
            assert ticket.deadline.expired(now=now) is False
            # Фазовый таймаут чтения укладывается в остаток скользящего окна
            assert ticket.deadline.clamp(10.0, now=now) == pytest.approx(10.0)
            _step_stream_chunk(ticket, now=now)
            assert ticket.header_deadline_s == pytest.approx(150.0)
            assert ticket.deadline.remaining(now=now) == pytest.approx(150.0)

        total_elapsed = chunk_timestamps[-1] - t0
        assert total_elapsed > 169.0
        assert ticket.deadline.expired(now=chunk_timestamps[-1] + 149.0) is False


def test_single_pause_over_150s_fails_with_upstream_timeout() -> None:
    """Одиночная пауза > 150s приводит к истечению скользящего бюджета и upstream_timeout."""
    adm = EgressAdmission(POOLS, f08_policy="v2", client_timeout_s=180.0)
    t0 = 2000.0

    with adm.admit("de", STREAM_PATH, started_at=t0, now=t0) as ticket:
        # Первые два чанка приходят с допустимыми паузами 15s и 90s
        _step_stream_chunk(ticket, now=t0 + 15.0)
        _step_stream_chunk(ticket, now=t0 + 105.0)

        # Следующий чанк задерживается на 150.5s (> 150s)
        stalled_at = (t0 + 105.0) + 150.5
        assert ticket.deadline.expired(now=stalled_at) is True
        with pytest.raises(TimeoutError, match="egress deadline expired"):
            ticket.deadline.clamp(1.0, now=stalled_at)

        with pytest.raises(EgressTimeoutError) as exc_info:
            _step_stream_chunk(ticket, now=stalled_at)

        assert exc_info.value.phase == "upstream_timeout"
        assert "upstream_timeout" in str(exc_info.value)


@pytest.mark.parametrize("header_delay_s", [40.5, 65.0, 120.0, 149.9])
def test_stream_generate_content_headers_after_40s_forwarded_under_v2(
    header_delay_s: float,
) -> None:
    """Заголовки на streamGenerateContent после 40s (в пределах 150s) пробрасываются под v2."""
    adm_v2 = EgressAdmission(
        POOLS,
        f08_policy="v2",
        f08_v2_risk_ack="AD-003/2026-11-14",
    )
    t0 = 500.0
    t_headers = t0 + header_delay_s

    # Под v2 на streamGenerateContent заголовки после 40s (<= 150s) успешно принимаются
    with adm_v2.admit("de", STREAM_PATH, started_at=t0, now=t_headers) as ticket:
        assert ticket.is_long_header is True
        assert ticket.deadline.expired(now=t_headers) is False
        assert ticket.header_deadline_s == pytest.approx(150.0 - header_delay_s)
        # При получении заголовков/первого чанка окно сбрасывается на полные 150s
        ticket.on_chunk(now=t_headers)
        assert ticket.deadline.remaining(now=t_headers) == pytest.approx(150.0)

    # На не-streaming пути под v2 и на любом пути под v1 задержка > 40s вызывает EgressTimeoutError
    with (
        pytest.raises(EgressTimeoutError) as non_stream_exc,
        adm_v2.admit("de", NON_STREAM_PATH, started_at=t0, now=t_headers),
    ):
        pass
    assert non_stream_exc.value.phase == "admission"

    adm_v1 = EgressAdmission(POOLS, f08_policy="v1")
    with (
        pytest.raises(EgressTimeoutError) as v1_exc,
        adm_v1.admit("de", STREAM_PATH, started_at=t0, now=t_headers),
    ):
        pass
    assert v1_exc.value.phase == "admission"


def test_max_inflight_long_header_limit_enforced() -> None:
    """Проверяет соблюдение квоты max_inflight_long_header и освобождение слотов."""
    assert DEFAULT_MAX_INFLIGHT_LONG_HEADER == 24
    adm = EgressAdmission(
        POOLS,
        proxy_slots=20,
        max_inflight_per_backend=10,
        max_inflight_long_header=2,
        f08_policy="v2",
        client_timeout_s=180.0,
    )

    with adm.admit("de", STREAM_PATH) as t1, adm.admit("de", STREAM_PATH) as t2:
        assert t1.is_long_header is True
        assert t2.is_long_header is True
        assert adm.inflight_long_header == 2
        assert adm.inflight_for("eg-de-1") == 2

        # 3-й параллельный streamGenerateContent отклоняется по лимиту long_header
        with pytest.raises(InflightLimitError) as exc_info, adm.admit("de", STREAM_PATH):
            pass
        assert exc_info.value.status_code == 503
        assert exc_info.value.outcome == "inflight_limit"
        assert exc_info.value.limit == 2

        # Обычный (не-streaming) запрос при этом проходит, т.к. backend-квота (2 < 10) свободна
        with adm.admit("de", NON_STREAM_PATH) as short_ticket:
            assert short_ticket.is_long_header is False
            assert adm.inflight_long_header == 2
            assert adm.inflight_for("eg-de-1") == 3

    # После выхода из контекстов счётчики сбрасываются и async-вход снова доступен
    assert adm.inflight_long_header == 0
    assert adm.inflight_for("eg-de-1") == 0

    async def _verify_async_admit() -> None:
        async with adm.admit("de", STREAM_PATH) as async_ticket:
            assert async_ticket.is_long_header is True
            assert adm.inflight_long_header == 1

    asyncio.run(_verify_async_admit())
    assert adm.inflight_long_header == 0
