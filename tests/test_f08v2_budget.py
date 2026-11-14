"""Тесты контракта и валидации бюджета F-08v2 (W3-F)."""

from __future__ import annotations

import math

import pytest

from model_georouter.egress_admission import EgressAdmission
from model_georouter.errors import ConfigError
from model_georouter.retry_budget import (
    F08_V2_CLIENT_MARGIN,
    F08_V2_NON_STREAM_DEADLINE,
    F08_V2_POLICY,
    F08_V2_RISK_ACK,
    F08_V2_STREAM_MARKER,
    F08_V2_T_IDLE,
    F08_V2_T_PRE,
    EgressDeadline,
    F08V2Budget,
    validate_f08_v2,
)

STREAM_PATH = "/v1beta/models/gemini-2.5-pro:streamGenerateContent"
NON_STREAM_PATH = "/v1beta/models/gemini-2.5-pro:generateContent"


def test_f08_v2_constants_and_sliding_default() -> None:
    """Проверяет константы F-08v2: T_pre = 19s (3x3 + 10) и скользящий T_idle = 150s."""
    # 3 попытки TCP/TLS connect по 3 секунды + 10 секунд на backoff и первые заголовки
    assert F08_V2_T_PRE == 3 * 3 + 10 == 19.0
    assert F08_V2_T_IDLE == 150.0
    assert F08_V2_NON_STREAM_DEADLINE == 40.0
    assert F08_V2_CLIENT_MARGIN == 11.0
    assert F08_V2_POLICY == "v2"
    assert F08_V2_STREAM_MARKER == "streamGenerateContent"
    assert F08_V2_RISK_ACK == "AD-003/2026-11-14"

    budget = F08V2Budget()
    assert budget.t_pre == 19.0
    assert budget.T_pre == 19.0
    assert budget.t_idle == 150.0
    assert budget.T_idle == 150.0
    assert budget.non_stream_deadline == 40.0
    assert budget.scope == "streamGenerateContent"

    # Скользящий дедлайн по умолчанию сдвигает окно ровно на T_idle = 150s
    initial = EgressDeadline.from_timeout(19.0, started_at=100.0)
    slid = initial.slide(now=112.0)
    assert slid.remaining(now=112.0) == pytest.approx(150.0)
    assert slid.remaining(now=262.0) == pytest.approx(0.0)
    assert slid.expired(now=262.0) is True


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"t_pre": 18.0, "client_timeout_s": 180.0}, "T_pre must be 19s"),
        ({"T_pre": 20.0, "client_timeout_s": 180.0}, "T_pre must be 19s"),
        ({"t_idle": 149.0, "client_timeout_s": 180.0}, "T_idle must be 150s"),
        ({"T_idle": 160.0, "client_timeout_s": 180.0}, "T_idle must be 150s"),
        ({"t_pre": True, "client_timeout_s": 180.0}, "T_pre must be 19s"),
        ({"scope": "generateContent", "client_timeout_s": 180.0}, "f08_v2.scope"),
    ],
)
def test_f08_v2_rejects_modified_t_pre_t_idle_or_scope(
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Отклоняет любые отклонения констант T_pre=19s, T_idle=150s и маркера потока."""
    with pytest.raises(ConfigError, match=match):
        _ = validate_f08_v2(**kwargs)


def test_non_streaming_paths_preserve_40s_deadline() -> None:
    """Не-streaming пути сохраняют жёсткий бюджет 40s даже при активной политике v2."""
    budget = validate_f08_v2(client_timeout_s=180.0)

    dl_non_stream, t_pre, sliding_idle = budget.resolve_for_path(
        NON_STREAM_PATH,
        started_at=500.0,
    )
    assert dl_non_stream.remaining(now=500.0) == pytest.approx(40.0)
    assert dl_non_stream.expired(now=540.0) is True
    assert t_pre == 19.0
    assert sliding_idle is None

    dl_stream, t_pre_stream, sliding_stream = budget.resolve_for_path(
        STREAM_PATH,
        started_at=500.0,
    )
    assert dl_stream.remaining(now=500.0) == pytest.approx(150.0)
    assert t_pre_stream == 19.0
    assert sliding_stream == 150.0

    # Проверка через EgressAdmission: билет не-streaming пути получает 40s и не скользит
    adm = EgressAdmission(
        {"de": {"egress_id": "eg-de", "proxies": ["http://127.0.0.1:8080"]}},
        f08_policy="v2",
        client_timeout_s=180.0,
    )
    with adm.admit("de", NON_STREAM_PATH, started_at=10.0, now=10.0) as ticket:
        assert ticket.is_long_header is False
        assert ticket.header_deadline_s == pytest.approx(40.0)
        assert ticket.sliding_idle_s is None
        assert ticket.deadline.remaining(now=10.0) == pytest.approx(40.0)
        _ = ticket.on_chunk(now=35.0)
        assert ticket.deadline.remaining(now=35.0) == pytest.approx(15.0)

    # Попытка переопределить non_stream_deadline != 40s отклоняется
    with pytest.raises(ConfigError, match="non_stream_deadline"):
        _ = validate_f08_v2(client_timeout_s=180.0, non_stream_deadline=60.0)
    with pytest.raises(ConfigError, match="non_stream_deadline"):
        _ = EgressAdmission(f08_policy="v2", client_timeout_s=180.0, total_egress_deadline=60.0)


@pytest.mark.parametrize("valid_timeout", [180.0, 180, 185.5, 300.0])
def test_client_timeout_invariant_accepted_when_margin_satisfied(
    valid_timeout: float,
) -> None:
    """Принимает заданный client_timeout_s при выполнении T_pre + T_idle <= client_timeout_s - 11."""
    budget = validate_f08_v2(client_timeout_s=valid_timeout)
    assert budget.client_timeout_s == float(valid_timeout)
    assert budget.t_pre + budget.t_idle <= float(valid_timeout) - 11.0

    adm = EgressAdmission(f08_policy="v2", client_timeout_s=valid_timeout)
    assert adm.f08_v2_budget is not None
    assert adm.f08_v2_budget.client_timeout_s == float(valid_timeout)


@pytest.mark.parametrize(
    "invalid_timeout",
    [179.99, 179.0, 169.0, 150.0, 40.0, 0.0, -10.0, math.nan, math.inf, True, "180"],
)
def test_client_timeout_invariant_violation_raises_config_error(
    invalid_timeout: object,
) -> None:
    """Отклоняет client_timeout_s, если нарушен инвариант T_pre + T_idle <= client_timeout_s - 11."""
    with pytest.raises(ConfigError, match="client_timeout_s"):
        _ = validate_f08_v2(client_timeout_s=invalid_timeout)

    with pytest.raises(ConfigError, match="client_timeout_s"):
        _ = EgressAdmission(
            f08_policy="v2",
            # Неверный тип намеренно подаётся для проверки отказа runtime-валидатора.
            client_timeout_s=invalid_timeout,
        )


@pytest.mark.parametrize(
    "bad_ack",
    [None, "", "AD-003", "AD-003/2026-11-13", "true", 123],
)
def test_missing_client_timeout_requires_valid_risk_ack(bad_ack: object) -> None:
    """Отсутствующий client_timeout_s без точного f08_v2_risk_ack отклоняется с ConfigError."""
    with pytest.raises(ConfigError, match="f08_v2_risk_ack"):
        _ = validate_f08_v2(client_timeout_s=None, f08_v2_risk_ack=bad_ack)

    with pytest.raises(ConfigError, match="f08_v2_risk_ack"):
        _ = EgressAdmission(
            f08_policy="v2",
            client_timeout_s=None,
            # Проверяем в том числе объекты, запрещённые статической сигнатурой.
            f08_v2_risk_ack=bad_ack,
        )


def test_missing_client_timeout_accepted_with_exact_risk_ack() -> None:
    """Отсутствующий client_timeout_s принимается при f08_v2_risk_ack = 'AD-003/2026-11-14'."""
    budget = validate_f08_v2(
        client_timeout_s=None,
        f08_v2_risk_ack="AD-003/2026-11-14",
    )
    assert budget.client_timeout_s is None
    assert budget.f08_v2_risk_ack == "AD-003/2026-11-14"
    assert budget.t_pre == 19.0
    assert budget.t_idle == 150.0

    adm = EgressAdmission(
        f08_policy="v2",
        client_timeout_s=None,
        f08_v2_risk_ack="AD-003/2026-11-14",
    )
    assert adm.f08_policy == "v2"
    assert adm.f08_v2_budget is not None
    assert adm.f08_v2_budget.f08_v2_risk_ack == "AD-003/2026-11-14"
