"""Контрактные тесты общего дедлайна egress и retry-бюджета F-08."""

from __future__ import annotations

import pytest

from model_georouter.retry_budget import EgressDeadline, RetryBudgetSpec


def test_f08_retry_budget_contract() -> None:
    budget = RetryBudgetSpec()

    budget.validate_f08()

    assert budget.total_egress_deadline == 40.0
    assert budget.connect_timeout == 3.0
    assert budget.pre_send_retries == 2
    assert budget.multi_proxy_attempts == 3
    assert budget.max_backoff_seconds == 2.0
    assert budget.worst_case_pre_send_seconds == 11.0
    assert budget.single_pool_max_retries == 0
    assert budget.primary_first_header_timeout == 5.0
    assert budget.penalize_on_header_timeout is True


def test_f08_rejects_budget_that_can_outlive_total_deadline() -> None:
    budget = RetryBudgetSpec(
        connect_timeout=13.0,
        total_egress_deadline=40.0,
        pre_send_retries=2,
        max_backoff_seconds=2.0,
    )

    with pytest.raises(ValueError, match="pre-send budget"):
        budget.validate_f08()


@pytest.mark.parametrize(
    ("budget", "message"),
    [
        (RetryBudgetSpec(total_egress_deadline=40.1), "total_egress_deadline"),
        (RetryBudgetSpec(connect_timeout=3.1), "connect_timeout"),
        (RetryBudgetSpec(pre_send_retries=3), "pre_send_retries"),
        (RetryBudgetSpec(single_pool_max_retries=1), "single_pool_max_retries"),
        (RetryBudgetSpec(primary_first_header_timeout=5.1), "primary_first_header_timeout"),
        (RetryBudgetSpec(penalize_on_header_timeout=False), "penalize_on_header_timeout"),
    ],
)
def test_f08_rejects_values_outside_failover_contract(
    budget: RetryBudgetSpec,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        budget.validate_f08()


def test_egress_deadline_clamps_every_phase_to_one_absolute_deadline() -> None:
    deadline = EgressDeadline.from_timeout(40.0, started_at=100.0)

    assert deadline.absolute_deadline == 140.0
    assert deadline.remaining(now=112.0) == 28.0
    assert deadline.clamp(3.0, now=112.0) == 3.0
    assert deadline.clamp(30.0, now=112.0) == 28.0
    assert deadline.remaining(now=141.0) == 0.0
    assert deadline.expired(now=141.0) is True


def test_egress_deadline_refuses_to_start_expired_phase() -> None:
    deadline = EgressDeadline(absolute_deadline=140.0)

    with pytest.raises(TimeoutError, match="egress deadline expired"):
        _ = deadline.clamp(5.0, now=140.0)
