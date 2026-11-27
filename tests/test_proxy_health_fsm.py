"""Unit tests for ProxyHealthFSM & ProxyPoolManager opt-in policy (Wave 6)."""

from __future__ import annotations

import random

import pytest

from model_georouter.config import PoolConfig
from model_georouter.errors import AllProxiesPenalizedError
from model_georouter.proxy_health_fsm import ProxyHealthFSM, ProxyHealthPolicy, ProxyNodeState
from model_georouter.proxy_pool import ProxyEndpoint, ProxyPoolManager


def test_exponential_backoff_and_half_open_single_probe() -> None:
    now = [100.0]
    fsm = ProxyHealthFSM(ProxyHealthPolicy(enabled=True), clock=lambda: now[0], rng=random.Random(7))
    ep1 = ProxyEndpoint("http", "127.0.0.1", 8001, "", "", "http://127.0.0.1:8001")
    ep2 = ProxyEndpoint("http", "127.0.0.1", 8002, "", "", "http://127.0.0.1:8002")

    u1 = fsm.record_failure(ep1.raw, 2.0, pool_size=2)
    assert 101.8 <= u1 <= 102.2
    now[0] = 103.0
    u2 = fsm.record_failure(ep1.raw, 2.0, pool_size=2)
    assert 106.6 <= u2 <= 107.4  # 2 * 2^1 = 4s +/-10%
    fsm.record_failure(ep2.raw, 2.0, pool_size=2)

    # All penalized -> keep-last-alive grants strictly ONE concurrent HALF_OPEN probe
    first, ka1 = fsm.available_endpoints([ep1, ep2], set(), allow_keep_alive=True)
    assert ka1 is True and first == [ep2]
    assert fsm._records[ep2.raw].state is ProxyNodeState.HALF_OPEN

    second, ka2 = fsm.available_endpoints([ep1, ep2], set(), allow_keep_alive=True)
    assert ka2 is True and second == [ep1]
    third, ka3 = fsm.available_endpoints([ep1, ep2], set(), allow_keep_alive=True)
    assert third == [] and ka3 is False

    fsm.release_probe(ep2.raw, True)
    assert fsm.is_penalized(ep2.raw) is False


def test_one_proxy_pool_preserves_legacy_fast_fail_even_with_policy() -> None:
    now = [10.0]
    mgr = ProxyPoolManager(clock=lambda: now[0], health_policy=ProxyHealthPolicy(enabled=True))
    pool = PoolConfig(
        name="single", type="http_connect", strategy="first_available", proxies=("http://127.0.0.1:9001",)
    )
    ep = mgr.endpoints(pool)[0]
    mgr.penalize(ep, 30.0)
    with pytest.raises(AllProxiesPenalizedError):
        mgr.ensure_available(pool)
