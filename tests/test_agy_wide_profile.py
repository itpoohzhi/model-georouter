"""Контрактные тесты профиля расширения AGY (160/160/320/128/512) без поломки базовых дефолтов и маршрутизации."""

from __future__ import annotations

import asyncio
import json
import plistlib
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from model_georouter import config_guard
from model_georouter import egress_admission as _ea
from model_georouter.egress_admission import EgressAdmission, InflightLimitError

ROOT = Path(__file__).resolve().parents[1]
STREAM_PATH = "/cloudcode/v1internal:streamGenerateContent?alt=sse"
DE_PORTS = ("http://127.0.0.1:10827", "http://127.0.0.1:10828", "http://127.0.0.1:10829")


def _merged_test_config(tmp_path: Path) -> dict[str, Any]:
    """Сливает server-overlay deploy/agy-wide-limits.json с фикстурными pools/rules без затирания route-us-pm."""
    overlay: dict[str, Any] = json.loads((ROOT / "deploy/agy-wide-limits.json").read_text(encoding="utf-8"))
    assert set(overlay) == {"server"}, "RED-GEO: overlay must only contain 'server' keys (never overwrite pools/rules)"
    return {
        "server": dict(overlay["server"]),
        "pools": {
            "direct": {"type": "direct"},
            "route-de": {
                "type": "http_connect",
                "egress_id": "de-213.165.42.220",
                "strategy": "first_available",
                "proxies": list(DE_PORTS),
            },
            "route-us-pm": {
                "type": "http_connect",
                "strategy": "first_available",
                "proxies": ["http://127.0.0.1:10820"],
            },
        },
        "rules": [
            {"match_prefix": ["claude-", "anthropic/"], "pool": "route-us-pm"},
            {"match_prefix": ["gemini-", "gpt-"], "pool": "route-de"},
        ],
        "default_pool": "direct",
        "geo_fallback_pool": "route-de",
    }


def test_agy_wide_deploy_overlay_preserves_timeouts_f08_and_launchd_limits(tmp_path: Path) -> None:
    """F-003/AD-003: overlay не снижает live-таймауты, хранит F-08 connect=3.0 и оба FD-лимита plist."""
    overlay: dict[str, Any] = json.loads((ROOT / "deploy/agy-wide-limits.json").read_text(encoding="utf-8"))
    srv: dict[str, Any] = overlay["server"]
    assert srv["max_inflight_per_backend"] == 160
    assert srv["max_inflight_long_header"] == 160
    assert srv["proxy_slots"] == 320
    assert srv["max_inflight_per_backend"] <= srv["proxy_slots"] // 2
    assert srv["direct_slots"] == 128
    assert srv["max_connections"] == 512
    assert srv["admission_wait_s"] == 0.0
    # Не понижать таймауты и сохранять F-08 + penalize_header_timeout_without_alternative=false:
    assert srv["connect_timeout"] == 3.0
    assert srv["headers_timeout"] >= 180.0
    assert srv["body_timeout"] >= 180.0
    assert srv["inactivity_timeout"] >= 300.0
    assert srv["body_buffer_max_bytes"] >= 268435456
    assert srv["penalize_header_timeout_without_alternative"] is False

    merged_path = tmp_path / "merged-config.json"
    merged_path.write_text(json.dumps(_merged_test_config(tmp_path)), encoding="utf-8")
    guarded = config_guard.load_and_validate_config(merged_path)
    assert guarded.max_inflight_per_backend == 160
    assert guarded.max_inflight_long_header == 160

    plist_path = ROOT / "deploy/launchd/com.user.model-georouter.plist"
    plist: dict[str, Any] = plistlib.loads(plist_path.read_bytes())
    soft_dict: dict[str, Any] = plist.get("SoftResourceLimits", {})
    hard_dict: dict[str, Any] = plist.get("HardResourceLimits", {})
    soft_nof = soft_dict.get("NumberOfFiles", 0)
    hard_nof = hard_dict.get("NumberOfFiles", 0)
    assert isinstance(soft_nof, int) and soft_nof >= 4096
    assert isinstance(hard_nof, int) and hard_nof >= 8192


def test_class_defaults_remain_untouched() -> None:
    """AD-004: константы класса EgressAdmission (40/24/80) сохраняются неизменными."""
    assert _ea.DEFAULT_PROXY_SLOTS == 80
    assert _ea.DEFAULT_MAX_INFLIGHT_PER_BACKEND == 40
    assert _ea.DEFAULT_MAX_INFLIGHT_LONG_HEADER == 24


def test_agy_wide_profile_admits_160_concurrent_stream_requests_without_wait(tmp_path: Path) -> None:
    """Профиль 160/160/320 пропускает 160 одновременных streamGenerateContent запросов и мгновенно режет 161-й."""
    merged_path = tmp_path / "merged-config.json"
    merged_path.write_text(json.dumps(_merged_test_config(tmp_path)), encoding="utf-8")
    guarded = config_guard.load_and_validate_config(merged_path)

    async def _run() -> None:
        admission = EgressAdmission(config=guarded)
        assert admission.max_inflight_per_backend == 160
        assert admission.max_inflight_long_header == 160
        async with AsyncExitStack() as stack:
            tickets: list[object] = []
            for idx in range(160):
                t = await stack.enter_async_context(admission.admit("route-de", STREAM_PATH, backend=DE_PORTS[idx % 3]))
                tickets.append(t)
            assert len(tickets) == 160
            assert admission.inflight_long_header == 160
            t0 = time.monotonic()
            try:
                async with admission.admit("route-de", STREAM_PATH, backend=DE_PORTS[0]):
                    raise AssertionError("expected InflightLimitError on 161st slot")
            except InflightLimitError as exc:
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                assert exc.status_code == 503
                assert elapsed_ms < 50.0
        assert admission.inflight_long_header == 0

    asyncio.run(_run())
