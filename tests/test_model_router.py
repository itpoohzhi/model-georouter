from __future__ import annotations

import gzip
import json
import random
import threading
import time

import pytest

from universal_ai_bridge.config import parse_config
from universal_ai_bridge.errors import BodyTooLargeError
from universal_ai_bridge.geo_cache import GeoCache
from universal_ai_bridge.model_router import (
    BodyInspector,
    ModelRouter,
    RegionErrorClassifier,
    model_from_path,
)

LIMIT = 16 * 1024 * 1024

BODIES = [
    (b'{"model":"gpt-5","messages":[]}', "gpt-5"),
    (b'{ \n "messages": [{"model":"inner"}], "model" : "claude-x" }', "claude-x"),
    (b'{"messages":[{"model":"inner"}],"x":{"model":"deep"}}', None),
    (b'{"prompt":"{\\"model\\":\\"fake\\"} } ]","model":"real"}', "real"),
    (b'{"mod\\u0065l":"x"}', "x"),
    (b'{"model":"mod\\u00e9l \\"q\\""}', 'modél "q"'),
    (b'{"model":null}', None),
    (b'{"model":123,"other":1}', None),
    (b'{"n":12.5e3,"t":true,"z":null,"model":"m"}', "m"),
    (b'{"a":{"b":[{"c":"]}"}]},"model":"deep"}', "deep"),
    (b'{"a":[1,2,[3,{"k":"}"}]],"model":"after-array"}', "after-array"),
    ('{"model":"модель-1"}'.encode(), "модель-1"),
    (b'{"model":"same","model":"same"}', "same"),  # RW-003: конфликтующие дубли — в test_council_cycle2
    (b"[]", None),
    (b"", None),
    (b"not json at all", None),
    (b'{"a":1}', None),
    (b"{}", None),
    (b'\xef\xbb\xbf{"model":"bom"}', None),  # BOM — не объект верхнего уровня: фолбэк на default
]


@pytest.mark.parametrize("body, expected", BODIES)
def test_inspector_whole_body(body, expected):
    inspector = BodyInspector(LIMIT)
    inspector.feed(body)
    assert inspector.model == expected
    assert inspector.body == body


@pytest.mark.parametrize("body, expected", BODIES)
def test_inspector_every_split_point(body, expected):
    for cut in range(len(body) + 1):
        inspector = BodyInspector(LIMIT)
        inspector.feed(body[:cut])
        inspector.feed(body[cut:])
        assert inspector.model == expected, f"split at {cut}"
        assert inspector.body == body


@pytest.mark.parametrize("body, expected", BODIES)
def test_inspector_byte_by_byte_and_random_chunks(body, expected):
    inspector = BodyInspector(LIMIT)
    for i in range(len(body)):
        inspector.feed(body[i : i + 1])
    assert inspector.model == expected
    rng = random.Random(1)
    for _ in range(20):
        inspector = BodyInspector(LIMIT)
        pos = 0
        while pos < len(body):
            step = rng.randint(1, 7)
            inspector.feed(body[pos : pos + step])
            pos += step
        assert inspector.model == expected


def test_inspector_agrees_with_json_loads_on_generated_bodies():
    rng = random.Random(42)
    for _ in range(200):
        payload = {
            "temperature": rng.random(),
            "messages": [{"role": "user", "content": "}{\"" * rng.randint(0, 3)}],
            "tools": [{"model": "nested"}],
        }
        model = rng.choice(["gpt-5", "claude-3", None])
        if model:
            keys = list(payload) + ["model"]
            rng.shuffle(keys)
            payload = {k: payload.get(k, model) for k in keys}
        body = json.dumps(payload, ensure_ascii=rng.random() < 0.5).encode()
        inspector = BodyInspector(LIMIT)
        for i in range(0, len(body), 5):
            inspector.feed(body[i : i + 5])
        assert inspector.model == payload.get("model")


def test_inspector_large_late_model_is_linear():
    body = b'{"a":"' + b"x" * 3_000_000 + b'","model":"late"}'
    inspector = BodyInspector(LIMIT)
    started = time.monotonic()
    for i in range(0, len(body), 1000):
        inspector.feed(body[i : i + 1000])
    assert inspector.model == "late"
    assert time.monotonic() - started < 3  # квадратичная деградация дала бы минуты


def test_inspector_keeps_scanning_after_model_and_buffering():  # RW-003: дубли model ищутся до конца объекта
    inspector = BodyInspector(LIMIT)
    inspector.feed(b'{"model":"m",')
    assert not inspector.scan_finished and inspector.model == "m"
    inspector.feed(b'"rest":[1,2,3]}')
    assert inspector.body == b'{"model":"m","rest":[1,2,3]}'


def test_inspector_enforces_body_limit():
    inspector = BodyInspector(100)
    inspector.feed(b"x" * 100)  # ровно лимит — допустимо
    with pytest.raises(BodyTooLargeError) as info:
        inspector.feed(b"y")
    assert info.value.limit == 100

    chunked = BodyInspector(100)
    chunked.feed(b'{"model":"m","p":"' + b"z" * 50)
    with pytest.raises(BodyTooLargeError):
        chunked.feed(b"z" * 50)


def test_inspector_default_limit_is_16_mib_from_config():
    assert parse_config(
        {"server": {"listen": "127.0.0.1", "port": 1}, "pools": {}, "rules": []}
    ).server.body_buffer_max_bytes == 16 * 1024 * 1024


def test_model_from_path():
    assert model_from_path("/v1beta/models/gemini-2.5-pro:generateContent") == "gemini-2.5-pro"
    assert model_from_path("/v1/chat/completions") is None


# ───────────────────────────── ModelRouter ─────────────────────────────


def make_cfg(rules, **extra):
    pools = {
        "direct": {"type": "direct"},
        "route-de": {"type": "http_connect", "proxies": ["http://127.0.0.1:10820"]},
        "route-hy2": {"type": "socks5", "proxies": ["socks5h://127.0.0.1:10821"]},
    }
    data = {"server": {"listen": "127.0.0.1", "port": 1}, "pools": pools, "rules": rules}
    data.update(extra)
    return parse_config(data)


def test_router_prefix_regex_and_default():
    router = ModelRouter.from_config(
        make_cfg(
            [
                {"match_prefix": ["claude-", "gpt-"], "pool": "route-de"},
                {"match_regex": r"^gemini-.*-preview$", "pool": "route-hy2"},
            ]
        )
    )
    assert router.match("claude-3-opus").pool == "route-de"
    assert router.match("gpt-5").pool == "route-de"
    assert router.match("gemini-2-preview").pool == "route-hy2"
    assert router.match("gemini-2-stable").pool == "direct"
    assert router.match("llama-3").pool == "direct"
    assert router.match("llama-3").source == "default"
    assert router.match(None).pool == "direct"
    assert router.match("").pool == "direct"


def test_router_first_matching_rule_wins_and_regex_uses_search():
    router = ModelRouter.from_config(
        make_cfg(
            [
                {"match_regex": "opus", "pool": "route-hy2"},
                {"match_prefix": ["claude-"], "pool": "route-de"},
            ]
        )
    )
    decision = router.match("claude-3-opus")
    assert (decision.pool, decision.rule_index, decision.source) == ("route-hy2", 0, "rule")
    assert router.match("claude-3-haiku").pool == "route-de"


def test_router_rule_with_prefix_and_regex_matches_either():
    router = ModelRouter.from_config(make_cfg([{"match_prefix": ["a-"], "match_regex": "z$", "pool": "route-de"}]))
    assert router.match("a-1").pool == "route-de"
    assert router.match("b-z").pool == "route-de"
    assert router.match("b-1").pool == "direct"


def test_router_custom_default_pool():
    router = ModelRouter.from_config(make_cfg([], default_pool="route-hy2"))
    assert router.match("anything").pool == "route-hy2"


def test_router_geo_cache_diverts_direct_models_to_fallback():
    cfg = make_cfg([{"match_prefix": ["claude-"], "pool": "route-hy2"}], geo_fallback_pool="route-de")
    router = ModelRouter.from_config(cfg)
    cache = GeoCache()
    assert router.decide("gpt-5", "up", cache).pool == "direct"
    cache.mark_blocked("up", "gpt-5")
    decision = router.decide("gpt-5", "up", cache)
    assert (decision.pool, decision.source) == ("route-de", "geo_cache")
    assert router.decide("gpt-5", "other-upstream", cache).pool == "direct"  # ключ включает upstream
    cache.mark_blocked("up", "claude-1")
    assert router.decide("claude-1", "up", cache).pool == "route-hy2"  # уже прокси-пул — не трогаем


def test_router_without_fallback_ignores_geo_cache():
    router = ModelRouter.from_config(make_cfg([], geo_fallback_pool=""))
    cache = GeoCache()
    cache.mark_blocked("up", "m")
    assert router.decide("m", "up", cache).pool == "direct"


# ───────────────────────────── GeoCache ─────────────────────────────


def test_geo_cache_ttl_24h_with_fake_clock():
    clock = [0.0]
    cache = GeoCache(clock=lambda: clock[0])  # TTL по умолчанию — 24 ч
    cache.mark_blocked("up", "m")
    clock[0] = 86399
    assert cache.is_blocked("up", "m")
    clock[0] = 86401
    assert not cache.is_blocked("up", "m")
    assert len(cache) == 0


def test_geo_cache_key_is_upstream_and_model_and_ttl_override():
    clock = [0.0]
    cache = GeoCache(clock=lambda: clock[0])
    cache.mark_blocked("a", "m", ttl_seconds=10)
    assert cache.is_blocked("a", "m")
    assert not cache.is_blocked("b", "m")
    assert not cache.is_blocked("a", "other")
    clock[0] = 11
    assert not cache.is_blocked("a", "m")


def test_geo_cache_ignores_missing_model_and_flushes():
    cache = GeoCache()
    cache.mark_blocked("a", None)
    assert len(cache) == 0 and not cache.is_blocked("a", None)
    cache.mark_blocked("a", "m1")
    cache.mark_blocked("a", "m2")
    assert cache.flush() == 2
    assert not cache.is_blocked("a", "m1")


def test_geo_cache_disk_persistence_roundtrip(tmp_path):
    path = tmp_path / "geo_cache.json"
    clock = [100.0]
    first = GeoCache(path, clock=lambda: clock[0])
    first.mark_blocked("up", "m", ttl_seconds=50)
    first.mark_blocked("up", "short", ttl_seconds=1)
    assert path.exists()
    clock[0] = 110.0
    second = GeoCache(path, clock=lambda: clock[0])
    assert second.is_blocked("up", "m")
    assert not second.is_blocked("up", "short")  # истёкшие записи не воскресают
    second.flush()
    assert json.loads(path.read_text()) == []


def test_geo_cache_corrupt_file_is_ignored(tmp_path):
    path = tmp_path / "geo_cache.json"
    path.write_text("{garbage")
    cache = GeoCache(path)
    assert len(cache) == 0
    cache.mark_blocked("u", "m")  # и после этого работает
    assert GeoCache(path).is_blocked("u", "m")


def test_geo_cache_is_thread_safe(tmp_path):
    cache = GeoCache(tmp_path / "c.json")
    errors = []

    def worker(n):
        try:
            for i in range(100):
                cache.mark_blocked("u", f"m{n}-{i % 5}")
                cache.is_blocked("u", f"m{n}-{i % 5}")
                len(cache)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(cache) == 8 * 5


# ───────────────────────────── RegionErrorClassifier ─────────────────────────────

DEFAULT_SIGNATURES = parse_config(
    {"server": {"listen": "127.0.0.1", "port": 1}, "pools": {}, "rules": []}
).server.geo_error_signatures


@pytest.mark.parametrize(
    "body",
    [
        b'{"error":{"type":"RegionError","message":"blocked"}}',
        b'{"error":"This model is NOT AVAILABLE IN YOUR REGION"}',
        b"Sorry, country not supported",
        b'{"message":"Location not supported for this API"}',
        b"request was geoblocked",
    ],
)
def test_classifier_detects_region_errors(body):
    result = RegionErrorClassifier(DEFAULT_SIGNATURES).classify(403, body)
    assert result.is_region_error is True
    assert result.reason.startswith("region_signature:")


@pytest.mark.parametrize(
    "body",
    [
        b'{"error":{"message":"Invalid API key provided"}}',
        b'{"error":{"type":"insufficient_quota","message":"You exceeded your current quota"}}',
        b'{"error":"Rate limit reached"}',
        b"<html><title>Attention Required! | Cloudflare</title>Sorry, you have been blocked</html>",
        b'{"error":"Forbidden"}',
        b"",
    ],
)
def test_classifier_auth_quota_waf_are_not_region_errors(body):
    assert RegionErrorClassifier(DEFAULT_SIGNATURES).classify(403, body).is_region_error is False


def test_classifier_only_looks_at_403():
    classifier = RegionErrorClassifier(DEFAULT_SIGNATURES)
    assert classifier.classify(200, b"RegionError").is_region_error is False
    assert classifier.classify(451, b"RegionError").is_region_error is False


def test_classifier_uses_configured_signatures_and_gzip():
    classifier = RegionErrorClassifier(["only-in-narnia"])
    assert classifier.classify(403, b"Service is ONLY-IN-NARNIA").is_region_error
    assert not classifier.classify(403, b"RegionError").is_region_error
    packed = gzip.compress(b'{"error":"only-in-narnia"}')
    assert classifier.classify(403, packed, "gzip").is_region_error
    assert not classifier.classify(403, b"\x00garbage", "gzip").is_region_error
    assert not classifier.classify(403, b"only-in-narnia", "br").is_region_error


# ───────────────────────────── Rework Cycle 1 ─────────────────────────────

DUPLICATE_BODIES = [
    (b'{"model":"a","model":"a"}', "a"),
    (b'{"model":null,"model":"b"}', "b"),
    (b'{"model":5,"x":1,"model":"b"}', "b"),
    (b'{"model":{"model":"n"},"model":"b"}', "b"),
    (b'{"model":["m"],"model":"b","model":"b"}', "b"),
    (b'{"model":"","model":"b"}', "b"),
    (b'{"model":null,"model":7}', None),
    (b'{"x":{"model":"n"},"model":"a","y":{"model":"m"},"model":"a"}', "a"),
]


@pytest.mark.parametrize("body, expected", DUPLICATE_BODIES)
def test_duplicate_model_keys_pick_first_valid_string_at_every_split(body, expected):  # RW-013
    for cut in range(len(body) + 1):
        inspector = BodyInspector(LIMIT)
        inspector.feed(body[:cut])
        inspector.feed(body[cut:])
        assert inspector.model == expected, f"split at {cut}"
    inspector = BodyInspector(LIMIT)
    for i in range(len(body)):
        inspector.feed(body[i : i + 1])
    assert inspector.model == expected


def test_geo_cache_lookups_are_not_blocked_by_slow_disk(tmp_path, monkeypatch):  # RW-007
    import os

    path = tmp_path / "c.json"
    cache = GeoCache(path)
    started, release = threading.Event(), threading.Event()
    real_replace = os.replace

    def slow_replace(src, dst):
        started.set()
        release.wait(5)
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", slow_replace)
    writer = threading.Thread(target=cache.mark_blocked, args=("u", "m"))
    writer.start()
    assert started.wait(2)
    done = threading.Event()

    def lookups():
        cache.is_blocked("u", "other")
        len(cache)
        done.set()

    reader = threading.Thread(target=lookups)
    reader.start()
    try:
        assert done.wait(1.0), "lookups blocked while disk write is in progress"
    finally:
        release.set()
        writer.join(5)
        reader.join(5)
    assert GeoCache(path).is_blocked("u", "m")


def test_geo_cache_stale_snapshot_never_overwrites_newer_one(tmp_path):  # RW-007
    path = tmp_path / "c.json"
    cache = GeoCache(path)
    with cache._lock:
        cache._entries[("u", "old")] = time.time() + 100
        stale = cache._snapshot()
        cache._entries[("u", "new")] = time.time() + 100
        fresh = cache._snapshot()
    cache._persist(fresh)
    cache._persist(stale)  # запоздавший писатель
    reloaded = GeoCache(path)
    assert reloaded.is_blocked("u", "old") and reloaded.is_blocked("u", "new")
