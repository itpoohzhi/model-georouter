"""Unit tests for MetadataCache HMAC scoping, singleflight, and negative_ttl=0 (Wave 7)."""

from __future__ import annotations

import threading
import time

from model_georouter.metadata_cache import MetadataCache, MetadataCacheConfig, MetadataQuery


def test_two_principals_and_tenant_headers_never_leak() -> None:
    cache = MetadataCache()
    l1 = cache.lookup_or_acquire(
        MetadataQuery("r1", "up:443", "GET", "/openai/v1/models", "", [("Authorization", "Bearer alpha")])
    )
    assert l1.kind == "leader"
    cache.complete(l1.key, 200, [("Content-Type", "application/json")], b'{"data":[{"id":"alpha"}]}')

    hit = cache.lookup_or_acquire(
        MetadataQuery("r1", "up:443", "GET", "/openai/v1/models", "", [("Authorization", "Bearer alpha")])
    )
    assert hit.kind == "hit" and hit.entry is not None and b"alpha" in hit.entry.body

    beta = cache.lookup_or_acquire(
        MetadataQuery("r1", "up:443", "GET", "/openai/v1/models", "", [("Authorization", "Bearer beta")])
    )
    assert beta.kind == "leader" and beta.key != l1.key
    cache.complete(beta.key, 200, [("Content-Type", "application/json")], b'{"data":[{"id":"beta"}]}')

    org2 = cache.lookup_or_acquire(
        MetadataQuery(
            "r1",
            "up:443",
            "GET",
            "/openai/v1/models",
            "",
            [("Authorization", "Bearer alpha"), ("OpenAI-Organization", "org-2")],
        )
    )
    assert org2.kind == "leader" and org2.key != l1.key


def test_bypass_on_cookie_query_secret_and_no_cache_and_negative_ttl_zero(monkeypatch) -> None:
    cache = MetadataCache()
    assert cache.lookup_or_acquire(MetadataQuery("r1", "up", "HEAD", "/v1/models", "", [])).kind == "bypass"
    assert (
        cache.lookup_or_acquire(MetadataQuery("r1", "up", "GET", "/v1/models", "", [("Cookie", "s=1")])).kind
        == "bypass"
    )
    assert cache.lookup_or_acquire(MetadataQuery("r1", "up", "GET", "/v1/models", "key=secret", [])).kind == "bypass"
    assert (
        cache.lookup_or_acquire(
            MetadataQuery("r1", "up", "GET", "/v1/models", "", [("Cache-Control", "no-cache")])
        ).kind
        == "bypass"
    )
    monkeypatch.setenv("GEOROUTER_METADATA_CACHE_ENABLED", "0")
    assert cache.lookup_or_acquire(MetadataQuery("r1", "up", "GET", "/v1/models", "", [])).kind == "bypass"
    monkeypatch.delenv("GEOROUTER_METADATA_CACHE_ENABLED", raising=False)

    err = cache.lookup_or_acquire(MetadataQuery("r1", "up", "GET", "/v1/models", "", [("x-api-key", "k1")]))
    assert err.kind == "leader"
    cache.complete(err.key, 502, [("Content-Type", "application/json")], b'{"error":"bad"}')
    again = cache.lookup_or_acquire(MetadataQuery("r1", "up", "GET", "/v1/models", "", [("x-api-key", "k1")]))
    assert again.kind == "leader"


def test_singleflight_coalesces_concurrent_requests_and_respects_max_inflight() -> None:
    cache = MetadataCache(MetadataCacheConfig(max_in_flight=2))
    hits = {"upstream": 0}
    lock = threading.Lock()
    results = []

    def worker(_idx: int) -> None:
        res = cache.lookup_or_acquire(
            MetadataQuery("r1", "up", "GET", "/v1/models", "", [("Authorization", "Bearer same")])
        )
        if res.kind == "leader":
            with lock:
                hits["upstream"] += 1
            time.sleep(0.04)
            cache.complete(res.key, 200, [("Content-Type", "application/json")], b'{"data":[1]}')
            results.append(b'{"data":[1]}')
        elif res.kind == "hit" and res.entry is not None:
            results.append(res.entry.body)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2)

    assert hits["upstream"] == 1
    assert len(results) == 12
