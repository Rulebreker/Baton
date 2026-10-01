"""Key pool: quota accounting, health penalties, selection order, persistence."""

from __future__ import annotations

import json

from baton.errors import ErrorKind
from baton.pool import KeyPool, KeyStatus
from baton.providers.base import RateInfo, Usage
from conftest import Harness, err, make_config, secret_for


def ids(pool: KeyPool, model: str = "auto", **kwargs) -> list[str]:
    return [key.id for key, _model in pool.candidates(model, **kwargs)]


def test_sticky_strategy_keeps_config_order_and_honours_preference(harness):
    h = harness(4)
    assert ids(h.pool) == ["k0", "k1", "k2", "k3"]
    assert ids(h.pool, prefer="k2")[0] == "k2"
    assert ids(h.pool, exclude={"k0", "k1"}) == ["k2", "k3"]


def test_arbitrary_pool_sizes_are_supported(harness):
    for count in (1, 6, 10, 15, 40):
        assert len(ids(harness(count).pool)) == count


def test_per_minute_limit_cools_key_down_and_recovers(harness):
    h = harness(2, limits={"rpm": 3})
    key = h.key(0)
    for _ in range(3):
        h.pool.record_success(key, Usage(1, 1))
    assert h.pool.status(key) is KeyStatus.COOLDOWN
    assert ids(h.pool) == ["k1"]
    # Short-window pressure is not a reason for a summary handoff.
    assert not h.pool.needs_handoff(key)
    h.clock.advance(61)
    assert h.pool.status(key) is KeyStatus.READY
    assert ids(h.pool)[0] == "k0"


def test_daily_quota_triggers_handoff_at_threshold_then_exhaustion(harness):
    h = harness(2, limits={"rpd": 10})
    key = h.key(0)
    for _ in range(8):
        h.pool.record_success(key, Usage(1, 1))
    assert not h.pool.needs_handoff(key)
    h.pool.record_success(key, Usage(1, 1))  # 9/10 = 90%
    assert h.pool.needs_handoff(key)
    assert h.pool.status(key) is KeyStatus.DRAINING
    # A draining key is still usable, but only after every healthier key.
    assert ids(h.pool) == ["k1", "k0"]
    h.pool.record_success(key, Usage(1, 1))
    assert h.pool.status(key) is KeyStatus.EXHAUSTED
    assert ids(h.pool) == ["k1"]


def test_daily_counters_reset_on_new_utc_day(harness):
    h = harness(1, limits={"rpd": 2})
    key = h.key(0)
    h.pool.record_success(key, Usage(1, 1))
    h.pool.record_success(key, Usage(1, 1))
    assert h.pool.status(key) is KeyStatus.EXHAUSTED
    h.clock.advance(86_400)
    assert h.pool.status(key) is KeyStatus.READY
    assert key.day_requests == 0


def test_token_reservation_prevents_concurrent_overshoot(harness):
    h = harness(2, limits={"tpm": 1000})
    key = h.key(0)
    h.pool.reserve(key, 600)  # an in-flight request has claimed 600 tokens
    assert ids(h.pool, est_tokens=600) == ["k1"]
    h.pool.release(key, 600)
    assert ids(h.pool, est_tokens=600)[0] == "k0"


def test_oversized_single_request_is_still_allowed_on_an_idle_key(harness):
    # Our configured limit may be wrong; the provider is the judge of a lone request.
    h = harness(1, limits={"tpm": 1000})
    assert ids(h.pool, est_tokens=5000) == ["k0"]


def test_failure_penalties_match_error_class(harness):
    h = harness(5)
    pool = h.pool

    pool.record_failure(h.key(0), err(ErrorKind.AUTH))
    assert pool.status(h.key(0)) is KeyStatus.DISABLED

    pool.record_failure(h.key(1), err(ErrorKind.QUOTA_EXHAUSTED, retry_after=7200))
    assert pool.status(h.key(1)) is KeyStatus.EXHAUSTED
    assert pool.needs_handoff(h.key(1))

    pool.record_failure(h.key(2), err(ErrorKind.RATE_LIMIT, retry_after=12))
    assert pool.status(h.key(2)) is KeyStatus.COOLDOWN
    h.clock.advance(13)
    assert pool.status(h.key(2)) is KeyStatus.READY

    # The request was at fault: the key must not be punished.
    pool.record_failure(h.key(3), err(ErrorKind.BAD_REQUEST))
    pool.record_failure(h.key(3), err(ErrorKind.CONTEXT_LENGTH))
    assert pool.status(h.key(3)) is KeyStatus.READY and h.key(3).failures == 0


def test_circuit_breaker_opens_then_half_opens(harness):
    h = harness(2)
    key = h.key(0)
    threshold = h.config.rotation.breaker_threshold
    for _ in range(threshold):
        h.clock.advance(120)  # let each backoff elapse so failures are separate probes
        h.pool.record_failure(key, err(ErrorKind.TRANSIENT))
    assert h.pool.status(key) is KeyStatus.COOLDOWN
    assert key.cooldown_until - h.clock() >= h.config.rotation.breaker_cooldown
    h.clock.advance(h.config.rotation.breaker_cooldown + 1)
    assert h.pool.status(key) is KeyStatus.READY          # half-open: one probe allowed
    h.pool.record_failure(key, err(ErrorKind.TRANSIENT))  # probe fails: longer cooldown
    assert key.cooldown_until - h.clock() >= 2 * h.config.rotation.breaker_cooldown
    h.pool.record_success(key, Usage(1, 1))               # a success fully closes the breaker
    assert key.failures == 0 and h.pool.status(key) is KeyStatus.READY


def test_provider_reported_headers_feed_utilisation(harness):
    h = harness(2)
    h.pool.record_success(h.key(0), Usage(1, 1), RateInfo(limit_requests=100, remaining_requests=1))
    assert h.pool.short_utilization(h.key(0)) >= 0.99
    assert ids(h.pool) == ["k1"]
    h.clock.advance(61)  # header data is a per-minute signal and expires
    assert ids(h.pool)[0] == "k0"


def test_model_resolution_tiers_aliases_and_rejection():
    config = make_config(keys=[
        {"id": "g", "provider": "gemini", "models": ["gemini-x", "gemini-y"]},
        {"id": "o", "provider": "openai", "models": ["gpt-x"]},
    ], model_aliases={"fast": {"gemini": "gemini-y", "o": "gpt-x"}})
    pool = Harness(config).pool
    g, o = pool.keys["g"], pool.keys["o"]
    assert pool.resolve_model(g, "auto") == ("gemini-x", 0)
    assert pool.resolve_model(g, "gemini-y") == ("gemini-y", 0)
    assert pool.resolve_model(o, "gemini-y") == ("gpt-x", 1)   # cross-model fallback
    assert pool.resolve_model(g, "fast") == ("gemini-y", 0)
    assert pool.resolve_model(o, "fast") == ("gpt-x", 0)
    # Exact matches are always tried before fallbacks.
    assert [key.id for key, _ in pool.candidates("gpt-x")] == ["o", "g"]

    strict = Harness(make_config(keys=[{"id": "o", "provider": "openai", "models": ["gpt-x"]}],
                                 rotation={"unknown_model": "reject"})).pool
    assert strict.candidates("something-else") == []
    assert not strict.model_is_known("something-else")


def test_manual_disable_and_enable(harness):
    h = harness(2)
    h.pool.set_enabled("k0", False)
    assert ids(h.pool) == ["k1"]
    h.pool.set_enabled("k0", True)
    assert ids(h.pool)[0] == "k0"
    assert not h.pool.set_enabled("missing", True)


def test_usage_state_survives_restart_and_contains_no_secrets(tmp_path, harness):
    h = harness(2, home=tmp_path, limits={"rpd": 10})
    for _ in range(4):
        h.pool.record_success(h.key(0), Usage(100, 50))
    h.pool.record_failure(h.key(1), err(ErrorKind.QUOTA_EXHAUSTED, retry_after=5000))
    h.pool.save(force=True)

    raw = (tmp_path / "state.json").read_text(encoding="utf-8")
    assert secret_for(0) not in raw and secret_for(1) not in raw
    assert json.loads(raw)["keys"]["k0"]["day_requests"] == 4

    secrets = {"k0": secret_for(0), "k1": secret_for(1)}
    reloaded = KeyPool(h.config, secrets, state_path=tmp_path / "state.json", clock=h.clock)
    assert reloaded.keys["k0"].day_requests == 4
    assert reloaded.keys["k0"].day_tokens == 600
    assert reloaded.status(reloaded.keys["k1"]) is KeyStatus.EXHAUSTED


def test_corrupt_state_file_is_ignored(tmp_path, harness):
    (tmp_path / "state.json").write_text("{not json", encoding="utf-8")
    h = harness(1, home=tmp_path)
    assert ids(h.pool) == ["k0"]


def test_secret_never_appears_in_repr_or_snapshot(harness):
    h = harness(1)
    assert secret_for(0) not in repr(h.key(0))
    assert secret_for(0) not in json.dumps(h.pool.snapshot())
    assert secret_for(0) not in h.key(0).masked


def test_keys_without_a_secret_are_skipped():
    config = make_config(2)
    pool = KeyPool(config, {"k1": secret_for(1)})
    assert list(pool.keys) == ["k1"]


def test_next_available_reports_soonest_recovery(harness):
    h = harness(2)
    h.pool.record_failure(h.key(0), err(ErrorKind.RATE_LIMIT, retry_after=30))
    h.pool.record_failure(h.key(1), err(ErrorKind.RATE_LIMIT, retry_after=8))
    assert 7 <= h.pool.next_available_in() <= 8
