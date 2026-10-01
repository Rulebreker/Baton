"""Regression tests for defects found by running Baton end to end."""

from __future__ import annotations

import asyncio
import json

from baton.pool import KeyPool, clear_parked_state
from baton.providers.base import StreamChunk, Usage
from conftest import secret_for


async def test_usage_burst_is_flushed_to_disk_without_a_clean_shutdown(tmp_path, harness, monkeypatch):
    """A burst of requests inside the debounce window must still be persisted.

    Before the fix only the first request of a burst was written; a killed
    process then restarted with nearly empty counters and overspent its keys.
    """
    monkeypatch.setattr("baton.pool._SAVE_INTERVAL", 0.05)
    h = harness(1, home=tmp_path, limits={"rpd": 100})
    for _ in range(9):
        h.pool.record_success(h.key(0), Usage(10, 5))      # all within the debounce window
    on_disk = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert on_disk["keys"]["k0"]["day_requests"] < 9        # debounced, as intended
    await asyncio.sleep(0.2)                                # trailing flush fires; no aclose() was called
    on_disk = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert on_disk["keys"]["k0"]["day_requests"] == 9

    restarted = KeyPool(h.config, {"k0": secret_for(0)}, state_path=tmp_path / "state.json", clock=h.clock)
    assert restarted.keys["k0"].day_requests == 9


async def test_trailing_usage_chunk_reaches_stream_consumers(harness):
    """OpenAI-style streams report usage in a final chunk that has no content."""
    h = harness(1)
    h.provider.stream_scripts[secret_for(0)] = [[
        StreamChunk(text="hi"), StreamChunk(finish_reason="stop"), StreamChunk(usage=Usage(30, 4)),
    ]]
    from conftest import request

    chunks = [chunk async for chunk in h.router.stream(request())]
    assert chunks[-1].usage == Usage(30, 4)
    assert h.key(0).day_tokens == 34                        # and the pool counted the real usage


def test_newer_google_key_format_is_redacted_even_when_never_registered():
    """A key pasted into the wrong prompt is not in the vault, so only the pattern can catch it."""
    from baton.redact import scrub

    pasted = "AQ." + "Ab8RN6J" * 6
    assert pasted not in scrub(f"Provider: {pasted}")
    assert "[REDACTED:google-key]" in scrub(f"key={pasted}")


def test_enabling_a_key_clears_its_persisted_parking(tmp_path, harness):
    from baton.errors import ErrorKind
    from baton.pool import KeyStatus
    from conftest import err

    h = harness(1, home=tmp_path)
    h.pool.record_failure(h.key(0), err(ErrorKind.AUTH))
    h.pool.save(force=True)
    secrets = {"k0": secret_for(0)}
    parked = KeyPool(h.config, secrets, state_path=tmp_path / "state.json", clock=h.clock)
    assert parked.status(parked.keys["k0"]) is KeyStatus.DISABLED     # the parking survived the restart

    clear_parked_state(tmp_path / "state.json", "k0")
    reloaded = KeyPool(h.config, secrets, state_path=tmp_path / "state.json", clock=h.clock)
    assert reloaded.status(reloaded.keys["k0"]) is KeyStatus.READY
    clear_parked_state(tmp_path / "missing.json", "k0")     # absent file: silently nothing to do
