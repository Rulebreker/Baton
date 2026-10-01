"""Summary handoff: the task must survive a key change, whatever goes wrong."""

from __future__ import annotations

import json

import pytest

from baton.errors import AllKeysExhaustedError, ErrorKind
from baton.handoff import (
    HANDOFF_OPEN,
    SECTIONS,
    build_compacted,
    files_touched,
    is_handoff_message,
    local_summary,
    split_for_handoff,
)
from baton.providers.base import StreamChunk, Usage
from baton.redact import Redactor
from baton.session import Session, assemble_tool_calls, pending_tool_calls
from conftest import GOOD_SUMMARY, err, is_summary_request, ok, secret_for


def tool_call(call_id: str, name: str = "read_file", **args) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def long_history(turns: int = 6) -> list[dict]:
    """system + `turns` rounds of: user, assistant(tool call), tool result, assistant."""
    messages: list[dict] = [{"role": "system", "content": "You are a coding agent."}]
    for index in range(turns):
        messages += [
            {"role": "user", "content": f"step {index}: please continue the widget work"},
            {"role": "assistant", "content": None, "tool_calls": [tool_call(f"c{index}", path=f"src/file{index}.py")]},
            {"role": "tool", "tool_call_id": f"c{index}", "content": f"contents of file{index}"},
            {"role": "assistant", "content": f"finished step {index}"},
        ]
    return messages


def assert_tool_rounds_intact(messages: list[dict]) -> None:
    """Every tool result follows its call, and every call has its result."""
    open_calls: set[str] = set()
    for message in messages:
        if message.get("role") == "assistant":
            assert not open_calls, "assistant turn began while tool results were still owed"
            open_calls = {call["id"] for call in message.get("tool_calls") or []}
        elif message.get("role") == "tool":
            assert message["tool_call_id"] in open_calls, "orphaned tool result"
            open_calls.discard(message["tool_call_id"])
        elif message.get("role") == "user":
            assert not open_calls, "user turn began while tool results were still owed"
    assert not open_calls


def fill_daily_quota(h, index: int, fraction: float = 0.9) -> None:
    key = h.key(index)
    limit = key.config.limits.rpd
    for _ in range(int(limit * fraction) - key.day_requests):
        h.pool.record_success(key, Usage(1, 1))


# --- splitting --------------------------------------------------------------

@pytest.mark.parametrize("keep", range(0, 30))
def test_split_never_separates_a_tool_call_from_its_result(keep):
    messages = long_history(6)
    system, older, recent = split_for_handoff(messages, keep)
    assert system + older + recent == messages
    assert [m["role"] for m in system] == ["system"]
    if recent:
        assert recent[0]["role"] == "user"
    assert_tool_rounds_intact(recent)
    assert_tool_rounds_intact(older)


def test_compacted_history_merges_note_into_first_user_turn():
    compacted = build_compacted([{"role": "system", "content": "s"}], "NOTE", [{"role": "user", "content": "next"}])
    assert [m["role"] for m in compacted] == ["system", "user"]
    assert compacted[1]["content"].startswith(HANDOFF_OPEN) and compacted[1]["content"].endswith("next")
    assert is_handoff_message(compacted[1])
    # The note never becomes a system message (it may contain untrusted text).
    assert all(HANDOFF_OPEN not in str(m["content"]) for m in compacted if m["role"] == "system")


def test_files_touched_collects_paths_from_tool_calls():
    assert files_touched(long_history(3)) == ["src/file0.py", "src/file1.py", "src/file2.py"]


def test_local_summary_has_every_section_and_scrubs_secrets():
    redactor = Redactor()
    leaked = "sk-" + "Zx9" * 12
    messages = long_history(3) + [{"role": "user", "content": f"my key is {leaked}, mail me at dev@example.org"}]
    summary = local_summary(messages, redactor, pii=True)
    for section in SECTIONS:
        assert f"## {section}" in summary
    assert leaked not in summary and "dev@example.org" not in summary
    assert "src/file1.py" in summary


# --- planned handoff --------------------------------------------------------

async def test_planned_handoff_at_ninety_percent_moves_session_with_summary(harness, tmp_path):
    h = harness(3, home=tmp_path, limits={"rpd": 100})
    session = Session(messages=long_history(6), active_key="k0")
    original = list(session.messages)
    fill_daily_quota(h, 0)                       # k0 reaches 90% of its daily quota
    session.messages.append({"role": "user", "content": "now finish the renderer"})

    route = await h.relay.complete(session)

    # The outgoing key wrote the summary; the next key answered the task.
    summary_calls = [(secret, req) for secret, req in h.provider.calls if is_summary_request(req)]
    assert [secret for secret, _ in summary_calls] == [secret_for(0)]
    assert route.key_id == "k1" and session.active_key == "k1"

    # The transcript was compacted: summary present, recent turns verbatim.
    assert len(session.messages) < len(original)
    assert session.messages[0] == original[0]                      # system prompt untouched
    assert any(is_handoff_message(m) and "Finish the renderer" in m["content"] for m in session.messages)
    assert session.messages[-1]["content"].endswith("now finish the renderer")
    assert_tool_rounds_intact(session.messages)
    assert session.events[-1].kind == "handoff" and session.events[-1].to_key == "k1"

    # What the receiving key was sent is exactly the compacted transcript.
    task_request = h.provider.calls_for(1)[-1]
    assert task_request.messages == session.messages

    # Nothing was lost: the full pre-handoff transcript is checkpointed.
    checkpoints = list((tmp_path / "checkpoints").glob("*.json"))
    assert len(checkpoints) == 1
    saved = json.loads(checkpoints[0].read_text(encoding="utf-8"))
    assert len(saved["messages"]) == len(original) + 1 and saved["from_key"] == "k0"


async def test_no_handoff_below_threshold(harness):
    h = harness(2, limits={"rpd": 100})
    session = Session(messages=long_history(6), active_key="k0")
    fill_daily_quota(h, 0, 0.5)
    before = list(session.messages)
    route = await h.relay.complete(session)
    assert route.key_id == "k0" and session.messages == before and not session.events


async def test_handoff_never_happens_mid_tool_round(harness):
    h = harness(2, limits={"rpd": 100})
    messages = long_history(6)
    messages += [
        {"role": "user", "content": "read two files"},
        {"role": "assistant", "content": None, "tool_calls": [tool_call("a"), tool_call("b")]},
        {"role": "tool", "tool_call_id": "a", "content": "first result"},
        # result "b" has not arrived yet
    ]
    session = Session(messages=list(messages), active_key="k0")
    fill_daily_quota(h, 0)
    assert pending_tool_calls(session.messages) == {"b"}
    assert await h.relay.prepare(session) is None
    assert session.messages == messages and session.active_key == "k0"

    session.messages.append({"role": "tool", "tool_call_id": "b", "content": "second result"})
    outcome = await h.relay.prepare(session)     # round complete: now it may hand off
    assert outcome is not None and session.active_key == "k1"
    assert_tool_rounds_intact(session.messages)


async def test_short_history_moves_key_without_summarising(harness):
    h = harness(2, limits={"rpd": 100})
    session = Session(messages=[{"role": "user", "content": "hi"}], active_key="k0")
    fill_daily_quota(h, 0)
    route = await h.relay.complete(session)
    assert route.key_id == "k1"
    assert not any(is_summary_request(req) for _secret, req in h.provider.calls)
    assert session.messages == [{"role": "user", "content": "hi"}]


async def test_stays_on_draining_key_when_no_healthier_key_exists(harness):
    h = harness(2, limits={"rpd": 100})
    fill_daily_quota(h, 0)
    fill_daily_quota(h, 1)
    session = Session(messages=long_history(6), active_key="k0")
    route = await h.relay.complete(session)
    assert route.key_id == "k0" and not session.events


# --- summary fallback chain -------------------------------------------------

async def test_summary_falls_back_to_another_key_when_outgoing_key_fails(harness):
    h = harness(3, limits={"rpd": 100})
    fill_daily_quota(h, 0)
    h.provider.script(0, err(ErrorKind.RATE_LIMIT, retry_after=600))   # outgoing key cannot summarise
    session = Session(messages=long_history(6), active_key="k0")
    outcome = await h.relay.prepare(session)
    assert outcome.source == "model" and "Finish the renderer" in outcome.summary
    summarisers = [secret for secret, req in h.provider.calls if is_summary_request(req)]
    assert summarisers == [secret_for(0), secret_for(1)]


async def test_summary_falls_back_to_local_extractor_when_no_model_can_summarise(harness):
    h = harness(3, limits={"rpd": 100})
    fill_daily_quota(h, 0)
    h.provider.default = lambda req: err(ErrorKind.TRANSIENT) if is_summary_request(req) else ok("ok")
    session = Session(messages=long_history(6), active_key="k0")
    outcome = await h.relay.prepare(session)
    assert outcome.source == "local"
    for section in SECTIONS:
        assert f"## {section}" in outcome.summary
    assert "src/file0.py" in outcome.summary
    assert any(is_handoff_message(m) for m in session.messages)


async def test_malformed_model_summary_is_rejected(harness):
    h = harness(2, limits={"rpd": 100})
    fill_daily_quota(h, 0)
    h.provider.default = lambda req: ok("Sure! Here you go.")       # not a real handoff note
    session = Session(messages=long_history(6), active_key="k0")
    outcome = await h.relay.prepare(session)
    assert outcome.source == "local"


async def test_compact_never_raises_even_if_summarizer_explodes(harness):
    h = harness(1)

    async def explode(*_args):
        raise RuntimeError("boom")

    h.handoff._summarize = explode
    outcome = await h.handoff.compact(long_history(6), reason="quota", from_key="k0", to_key="k0")
    assert outcome.source == "local" and outcome.messages_after < outcome.messages_before


# --- scrubbing --------------------------------------------------------------

async def test_secrets_and_pii_never_reach_summary_request_summary_or_checkpoint(harness, tmp_path):
    h = harness(2, home=tmp_path, limits={"rpd": 100})
    pasted = "sk-ant-" + "Qw7" * 12
    messages = long_history(6)
    messages[1]["content"] += f" here is my key {pasted} and my address jane@example.org"
    # A model that parrots sensitive data back into its note.
    h.provider.default = lambda req: ok(
        GOOD_SUMMARY + f"\nUser key: {pasted}. Pool key: {secret_for(1)}. Contact jane@example.org"
        if is_summary_request(req) else "ok"
    )
    fill_daily_quota(h, 0)
    session = Session(messages=messages, active_key="k0")
    outcome = await h.relay.prepare(session)

    summary_request = next(req for _s, req in h.provider.calls if is_summary_request(req))
    sent_to_summariser = json.dumps(summary_request.messages)
    checkpoint = outcome.checkpoint.read_text(encoding="utf-8")
    for text in (sent_to_summariser, outcome.summary, json.dumps(session.messages[:2]), checkpoint):
        assert pasted not in text
        assert secret_for(1) not in text
        assert "jane@example.org" not in text


# --- unplanned failover and context overflow --------------------------------

async def test_mid_task_key_failure_retries_same_request_with_full_context(harness):
    h = harness(2)
    session = Session(messages=long_history(6), active_key="k0")
    before = list(session.messages)
    h.provider.script(0, err(ErrorKind.QUOTA_EXHAUSTED, retry_after=3600))
    route = await h.relay.complete(session)
    assert route.key_id == "k1" and session.active_key == "k1"
    assert session.messages == before                              # lossless: no summary needed
    assert h.provider.calls_for(1)[0].messages == before
    assert session.events[-1].kind == "failover"


async def test_context_overflow_compacts_and_retries(harness):
    h = harness(1)
    session = Session(messages=long_history(6) + [{"role": "user", "content": "go on"}], active_key="k0")
    h.provider.script(0, err(ErrorKind.CONTEXT_LENGTH, status=400))
    route = await h.relay.complete(session)
    assert route.result.content == "ok"
    assert any(is_handoff_message(m) for m in session.messages)
    assert session.events[-1].kind == "compaction"


async def test_context_fill_ratio_triggers_proactive_compaction(harness):
    h = harness(keys=[{"id": "k0", "provider": "openai", "models": ["m"], "context_window": 2000}])
    messages = long_history(6)
    messages[1]["content"] = "background " * 700                   # ~1900 tokens of a 2000 window
    session = Session(messages=messages + [{"role": "user", "content": "continue"}], active_key="k0")
    outcome = await h.relay.prepare(session)
    assert outcome is not None and outcome.reason == "context"
    assert session.active_key == "k0"                              # same key, smaller transcript


async def test_handoff_disabled_leaves_transcript_alone(harness):
    h = harness(2, limits={"rpd": 100}, handoff={"enabled": False})
    fill_daily_quota(h, 0)
    session = Session(messages=long_history(6), active_key="k0")
    before = list(session.messages)
    assert await h.relay.prepare(session) is None and session.messages == before


# --- streaming --------------------------------------------------------------

async def test_stream_restart_after_partial_failure_discards_partial_text(harness):
    h = harness(2)
    h.provider.stream_scripts[secret_for(0)] = [[StreamChunk(text="half an ans"), err(ErrorKind.TRANSIENT)]]
    h.provider.stream_scripts[secret_for(1)] = [[
        StreamChunk(text="full answer"), StreamChunk(finish_reason="stop", usage=Usage(10, 2)),
    ]]
    shown: list[str] = []
    session = Session(messages=[{"role": "user", "content": "hi"}], active_key="k0")
    route = await h.relay.stream(session, on_text=shown.append, on_restart=shown.clear)
    assert route.result.content == "full answer" and shown == ["full answer"]
    assert route.key_id == "k1" and session.active_key == "k1"
    assert session.messages == [{"role": "user", "content": "hi"}]  # transcript untouched until success


def test_streamed_tool_call_fragments_are_assembled():
    calls = assemble_tool_calls([
        {"index": 0, "id": "c1", "type": "function", "function": {"name": "read_file", "arguments": ""}},
        {"index": 0, "function": {"arguments": '{"path": '}},
        {"index": 1, "id": "c2", "function": {"name": "list_dir", "arguments": "{}"}},
        {"index": 0, "function": {"arguments": '"a.py"}'}},
    ])
    assert [c["id"] for c in calls] == ["c1", "c2"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "a.py"}
    # Truncated JSON arguments are neutralised rather than stored broken.
    broken = assemble_tool_calls([{"index": 0, "id": "c3", "function": {"name": "x", "arguments": '{"path": "a'}}])
    assert broken[0]["function"]["arguments"] == "{}"


# --- end to end -------------------------------------------------------------

async def test_long_task_survives_rotation_across_six_keys_then_fails_closed(harness, tmp_path):
    """Simulate a long agent task on six small-quota keys.

    Every turn must be answered, the session must walk through the keys in
    order with a handoff at each boundary, and when the whole pool is spent
    Baton must fail closed with a clear error instead of crashing or hanging.
    """
    h = harness(6, home=tmp_path, limits={"rpd": 10})
    session = Session(messages=[{"role": "system", "content": "You are a coding agent."}])
    served: list[str] = []
    answered = 0
    with pytest.raises(AllKeysExhaustedError) as caught:
        for turn in range(200):
            session.messages.append({"role": "user", "content": f"turn {turn}: keep going"})
            route = await h.relay.complete(session)
            session.messages.append({"role": "assistant", "content": route.result.content})
            assert_tool_rounds_intact(session.messages)
            served.append(route.key_id)
            answered += 1

    # Keys were used in order, each one drained before the next took over.
    order = list(dict.fromkeys(served))
    assert order == [f"k{i}" for i in range(6)]
    assert answered >= 6 * 8                                       # nearly the whole pooled quota was usable
    handoffs = [event for event in session.events if event.kind == "handoff"]
    assert [event.to_key for event in handoffs] == ["k1", "k2", "k3", "k4", "k5"]
    assert "exhausted" in str(caught.value)
    # The transcript stayed bounded thanks to compaction, and kept its system prompt.
    assert session.messages[0]["role"] == "system" and len(session.messages) < 40
    # No key was pushed past its configured daily quota.
    assert all(key.day_requests <= 10 for key in h.pool.keys.values())
