"""Agent tools (sandboxing, approvals) and the turn loop's transcript guarantees."""

from __future__ import annotations

import json
import os
import sys

import pytest

from baton.agent import TOOL_SCHEMAS, Agent, Toolbox
from baton.agent.tools import clean_environment, is_secret_file
from baton.errors import AllKeysExhaustedError, ErrorKind
from baton.providers.base import StreamChunk, Usage
from baton.runtime import Runtime
from conftest import Harness, err, make_config, secret_for


class Approvals:
    def __init__(self, answer: bool = True) -> None:
        self.answer = answer
        self.asked: list[tuple[str, str]] = []

    async def __call__(self, name: str, description: str) -> bool:
        self.asked.append((name, description))
        return self.answer


@pytest.fixture
def box(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "src" / "app.py").write_text("def main():\n    return 1\n", encoding="utf-8")
    (workspace / ".env").write_text("TOKEN=abc", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("outside", encoding="utf-8")
    approvals = Approvals()
    return Toolbox(workspace, approve=approvals, command_timeout=20), approvals, workspace


# --- sandbox ----------------------------------------------------------------

async def test_read_write_edit_roundtrip(box):
    toolbox, approvals, workspace = box
    assert "def main" in (await toolbox.run("read_file", {"path": "src/app.py"})).output
    assert (await toolbox.run("read_file", {"path": "src/app.py", "offset": 2, "limit": 1})).output.strip() == "return 1"
    written = await toolbox.run("write_file", {"path": "notes/todo.md", "content": "- one\n"})
    assert written.ok and (workspace / "notes" / "todo.md").read_text(encoding="utf-8") == "- one\n"
    edited = await toolbox.run("edit_file", {"path": "src/app.py", "old_string": "return 1", "new_string": "return 2"})
    assert edited.ok and "return 2" in (workspace / "src" / "app.py").read_text(encoding="utf-8")
    assert [name for name, _ in approvals.asked] == ["write_file", "edit_file"]


@pytest.mark.parametrize("path", ["../outside.txt", "..\\outside.txt", "src/../../outside.txt"])
async def test_paths_cannot_escape_the_workspace(box, path):
    toolbox, _, _ = box
    result = await toolbox.run("read_file", {"path": path})
    assert not result.ok and "outside the workspace" in result.output
    assert not (await toolbox.run("write_file", {"path": path, "content": "x"})).ok


async def test_absolute_paths_outside_are_refused(box, tmp_path):
    toolbox, _, _ = box
    assert not (await toolbox.run("read_file", {"path": str(tmp_path / "outside.txt")})).ok


@pytest.mark.skipif(sys.platform == "win32", reason="creating symlinks needs elevated rights on Windows")
async def test_symlink_escape_is_refused(box, tmp_path):
    toolbox, _, workspace = box
    os.symlink(tmp_path / "outside.txt", workspace / "link.txt")
    assert not (await toolbox.run("read_file", {"path": "link.txt"})).ok


@pytest.mark.parametrize("name", [".env", ".env.production", "keys.vault", "master.key", "id_rsa", "server.pem", ".npmrc"])
def test_secret_files_are_recognised(tmp_path, name):
    assert is_secret_file(tmp_path / name)
    assert not is_secret_file(tmp_path / ".env.example")


async def test_secret_files_cannot_be_read_listed_in_search_or_overwritten(box):
    toolbox, _, _ = box
    assert not (await toolbox.run("read_file", {"path": ".env"})).ok
    assert not (await toolbox.run("write_file", {"path": ".env", "content": "x"})).ok
    assert "abc" not in (await toolbox.run("search_files", {"pattern": "abc"})).output
    assert ".env" not in (await toolbox.run("find_files", {"pattern": "*"})).output.split()


async def test_protected_directory_and_git_internals_are_off_limits(tmp_path):
    workspace = tmp_path
    data = workspace / ".baton"
    data.mkdir()
    (data / "config.yaml").write_text("keys: []", encoding="utf-8")
    (workspace / ".git" / "hooks").mkdir(parents=True)
    toolbox = Toolbox(workspace, approve=Approvals(), protected=(data,))
    assert not (await toolbox.run("read_file", {"path": ".baton/config.yaml"})).ok
    hook = await toolbox.run("write_file", {"path": ".git/hooks/pre-commit", "content": "#!/bin/sh\n"})
    assert not hook.ok and ".git" in hook.output


async def test_search_and_find(box):
    toolbox, _, _ = box
    found = await toolbox.run("search_files", {"pattern": r"def \w+", "glob": "*.py"})
    assert found.output.startswith("src/app.py:1:")
    assert (await toolbox.run("find_files", {"pattern": "*.py"})).output == "src/app.py"
    assert "src/" in (await toolbox.run("list_dir", {})).output
    bad = await toolbox.run("search_files", {"pattern": "("})
    assert not bad.ok                                           # invalid regex is reported, not raised


async def test_tool_errors_never_raise(box):
    toolbox, _, _ = box
    for name, arguments in [
        ("nope", {}), ("read_file", {}), ("read_file", {"path": 5}), ("read_file", {"path": "missing.txt"}),
        ("read_file", {"path": "a", "bogus": 1}), ("edit_file", {"path": "src/app.py", "old_string": "zzz", "new_string": ""}),
        ("edit_file", {"path": "src/app.py", "old_string": "", "new_string": "x"}), ("list_dir", {"path": "src/app.py"}),
    ]:
        result = await toolbox.run(name, arguments)
        assert not result.ok and result.output.startswith("Error")


async def test_tool_output_is_scrubbed_and_truncated(tmp_path):
    leaked = "sk-" + "Hj4" * 14
    (tmp_path / "config.txt").write_text(f"api key {leaked}\n" + "x" * 5000, encoding="utf-8")
    toolbox = Toolbox(tmp_path, approve=Approvals(), max_output_chars=1000)
    result = await toolbox.run("read_file", {"path": "config.txt"})
    assert leaked not in result.output and "omitted" in result.output and len(result.output) < 1200


# --- approvals --------------------------------------------------------------

async def test_declined_actions_do_nothing(tmp_path):
    approvals = Approvals(answer=False)
    toolbox = Toolbox(tmp_path, approve=approvals)
    result = await toolbox.run("write_file", {"path": "a.txt", "content": "x"})
    assert not result.ok and "declined" in result.output and not (tmp_path / "a.txt").exists()
    assert not (await toolbox.run("run_command", {"command": "echo hi"})).ok


async def test_approval_modes(tmp_path):
    approvals = Approvals()
    auto_edit = Toolbox(tmp_path, approve=approvals, approval_mode="auto_edit", command_timeout=20)
    await auto_edit.run("write_file", {"path": "a.txt", "content": "x"})
    assert approvals.asked == []                                # edits are automatic
    await auto_edit.run("run_command", {"command": "echo hi"})
    assert [name for name, _ in approvals.asked] == ["run_command"]   # commands still ask

    ask = Toolbox(tmp_path, approve=approvals)
    ask.remember_approval("write_file")
    ask.remember_approval("run_command")                        # must be ignored
    approvals.asked.clear()
    await ask.run("write_file", {"path": "b.txt", "content": "x"})
    await ask.run("run_command", {"command": "echo hi"})
    assert [name for name, _ in approvals.asked] == ["run_command"]


# --- shell ------------------------------------------------------------------

async def test_run_command_reports_output_and_exit_code(box):
    toolbox, _, _ = box
    result = await toolbox.run("run_command", {"command": "echo baton-ok"})
    assert "baton-ok" in result.output and "[exit code 0]" in result.output
    failed = await toolbox.run("run_command", {"command": "exit 3"})
    assert "[exit code 3]" in failed.output


async def test_run_command_times_out_and_is_killed(tmp_path):
    toolbox = Toolbox(tmp_path, approve=Approvals(), command_timeout=1.5)
    command = f'"{sys.executable}" -c "import time; time.sleep(30)"'
    result = await toolbox.run("run_command", {"command": command})
    assert not result.ok and "timed out" in result.output


async def test_shell_environment_carries_no_credentials(box, monkeypatch):
    toolbox, _, _ = box
    monkeypatch.setenv("GEMINI_API_KEY", "should-not-be-visible-1")
    monkeypatch.setenv("BATON_PASSPHRASE", "should-not-be-visible-2")
    monkeypatch.setenv("HARMLESS_SETTING", "visible")
    environment = clean_environment()
    assert "GEMINI_API_KEY" not in environment and "BATON_PASSPHRASE" not in environment
    assert environment["HARMLESS_SETTING"] == "visible"
    command = f'"{sys.executable}" -c "import os; print(sorted(os.environ))"'
    output = (await toolbox.run("run_command", {"command": command})).output
    assert "GEMINI_API_KEY" not in output and "HARMLESS_SETTING" in output


# --- turn loop --------------------------------------------------------------

class Recorder:
    def __init__(self) -> None:
        self.text: list[str] = []
        self.tools: list[str] = []
        self.events: list[str] = []
        self.turns = 0

    def on_text(self, text): self.text.append(text)
    def on_stream_restart(self): self.text.clear()
    def on_tool_start(self, name, arguments): self.tools.append(name)
    def on_tool_end(self, name, result): pass
    def on_session_event(self, event): self.events.append(event.kind)
    def on_turn_end(self): self.turns += 1


def tool_chunks(*calls: tuple[str, str, dict]) -> list[StreamChunk]:
    deltas = [{"index": i, "id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
              for i, (cid, name, args) in enumerate(calls)]
    return [StreamChunk(tool_calls=deltas), StreamChunk(finish_reason="tool_calls", usage=Usage(20, 5))]


def text_chunks(text: str) -> list[StreamChunk]:
    return [StreamChunk(text=text), StreamChunk(finish_reason="stop", usage=Usage(20, 5))]


def make_agent(tmp_path, count: int = 2, approve: bool = True, **sections):
    h = Harness(make_config(count, **sections))
    runtime = Runtime(h.config, tmp_path, h.pool, h.router, h.handoff, h.relay, client=None)  # type: ignore[arg-type]
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    recorder = Recorder()
    toolbox = Toolbox(tmp_path, approve=Approvals(approve), command_timeout=20)
    return Agent(runtime, toolbox, recorder), h, recorder


def assert_valid_transcript(messages: list[dict]) -> None:
    owed: set[str] = set()
    for message in messages:
        if message["role"] == "tool":
            assert message["tool_call_id"] in owed
            owed.discard(message["tool_call_id"])
        else:
            assert not owed, f"{message['role']} message while tool results are owed"
            owed = {c["id"] for c in message.get("tool_calls") or []}
    assert not owed


async def test_turn_runs_tools_then_answers(tmp_path):
    agent, h, recorder = make_agent(tmp_path)
    h.provider.stream_scripts[secret_for(0)] = [
        tool_chunks(("c1", "read_file", {"path": "a.txt"}), ("c2", "list_dir", {})),
        text_chunks("The file says alpha."),
    ]
    await agent.run_turn("what is in a.txt?")
    roles = [m["role"] for m in agent.session.messages]
    assert roles == ["system", "user", "assistant", "tool", "tool", "assistant"]
    assert agent.session.messages[3]["content"] == "alpha"
    assert recorder.tools == ["read_file", "list_dir"] and "".join(recorder.text) == "The file says alpha."
    assert recorder.turns == 1
    assert_valid_transcript(agent.session.messages)
    # The model was offered the tool schemas.
    assert h.provider.calls[0][1].tools == TOOL_SCHEMAS


async def test_malformed_and_failing_tool_calls_still_get_results(tmp_path):
    agent, h, _ = make_agent(tmp_path)
    bad = [StreamChunk(tool_calls=[
        {"index": 0, "id": "c1", "function": {"name": "read_file", "arguments": '["not", "an", "object"]'}},
        {"index": 1, "id": "c2", "function": {"name": "no_such_tool", "arguments": "{}"}},
        {"index": 2, "id": "c3", "function": {"name": "read_file", "arguments": '{"path": "../escape"}'}},
    ]), StreamChunk(finish_reason="tool_calls")]
    h.provider.stream_scripts[secret_for(0)] = [bad, text_chunks("done")]
    await agent.run_turn("go")
    results = [m["content"] for m in agent.session.messages if m["role"] == "tool"]
    assert len(results) == 3 and all(r.startswith("Error") for r in results)
    assert_valid_transcript(agent.session.messages)


async def test_interrupt_during_tools_leaves_a_valid_transcript(tmp_path):
    agent, h, _ = make_agent(tmp_path)
    h.provider.stream_scripts[secret_for(0)] = [
        tool_chunks(("c1", "read_file", {"path": "a.txt"}), ("c2", "read_file", {"path": "a.txt"})),
    ]
    calls = {"n": 0}
    original = agent.toolbox.run

    async def interrupted(name, arguments):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return await original(name, arguments)

    agent.toolbox.run = interrupted  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        await agent.run_turn("read twice")
    assert_valid_transcript(agent.session.messages)
    assert "Cancelled" in agent.session.messages[-1]["content"]


async def test_failed_turn_takes_the_user_message_back(tmp_path):
    agent, h, recorder = make_agent(tmp_path)
    for index in range(2):
        h.provider.stream_scripts[secret_for(index)] = [[err(ErrorKind.QUOTA_EXHAUSTED, retry_after=3600)]]
    with pytest.raises(AllKeysExhaustedError):
        await agent.run_turn("hello")
    assert [m["role"] for m in agent.session.messages] == ["system"]   # ready for a clean retry
    assert recorder.turns == 1


async def test_tool_iteration_limit_stops_runaway_loops(tmp_path):
    agent, h, _ = make_agent(tmp_path, count=1, agent={"max_tool_iterations": 3})
    looping = [tool_chunks((f"c{i}", "list_dir", {})) for i in range(3)]
    h.provider.stream_scripts[secret_for(0)] = [*looping, text_chunks("stopping here")]
    await agent.run_turn("loop forever")
    assert agent.session.messages[-1] == {"role": "assistant", "content": "stopping here"}
    assert len([m for m in agent.session.messages if m["role"] == "tool"]) == 3
    assert_valid_transcript(agent.session.messages)


async def test_agent_task_continues_across_a_key_handoff(tmp_path):
    """The headline behaviour: a multi-step task crosses a key boundary and finishes."""
    agent, h, recorder = make_agent(tmp_path, count=2, limits={"rpd": 20}, handoff={"min_messages": 4, "keep_recent_messages": 2})
    for _ in range(17):                                          # k0 is at 85% of its daily quota
        h.pool.record_success(h.key(0), Usage(1, 1))
    agent.session.active_key = "k0"
    h.provider.stream_scripts[secret_for(0)] = [tool_chunks(("c1", "read_file", {"path": "a.txt"}))]   # -> 90%
    h.provider.stream_scripts[secret_for(1)] = [
        tool_chunks(("c2", "write_file", {"path": "b.txt", "content": "beta"})),
        text_chunks("Both files are in place."),
    ]
    await agent.run_turn("copy the idea of a.txt into b.txt")
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "beta"
    assert agent.session.active_key == "k1" and "handoff" in recorder.events
    assert agent.session.messages[-1]["content"] == "Both files are in place."
    assert_valid_transcript(agent.session.messages)
    assert agent.session.messages[0]["role"] == "system"


def test_reset_keeps_only_the_system_prompt(tmp_path):
    agent, _, _ = make_agent(tmp_path)
    agent.session.messages += [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    agent.reset()
    assert [m["role"] for m in agent.session.messages] == ["system"]
    assert str(tmp_path) in agent.session.messages[0]["content"]
