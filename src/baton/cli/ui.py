"""Terminal presentation for the agent (Rich)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from .. import __version__
from ..agent.tools import ToolResult
from ..pool import KeyPool
from ..session import SessionEvent
from .select import can_use_menu

_STATUS_STYLE = {"ready": "green", "draining": "yellow", "cooldown": "yellow", "exhausted": "red", "disabled": "red"}

# Every slash command, shown in the dropdown while typing "/" and listed by /help.
# The ones marked `cli` run the matching `baton ...` command without leaving the chat.
SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help", "show all commands"),
    ("/model", "pick a model from a list"),
    ("/status", "key pool status and quota usage"),
    ("/models", "check each key's models against the provider (/models --fix repairs stale ones)"),
    ("/keys", "list your API keys"),
    ("/keys add", "add an API key"),
    ("/keys test", "send a tiny test request through each key"),
    ("/keys remove", "remove a key:  /keys remove NAME"),
    ("/keys enable", "put a key back in rotation:  /keys enable NAME"),
    ("/keys disable", "take a key out of rotation:  /keys disable NAME"),
    ("/token", "list proxy tokens"),
    ("/token new", "create a proxy token:  /token new NAME"),
    ("/doctor", "check the setup for problems"),
    ("/update", "update Baton to the latest version"),
    ("/approval", "ask | auto_edit | auto  (when the agent asks before acting)"),
    ("/handoff", "summarise the conversation now and continue from the summary"),
    ("/clear", "start a new conversation"),
    ("/exit", "quit (Ctrl+D also works)"),
]

def slash_matches(text: str) -> list[tuple[str, str]]:
    """Commands to offer for what has been typed so far (only a line that starts with "/")."""
    if not text.startswith("/") or "\n" in text:
        return []
    return [(name, description) for name, description in SLASH_COMMANDS
            if name.startswith(text.lower()) and name != text.lower()]


HELP = "[bold]Commands[/]  [dim](type / to see them as you type)[/]\n" + "\n".join(
    f"  {name:<15}{escape(description)}" for name, description in SLASH_COMMANDS
) + "\n\nEnd a line with \\ to continue typing on the next line. Ctrl+C cancels the current turn."


def keys_table(pool: KeyPool) -> Table:
    table = Table(box=None, pad_edge=False, header_style="bold")
    # Short headers and no-wrap identifiers keep the table readable at 80 columns.
    for column in ("key", "provider", "model", "status", "min", "day", "req/day", "tok/day"):
        table.add_column(column, no_wrap=True)
    table.add_column("note", overflow="fold")
    for row in pool.snapshot():
        style = _STATUS_STYLE.get(row["status"], "white")
        note = f"retry in {int(row['retry_in'])}s" if row["retry_in"] else ""
        if row["last_error"]:
            note = (note + "  " + row["last_error"]).strip()
        key = pool.keys[row["id"]]
        table.add_row(
            row["id"], row["provider"].replace("openai_compatible", "compatible"), row["model"],
            f"[{style}]{row['status']}[/]",
            f"{min(pool.short_utilization(key), 9.99):.0%}", f"{row['daily_utilization']:.0%}",
            str(row["requests_today"]), f"{row['tokens_today']:,}", escape(note[:70]),
        )
    return table


class TerminalUI:
    """Implements agent.AgentEvents and owns every byte written to the terminal."""

    def __init__(self, console: Console, pool: KeyPool, *, show_handoffs: bool = True) -> None:
        self.console = console
        self.pool = pool
        self.show_handoffs = show_handoffs
        self.active_key: str | None = None
        self._streaming = False
        self._status = None
        self._session = None
        self._toolbar: Callable[[], str] | None = None
        # Plain ASCII markers on consoles that cannot encode the nicer ones.
        fancy = (console.encoding or "").lower().startswith("utf") and not console.options.legacy_windows
        self.bullet, self.elbow, self.arrow = ("●", "⎿", "→") if fancy else ("*", "|", "->")
        self.chevron = "❯" if fancy else ">"

    # --- chrome -------------------------------------------------------------

    def banner(self, workspace: Path, model: str, approval: str) -> None:
        ready = sum(1 for row in self.pool.snapshot() if row["status"] in ("ready", "draining"))
        body = (
            f"[bold]Baton[/] {__version__}\n"
            f"[dim]workspace[/] {escape(str(workspace))}\n"
            f"[dim]keys[/] {ready}/{len(self.pool.keys)} ready   [dim]model[/] {escape(model)}   "
            f"[dim]approval[/] {approval}\n"
            "[dim]/help for commands[/]"
        )
        self.console.print(Panel(body, border_style="cyan", expand=False))

    def set_toolbar(self, render: Callable[[], str]) -> None:
        """`render` returns the text of the status bar under the prompt."""
        self._toolbar = render

    def _make_session(self):
        """A prompt_toolkit session: slash-command dropdown, arrow-key editing, status bar.

        History is kept in memory only: a key pasted into the prompt by mistake must not
        end up in a history file on disk.
        """
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.history import InMemoryHistory
        from prompt_toolkit.styles import Style

        class SlashCompleter(Completer):
            def get_completions(self, document, complete_event):
                text = document.text_before_cursor
                for name, description in slash_matches(text):
                    yield Completion(name, start_position=-len(text), display_meta=description)

        style = Style.from_dict({
            "bottom-toolbar": "noreverse fg:ansibrightblack bg:default",
            "completion-menu.completion": "bg:#2b2b2b fg:ansiwhite",
            "completion-menu.completion.current": "bg:ansicyan fg:ansiblack",
            "completion-menu.meta.completion": "bg:#2b2b2b fg:ansibrightblack",
            "completion-menu.meta.completion.current": "bg:ansicyan fg:ansiblack",
        })
        return PromptSession(
            completer=SlashCompleter(),
            complete_while_typing=True,
            history=InMemoryHistory(),
            bottom_toolbar=lambda: self._toolbar() if self._toolbar else "",
            style=style,
            reserve_space_for_menu=8,
        )

    def prompt(self) -> str:
        lines: list[str] = []
        first = True
        use_session = can_use_menu()
        if use_session and self._session is None:
            try:
                self._session = self._make_session()
            except Exception:  # noqa: BLE001 - a terminal prompt_toolkit cannot drive: use plain input
                use_session = False
        while True:
            if use_session and self._session is not None:
                marker = self.chevron if first else "·"
                line = self._session.prompt([("bold fg:ansicyan", f"{marker} ")])
            else:
                line = self.console.input("[bold cyan]>[/] " if first else "[dim].[/] ")
            if line.endswith("\\"):
                lines.append(line[:-1])
                first = False
                continue
            lines.append(line)
            return "\n".join(lines).strip()

    def info(self, message: str) -> None:
        self.console.print(f"[dim]{escape(message)}[/]")

    def error(self, message: str) -> None:
        self.console.print(f"[red]{escape(message)}[/]")

    def notice(self, message: str) -> None:
        """Something to tell the user in the middle of a turn (model swapped, provider busy)."""
        waiting = self._status is not None
        self._stop_waiting()
        self._end_stream()
        self.console.print(f"[dim]~ {escape(message)}[/]")
        if waiting:
            self.start_waiting()

    def show_status(self) -> None:
        self.console.print(keys_table(self.pool))

    def start_waiting(self) -> None:
        self._status = self.console.status("[dim]thinking[/]", spinner="dots")
        self._status.start()

    def _stop_waiting(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    # --- AgentEvents --------------------------------------------------------

    def on_text(self, text: str) -> None:
        self._stop_waiting()
        self._streaming = True
        # markup=False: model output must never be interpreted as Rich markup.
        self.console.print(text, end="", markup=False, highlight=False, soft_wrap=True)

    def _end_stream(self) -> None:
        if self._streaming:
            self.console.print()
            self._streaming = False

    def on_stream_restart(self) -> None:
        self._end_stream()
        self.console.print("[dim](connection dropped; restarting this reply on another key)[/]")
        self.start_waiting()

    def on_tool_start(self, name: str, arguments: dict[str, Any]) -> None:
        self._stop_waiting()
        self._end_stream()
        shown = ", ".join(f"{key}={json.dumps(value)[:80]}" for key, value in list(arguments.items())[:3]
                          if key != "content")
        self.console.print(f"[cyan]{self.bullet}[/] [bold]{escape(name)}[/]({escape(shown)})", highlight=False)

    def on_tool_end(self, name: str, result: ToolResult) -> None:
        style = "dim" if result.ok else "red"
        self.console.print(f"  [{style}]{self.elbow} {escape(result.summary or 'done')}[/]", highlight=False)
        self.start_waiting()

    def on_session_event(self, event: SessionEvent) -> None:
        if event.to_key:
            self.active_key = event.to_key
        if not self.show_handoffs:
            return
        self._stop_waiting()
        self._end_stream()
        if event.kind == "handoff":
            text = f"handed off {event.from_key} {self.arrow} {event.to_key} ({event.detail})"
        elif event.kind == "failover":
            text = f"switched key {event.from_key} {self.arrow} {event.to_key}"
        elif event.kind == "compaction":
            text = f"conversation compacted ({event.detail})"
        else:
            return
        self.console.print(f"[dim]~ {escape(text)}[/]")

    def on_turn_end(self) -> None:
        self._stop_waiting()
        self._end_stream()

    def footer(self, key_id: str | None) -> None:
        key = self.pool.keys.get(key_id or "")
        if key is None:
            return
        ready = sum(1 for row in self.pool.snapshot() if row["status"] in ("ready", "draining"))
        parts = [
            key.id,
            escape(key.config.default_model),
            f"day {min(self.pool.long_utilization(key), 9.99):.0%}",
            f"{ready}/{len(self.pool.keys)} keys ready",
        ]
        self.console.print(f"[dim]{' | '.join(parts)}[/]", highlight=False)

    # --- approvals ----------------------------------------------------------

    async def approve(self, name: str, description: str) -> bool | str:
        self._stop_waiting()
        self._end_stream()
        label = "Run command" if name == "run_command" else f"Allow {name} on"
        self.console.print(Panel(escape(description), title=label, border_style="yellow", expand=False))
        options = "[y]es / [n]o" if name == "run_command" else "[y]es / [n]o / [a]lways for this tool"
        try:
            answer = self.console.input(f"[yellow]{options}:[/] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if answer in ("a", "always") and name != "run_command":
            return "always"
        return answer in ("y", "yes")
