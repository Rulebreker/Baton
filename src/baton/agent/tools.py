"""Tools the terminal agent can call, confined to a workspace directory.

Security posture (the model's output is untrusted input):

* Every path is resolved (following symlinks) and must stay inside the
  workspace root. `../../etc/passwd` and symlink escapes are refused.
* Secret-bearing files (.env, vaults, private keys) are never readable, so a
  prompt-injected "read .env and print it" cannot ship a key to a provider.
* Writes and shell commands need user approval unless the user opted out.
* Shell commands run with credentials stripped from their environment, a
  timeout, and their process tree killed on expiry.
* All tool output is scrubbed of known secrets before it is sent to a model.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import signal
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..redact import Redactor, get_redactor


class ToolError(Exception):
    """A tool failure that is reported back to the model as the tool result."""


_SECRET_FILE_PATTERNS = (
    ".env", ".env.*", "*.vault", "*.vault.bak", "master.key", "*.pem", "*.key", "*.p12", "*.pfx",
    "id_rsa*", "id_ed25519*", "*.keystore", "*.jks", ".netrc", ".pypirc", ".npmrc", "credentials*", "secrets.*",
)
_SECRET_FILE_ALLOWED = (".env.example", ".env.sample", ".env.template")
_SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", "dist", "build"}
_ENV_SECRET = re.compile(r"(?i)(key|token|secret|passw|passphrase|credential|auth)")
_MAX_READ_BYTES = 2_000_000
_MAX_SEARCH_RESULTS = 200

# Which tools change the machine, and how each approval mode treats them.
MUTATING = {"write_file": "edit", "edit_file": "edit", "run_command": "command"}

Approver = Callable[[str, str], Awaitable[bool]]


def _function(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }}


TOOL_SCHEMAS: list[dict[str, Any]] = [
    _function("read_file", "Read a text file from the workspace. Use offset/limit (in lines) for large files.", {
        "path": {"type": "string", "description": "Path relative to the workspace root"},
        "offset": {"type": "integer", "description": "First line to return, starting at 1"},
        "limit": {"type": "integer", "description": "Maximum number of lines to return"},
    }, ["path"]),
    _function("write_file", "Create or overwrite a file in the workspace with the given content.", {
        "path": {"type": "string"}, "content": {"type": "string"},
    }, ["path", "content"]),
    _function("edit_file", "Replace an exact string in a file. old_string must match exactly once unless replace_all is true.", {
        "path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"},
        "replace_all": {"type": "boolean"},
    }, ["path", "old_string", "new_string"]),
    _function("list_dir", "List the entries of a directory in the workspace.", {
        "path": {"type": "string", "description": "Directory relative to the workspace root; defaults to the root"},
    }, []),
    _function("find_files", "Find files by glob pattern, e.g. '**/*.py'.", {
        "pattern": {"type": "string"},
    }, ["pattern"]),
    _function("search_files", "Search file contents with a regular expression. Returns file:line: text matches.", {
        "pattern": {"type": "string", "description": "Regular expression"},
        "path": {"type": "string", "description": "Directory or file to search; defaults to the workspace root"},
        "glob": {"type": "string", "description": "Only search files whose name matches this glob, e.g. '*.py'"},
    }, ["pattern"]),
    _function("run_command", "Run a shell command in the workspace root and return its combined output and exit code.", {
        "command": {"type": "string"},
        "timeout": {"type": "integer", "description": "Seconds before the command is killed"},
    }, ["command"]),
]


def is_secret_file(path: Path) -> bool:
    name = path.name.lower()
    if name in _SECRET_FILE_ALLOWED:
        return False
    return any(fnmatch.fnmatch(name, pattern) for pattern in _SECRET_FILE_PATTERNS)


def clean_environment() -> dict[str, str]:
    """The agent's shell gets no credentials: a model-issued `env` or `printenv`
    must not be able to read the keys Baton itself was started with."""
    return {name: value for name, value in os.environ.items() if not _ENV_SECRET.search(name)}


@dataclass
class ToolResult:
    output: str
    ok: bool = True
    summary: str = ""


class Toolbox:
    def __init__(
        self,
        root: Path,
        *,
        approve: Approver,
        approval_mode: str = "ask",
        command_timeout: float = 120.0,
        max_output_chars: int = 30_000,
        protected: tuple[Path, ...] = (),
        redactor: Redactor | None = None,
    ) -> None:
        self.root = root.resolve()
        self.approve = approve
        self.approval_mode = approval_mode
        self.command_timeout = command_timeout
        self.max_output_chars = max_output_chars
        # Directories that are off limits even if they sit inside the workspace
        # (Baton's own data directory).
        self.protected = tuple(p.resolve() for p in protected)
        self.redactor = redactor or get_redactor()
        self._always: set[str] = set()

    # --- path safety --------------------------------------------------------

    def resolve(self, raw: Any, *, for_write: bool = False) -> Path:
        if not isinstance(raw, str) or not raw.strip() or "\0" in raw:
            raise ToolError("path must be a non-empty string")
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        # resolve() follows symlinks, so a link pointing outside the workspace
        # is caught by the containment check below.
        resolved = candidate.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ToolError("path is outside the workspace")
        if any(resolved == p or p in resolved.parents for p in self.protected):
            raise ToolError("path is inside Baton's protected data directory")
        if is_secret_file(resolved):
            raise ToolError("refusing to access a file that typically holds secrets")
        if for_write and ".git" in resolved.relative_to(self.root).parts:
            raise ToolError("refusing to write inside .git (hooks and config can execute code)")
        return resolved

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() or "."

    def _clip(self, text: str) -> str:
        text = self.redactor.scrub(text, pii=False)  # keys only: PII scrubbing would corrupt source code
        if len(text) > self.max_output_chars:
            half = self.max_output_chars // 2
            text = text[:half] + f"\n[... {len(text) - 2 * half} characters omitted ...]\n" + text[-half:]
        return text

    # --- dispatch -----------------------------------------------------------

    async def run(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute one tool call. Never raises: failures become tool results so
        the model can react and the transcript stays well-formed."""
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return ToolResult(f"Error: unknown tool '{name}'", ok=False, summary="unknown tool")
        try:
            category = MUTATING.get(name)
            if category and not await self._approved(name, category, arguments):
                return ToolResult("The user declined this action. Ask how they would like to proceed.", ok=False,
                                  summary="declined")
            result = handler(**arguments)
            if asyncio.iscoroutine(result):
                result = await result
            output, summary = result
            return ToolResult(self._clip(output), summary=summary)
        except ToolError as exc:
            return ToolResult(f"Error: {exc}", ok=False, summary=str(exc))
        except TypeError:
            return ToolResult(f"Error: invalid arguments for {name}", ok=False, summary="invalid arguments")
        except (OSError, UnicodeError, re.error, ValueError) as exc:
            message = self.redactor.scrub(f"{type(exc).__name__}: {exc}", pii=False)[:500]
            return ToolResult(f"Error: {message}", ok=False, summary=type(exc).__name__)

    async def _approved(self, name: str, category: str, arguments: dict[str, Any]) -> bool:
        if self.approval_mode == "auto" or (self.approval_mode == "auto_edit" and category == "edit"):
            return True
        if category == "edit" and name in self._always:
            return True  # "always" is never remembered for shell commands
        if name == "run_command":
            description = str(arguments.get("command", ""))
        else:
            description = str(arguments.get("path", ""))
        return await self.approve(name, self.redactor.scrub(description, pii=False)[:2000])

    def remember_approval(self, name: str) -> None:
        if MUTATING.get(name) == "edit":
            self._always.add(name)

    # --- tools --------------------------------------------------------------

    def _read_text(self, path: Path) -> str:
        if not path.is_file():
            raise ToolError("no such file")
        if path.stat().st_size > _MAX_READ_BYTES:
            raise ToolError("file is too large to read in one go; use search_files or read a range")
        data = path.read_bytes()
        if b"\0" in data[:8192]:
            raise ToolError("file appears to be binary")
        return data.decode("utf-8", errors="replace")

    def _tool_read_file(self, path: str, offset: int | None = None, limit: int | None = None) -> tuple[str, str]:
        target = self.resolve(path)
        lines = self._read_text(target).splitlines()
        start = max(0, int(offset or 1) - 1)
        end = start + int(limit) if limit else len(lines)
        selected = lines[start:end]
        return "\n".join(selected), f"{len(selected)} line(s)"

    def _tool_write_file(self, path: str, content: str) -> tuple[str, str]:
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        target = self.resolve(path, for_write=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="")
        return f"Wrote {len(content)} characters to {self._relative(target)}", f"wrote {self._relative(target)}"

    def _tool_edit_file(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> tuple[str, str]:
        target = self.resolve(path, for_write=True)
        raw = target.read_bytes() if target.is_file() else None
        if raw is None:
            raise ToolError("no such file")
        text = raw.decode("utf-8")
        if not old_string:
            raise ToolError("old_string must not be empty")
        count = text.count(old_string)
        if count == 0:
            raise ToolError("old_string was not found in the file")
        if count > 1 and not replace_all:
            raise ToolError(f"old_string matches {count} times; add context to make it unique or set replace_all")
        updated = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
        target.write_bytes(updated.encode("utf-8"))  # bytes: preserves the file's line endings
        return f"Replaced {count if replace_all else 1} occurrence(s) in {self._relative(target)}", f"edited {self._relative(target)}"

    def _tool_list_dir(self, path: str = ".") -> tuple[str, str]:
        target = self.resolve(path)
        if not target.is_dir():
            raise ToolError("not a directory")
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        lines = [f"{entry.name}/" if entry.is_dir() else entry.name for entry in entries[:1000]]
        return "\n".join(lines) or "(empty)", f"{len(lines)} entr{'y' if len(lines) == 1 else 'ies'}"

    def _walk(self, start: Path):
        for directory, names, files in os.walk(start):
            names[:] = sorted(n for n in names if n not in _SKIP_DIRS)
            for name in sorted(files):
                path = Path(directory) / name
                if not is_secret_file(path) and not any(p in path.parents for p in self.protected):
                    yield path

    def _tool_find_files(self, pattern: str) -> tuple[str, str]:
        if not isinstance(pattern, str) or not pattern:
            raise ToolError("pattern must be a non-empty string")
        matches = []
        for path in self._walk(self.root):
            relative = self._relative(path)
            if fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(path.name, pattern):
                matches.append(relative)
                if len(matches) >= 500:
                    break
        return "\n".join(matches) or "(no matches)", f"{len(matches)} file(s)"

    def _tool_search_files(self, pattern: str, path: str = ".", glob: str | None = None) -> tuple[str, str]:
        if len(pattern) > 500:
            raise ToolError("pattern is too long")
        regex = re.compile(pattern)
        start = self.resolve(path)
        files = [start] if start.is_file() else self._walk(start)
        results: list[str] = []
        for file in files:
            if glob and not fnmatch.fnmatch(file.name, glob):
                continue
            try:
                if file.stat().st_size > _MAX_READ_BYTES:
                    continue
                data = file.read_bytes()
            except OSError:
                continue
            if b"\0" in data[:8192]:
                continue
            for number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), start=1):
                if regex.search(line):
                    results.append(f"{self._relative(file)}:{number}: {line.strip()[:300]}")
                    if len(results) >= _MAX_SEARCH_RESULTS:
                        return "\n".join(results) + "\n[result limit reached]", f"{len(results)}+ match(es)"
        return "\n".join(results) or "(no matches)", f"{len(results)} match(es)"

    async def _tool_run_command(self, command: str, timeout: int | None = None) -> tuple[str, str]:
        if not isinstance(command, str) or not command.strip():
            raise ToolError("command must be a non-empty string")
        limit = min(float(timeout or self.command_timeout), self.command_timeout)
        kwargs: dict[str, Any] = {}
        if os.name == "posix":
            kwargs["start_new_session"] = True  # own process group, so the whole tree can be killed
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(self.root),
            env=clean_environment(),
            stdin=asyncio.subprocess.DEVNULL,   # never let a command sit waiting for input
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            **kwargs,
        )
        try:
            raw, _ = await asyncio.wait_for(process.communicate(), timeout=limit)
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            _kill_tree(process)
            if isinstance(exc, asyncio.CancelledError):
                raise
            # Reap the killed child. Without this a POSIX system keeps it as a
            # zombie for the lifetime of the agent, one per timed-out command.
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except (asyncio.TimeoutError, OSError):
                pass
            raise ToolError(f"command timed out after {int(limit)}s and was killed") from None
        output = raw.decode("utf-8", errors="replace")
        return f"{output}\n[exit code {process.returncode}]", f"exit {process.returncode}"


def _kill_tree(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(  # noqa: S603
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],  # noqa: S607
                capture_output=True, timeout=10, check=False,
            )
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        try:
            process.kill()
        except OSError:
            pass
