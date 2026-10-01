"""`baton` command-line entry point."""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import stat
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .. import __version__
from ..config import (
    BatonConfig,
    KeyConfig,
    Limits,
    find_config,
    load_config,
    load_dotenv,
    save_config,
)
from ..errors import AllKeysExhaustedError, BatonError, ConfigError, ProviderError, VaultError
from ..logging_setup import setup_logging
from ..paths import baton_home, ensure_private_dir
from ..pool import clear_parked_state
from ..providers import PROVIDER_DEFAULTS
from ..providers.base import ChatRequest
from ..redact import scrub
from ..runtime import Runtime
from ..vault import PASSPHRASE_ENV, VAULT_FILE, Vault, peek_vault_mode, raw_key_location
from .select import can_use_menu, choose

log = logging.getLogger("baton.cli")
console = Console()
stderr = Console(stderr=True)

app = typer.Typer(
    add_completion=False,
    help="Baton pools LLM API keys, rotates them, and hands work off between keys without losing context.",
)
keys_app = typer.Typer(help="Manage the pooled provider API keys.", no_args_is_help=True)
token_app = typer.Typer(help="Manage client tokens for the local proxy endpoint.", no_args_is_help=True)
app.add_typer(keys_app, name="keys")
app.add_typer(token_app, name="token")

PROVIDER_CHOICES = [
    ("gemini", "Google Gemini"),
    ("openai", "OpenAI"),
    ("anthropic", "Anthropic Claude"),
    ("openai_compatible", "Other (Groq, OpenRouter, Ollama, LM Studio, ...)"),
]


# --- shared helpers ---------------------------------------------------------

def fail(message: str) -> None:
    stderr.print(f"[red]error:[/] {escape(scrub(message))}")
    raise typer.Exit(1)


def is_interactive() -> bool:
    """True when a person is at the keyboard (so prompts and approvals are possible)."""
    return sys.stdin.isatty()


def config_path(ctx: typer.Context) -> Path:
    return find_config((ctx.obj or {}).get("config"))


def prepare_environment(*, console_logs: bool = False, level: str = "INFO") -> Path:
    home = ensure_private_dir(baton_home())
    # Only Baton's own private directory is trusted as a source of settings. A
    # `.env` in the working directory belongs to whatever project the agent is
    # pointed at and must not be able to redirect Baton (BATON_CONFIG, ...).
    load_dotenv(home / ".env")
    setup_logging(home / "logs", level=level, console=console_logs)
    return home


def load(ctx: typer.Context, *, console_logs: bool = False) -> tuple[BatonConfig, Path, Path]:
    home = prepare_environment(console_logs=console_logs)
    path = config_path(ctx)
    try:
        config = load_config(path)
    except ConfigError as exc:
        fail(str(exc))
    setup_logging(home / "logs", level=config.logging.level, console=console_logs)
    return config, path, home


def ask_passphrase() -> str:
    if not sys.stdin.isatty():
        raise VaultError(f"vault is passphrase-protected; set {PASSPHRASE_ENV} for non-interactive use")
    console.print("[dim]This vault was made with a passphrase. Baton no longer uses one: enter it this one last "
                  "time and the vault is converted, so you are never asked again.[/]")
    return typer.prompt("Vault passphrase", hide_input=True)


def open_vault(home: Path, *, required: bool = True) -> Vault | None:
    path = home / VAULT_FILE
    if not path.is_file():
        if required:
            fail(f"no vault found at {path}. Run `baton init` first.")
        return None
    try:
        vault = Vault.open(path, passphrase_provider=ask_passphrase)
    except VaultError as exc:
        fail(str(exc))
        return None
    if vault.mode == "passphrase":
        # Legacy vault: convert it now that it is open, so this was the last passphrase prompt.
        try:
            where = vault.migrate_to_keyfile()
        except VaultError as exc:
            stderr.print(f"[yellow]warning:[/] could not convert the vault ({escape(scrub(str(exc)))}); "
                         "it keeps its passphrase for now.")
        else:
            console.print(f"[green]Vault converted:[/] no passphrase needed from now on (key kept in the {where}); "
                          "the old passphrase-protected copy was erased.")
    return vault


def build_runtime(ctx: typer.Context, *, console_logs: bool = False) -> Runtime:
    config, _path, home = load(ctx, console_logs=console_logs)
    if not config.keys:
        fail("no API keys are configured. Run `baton init` or `baton keys add`.")
    needs_vault = any(key.secret == "vault" for key in config.keys)
    vault = open_vault(home, required=needs_vault)
    try:
        runtime = Runtime.build(config, vault, home=home)
    except BatonError as exc:
        fail(str(exc))
        raise typer.Exit(1) from None
    for problem in runtime.problems:
        stderr.print(f"[yellow]warning:[/] {escape(scrub(problem))}")
    return runtime


def prompt_optional_int(label: str) -> int | None:
    while True:
        raw = typer.prompt(label, default="", show_default=False).strip()
        if not raw:
            return None
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        console.print("[yellow]Enter a positive whole number, or leave blank.[/]")


def read_secret_key() -> str:
    """API keys are only ever typed into a hidden prompt or piped on stdin,
    never passed as a command-line argument (which would land in shell history
    and in the process list)."""
    if sys.stdin.isatty():
        return typer.prompt("API key (input hidden)", hide_input=True).strip()
    return sys.stdin.readline().strip()


def unique_key_id(config: BatonConfig, provider: str) -> str:
    stem = provider.replace("_", "-")
    number = sum(1 for key in config.keys if key.provider == provider) + 1
    while config.key(f"{stem}-{number}") is not None:
        number += 1
    return f"{stem}-{number}"


def add_key_interactive(config: BatonConfig, vault: Vault, *, advanced: bool = False) -> KeyConfig | None:
    """Ask for the provider and the key. The name, model list and quota limits are chosen
    automatically unless `advanced` is set (or the provider has no built-in defaults)."""
    provider = choose("Which provider is this key for?", PROVIDER_CHOICES, default="gemini")
    defaults = PROVIDER_DEFAULTS[provider]
    key_id = unique_key_id(config, provider)
    if advanced:
        key_id = typer.prompt("Name for this key", default=key_id).strip().lower()
    base_url = None
    if defaults.base_url is None:
        base_url = typer.prompt("Base URL (e.g. https://api.groq.com/openai/v1)").strip()
        models = [typer.prompt("Model name (e.g. llama-3.3-70b-versatile)").strip()]
    else:
        # "auto": Baton asks the provider which models exist right now, so nothing goes stale.
        models = ["auto"]
        if advanced:
            models = [typer.prompt("Model (or 'auto' to pick the best available)", default="auto").strip()]
    limits = Limits()
    if advanced:
        console.print("[dim]Quota limits let Baton hand off *before* a key runs dry. Leave blank if unknown;\n"
                      "Baton then relies on provider headers and 429 responses.[/]")
        limits = Limits(
            rpm=prompt_optional_int("  Requests per minute"),
            rpd=prompt_optional_int("  Requests per day"),
            tpm=prompt_optional_int("  Tokens per minute"),
            tpd=prompt_optional_int("  Tokens per day"),
        )
    secret = read_secret_key()
    model = models[0]
    try:
        key = KeyConfig(id=key_id, provider=provider, base_url=base_url, models=list(dict.fromkeys(models)),
                        limits=limits)
        if config.key(key.id) is not None:
            raise ValueError(f"a key named '{key.id}' already exists")
        vault.set_key(key.id, secret, host=urlparse(str(key.base_url)).hostname)
    except (ValueError, VaultError) as exc:
        console.print(f"[red]Could not add key:[/] {escape(scrub(str(exc)))}")
        return None
    config.keys.append(key)
    console.print(f"[green]Added[/] {key.id} ({provider}, {model}). The key is encrypted in the vault.")
    return key


async def test_key(runtime: Runtime, key_id: str) -> tuple[bool, str]:
    request = ChatRequest(model="auto", messages=[{"role": "user", "content": "Reply with the single word: OK"}],
                          max_tokens=64)
    try:
        route = await runtime.router.complete(request, pin=key_id)
        return True, f"ok ({route.model}, {route.result.usage.total_tokens} tokens)"
    except AllKeysExhaustedError as exc:
        key = runtime.pool.keys.get(key_id)
        return False, (key.last_error if key and key.last_error else str(exc))
    except BatonError as exc:
        return False, str(exc)


# --- chat helpers -----------------------------------------------------------

# Slash commands that run the matching `baton ...` command without leaving the chat.
INLINE_COMMANDS: dict[str, Callable[[list[str]], list[str]]] = {
    "/keys": lambda parts: ["keys", *(parts or ["list"])],
    "/token": lambda parts: ["token", *(parts or ["list"])],
    "/doctor": lambda parts: ["doctor", *parts],
    "/models": lambda parts: ["models", *parts],
    "/update": lambda parts: ["update"],
    "/version": lambda parts: ["--version"],
}


def normalize_input(text: str) -> str:
    """Accept what people naturally type at the chat prompt: `exit`, `quit`, and the shell form
    `baton keys list` (which would otherwise be sent to the model as a message)."""
    lowered = text.strip().lower()
    if lowered in ("exit", "quit"):
        return "/exit"
    first, _, rest = text.strip().partition(" ")
    if first.lower() == "baton" and rest.strip():
        return "/" + rest.strip()
    return text


def run_inline(ctx: typer.Context, args: list[str]) -> None:
    """Run `baton <args>` inside the running chat. Any failure is reported, never fatal."""
    config = (ctx.obj or {}).get("config")
    full = ([] if config is None else ["--config", str(config)]) + args
    try:
        typer.main.get_command(app).main(args=full, prog_name="baton", standalone_mode=False)
    except (typer.Exit, SystemExit):
        pass
    except (typer.Abort, KeyboardInterrupt):
        console.print("[dim]cancelled[/]")
    except Exception as exc:  # noqa: BLE001 - e.g. a usage error such as a missing argument
        console.print(f"[red]{escape(scrub(str(exc)) or type(exc).__name__)}[/]")


def exhausted_hint(kinds: dict[str, str]) -> str:
    """Advice that matches what actually happened to the keys: one sentence per cause present."""
    present = set(kinds.values())
    hints = []
    if "model_not_found" in present:
        hints.append("A model was not found: pick another with /model, or run /models --fix.")
    if "auth" in present:
        hints.append("A key was rejected: replace it with /keys remove NAME, then /keys add.")
    if "overloaded" in present:
        hints.append("The provider is busy, not your keys: wait a minute and send it again.")
    if present & {"rate_limit", "quota_exhausted"}:
        hints.append("Some keys are out of quota: /status shows when they retry.")
    return " ".join(hints) or "Check /status, add keys with /keys add, or try again in a moment."


# --- root -------------------------------------------------------------------

@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to the config file."),
    version: bool = typer.Option(False, "--version", help="Show the version and exit."),
) -> None:
    if version:
        console.print(f"baton {__version__}")
        raise typer.Exit()
    ctx.obj = {"config": config}
    if ctx.invoked_subcommand is None:
        run_chat(ctx, "auto", None, None, None)


# --- init -------------------------------------------------------------------

@app.command()
def init(
    ctx: typer.Context,
    advanced: bool = typer.Option(False, "--advanced", help="Also ask for a key name, default model and quota limits."),
) -> None:
    """First-run wizard: create the encrypted vault and add your first key."""
    home = prepare_environment()
    path = config_path(ctx)
    console.print(f"[bold]Baton setup[/]\nData directory: {escape(str(home))}\nConfig file:    {escape(str(path))}\n")

    if path.is_file():
        try:
            config = load_config(path)
        except ConfigError as exc:
            fail(str(exc))
        console.print(f"Existing configuration found with {len(config.keys)} key(s); new keys will be added to it.\n")
    else:
        config = BatonConfig()

    vault_path = home / VAULT_FILE
    if vault_path.is_file():
        vault = open_vault(home)
    else:
        # No password to type: the vault is encrypted with a random key kept in the OS keychain,
        # or in a private file next to it. (Vaults made earlier with a passphrase still open.)
        try:
            vault = Vault.create(vault_path)
        except VaultError as exc:
            fail(str(exc))
        console.print("[green]Encrypted key vault created.[/]\n")
    assert vault is not None  # noqa: S101 - open_vault exits on failure

    console.print("[bold]Add an API key[/] (you can add more later with `baton keys add`)")
    added: list[KeyConfig] = []
    while True:
        key = add_key_interactive(config, vault, advanced=advanced)
        if key:
            added.append(key)
            save_config(path, config)  # save after each key: an interrupted wizard loses nothing
        if not typer.confirm("Add another key?", default=not config.keys):
            break

    if added and typer.confirm("Test the new key(s) now with one tiny request each?", default=True):
        try:
            runtime = Runtime.build(config, vault, home=home)
        except BatonError as exc:
            fail(str(exc))

        async def run_tests() -> None:
            try:
                for key in added:
                    ok, detail = await test_key(runtime, key.id)
                    colour = "green" if ok else "red"
                    console.print(f"  [{colour}]{'ok  ' if ok else 'FAIL'}[/] {key.id}: {escape(scrub(detail))}")
            finally:
                await runtime.aclose()

        asyncio.run(run_tests())

    save_config(path, config)
    console.print("[green]Setup complete.[/] Next:\n"
                  "  baton                  start the terminal agent in the current directory\n"
                  "  baton token new NAME   create a token, then `baton serve` for the OpenAI-compatible endpoint\n"
                  "  baton status           show key health and quota usage")


# --- keys -------------------------------------------------------------------

@keys_app.command("add")
def keys_add(
    ctx: typer.Context,
    advanced: bool = typer.Option(False, "--advanced", help="Also ask for a key name, default model and quota limits."),
) -> None:
    """Add a key interactively. The key itself is read from a hidden prompt or stdin, never from argv."""
    config, path, home = load(ctx)
    vault = open_vault(home)
    if vault and add_key_interactive(config, vault, advanced=advanced):
        save_config(path, config)


@keys_app.command("list")
def keys_list(ctx: typer.Context) -> None:
    """List configured keys (never prints the keys themselves)."""
    config, _path, _home = load(ctx)
    table = Table(box=None, header_style="bold")
    for column in ("id", "provider", "models", "secret", "enabled", "rpm", "rpd", "tpm", "tpd"):
        table.add_column(column, overflow="fold")  # fold, never truncate: ids must stay copyable
    for key in config.keys:
        source = key.secret
        if key.secret.startswith("env:"):
            source += " (set)" if os.environ.get(key.secret[4:]) else " (NOT SET)"
        limits = key.limits
        table.add_row(key.id, key.provider, ", ".join(key.models), source, "yes" if key.enabled else "no",
                      *[str(v) if v else "-" for v in (limits.rpm, limits.rpd, limits.tpm, limits.tpd)])
    console.print(table if config.keys else "No keys configured. Run `baton keys add`.")


@keys_app.command("remove")
def keys_remove(ctx: typer.Context, key_id: str, yes: bool = typer.Option(False, "--yes", "-y")) -> None:
    """Remove a key from the config and delete its secret from the vault."""
    config, path, home = load(ctx)
    key = config.key(key_id)
    if key is None:
        fail(f"no key named '{key_id}'")
    if not yes and not typer.confirm(f"Remove key '{key_id}' and delete its stored secret?"):
        raise typer.Exit()
    config.keys.remove(key)
    save_config(path, config)
    vault = open_vault(home, required=False)
    if vault:
        vault.remove_key(key_id)
    console.print(f"Removed {key_id}.")


def _set_enabled(ctx: typer.Context, key_id: str, enabled: bool) -> None:
    config, path, home = load(ctx)
    key = config.key(key_id)
    if key is None:
        fail(f"no key named '{key_id}'")
    key.enabled = enabled
    save_config(path, config)
    if enabled:
        clear_parked_state(home / "state.json", key_id)
    console.print(f"{key_id} {'enabled' if enabled else 'disabled'}.")


@keys_app.command("enable")
def keys_enable(ctx: typer.Context, key_id: str) -> None:
    """Put a key back into rotation."""
    _set_enabled(ctx, key_id, True)


@keys_app.command("disable")
def keys_disable(ctx: typer.Context, key_id: str) -> None:
    """Take a key out of rotation without deleting it."""
    _set_enabled(ctx, key_id, False)


@keys_app.command("test")
def keys_test(ctx: typer.Context, key_id: str | None = typer.Argument(None)) -> None:
    """Send one tiny request through each key (or one key) and report the outcome."""
    runtime = build_runtime(ctx)
    targets = [key_id] if key_id else list(runtime.pool.keys)
    if key_id and key_id not in runtime.pool.keys:
        fail(f"no usable key named '{key_id}'")

    async def run() -> bool:
        all_ok = True
        try:
            for target in targets:
                ok, detail = await test_key(runtime, target)
                all_ok = all_ok and ok
                colour = "green" if ok else "red"
                console.print(f"[{colour}]{'ok  ' if ok else 'FAIL'}[/] {target}: {escape(scrub(detail))}")
        finally:
            await runtime.aclose()
        return all_ok

    if not asyncio.run(run()):
        raise typer.Exit(1)


# --- tokens -----------------------------------------------------------------

@token_app.command("new")
def token_new(ctx: typer.Context, name: str = typer.Argument("default")) -> None:
    """Create a proxy client token. It is displayed once and stored only as a hash."""
    home = prepare_environment()
    vault = open_vault(home)
    try:
        token = vault.create_token(name)
    except VaultError as exc:
        fail(str(exc))
    console.print(f"Token for client '{name}' (shown once):\n")
    console.print(f"    {token}\n", markup=False, highlight=False)
    console.print("Use it as the API key with base URL http://127.0.0.1:8787/v1")


@token_app.command("list")
def token_list(ctx: typer.Context) -> None:
    """List proxy client names (tokens cannot be shown again)."""
    home = prepare_environment()
    vault = open_vault(home)
    names = vault.token_names()
    if not names:
        console.print("No tokens. Create one with `baton token new NAME`.")
    for name, created in sorted(names.items()):
        console.print(f"{name}  [dim]created {time.strftime('%Y-%m-%d', time.localtime(created))}[/]")


@token_app.command("revoke")
def token_revoke(ctx: typer.Context, name: str) -> None:
    """Revoke a proxy client token immediately (restart `baton serve` to apply)."""
    home = prepare_environment()
    vault = open_vault(home)
    if not vault.revoke_token(name):
        fail(f"no token named '{name}'")
    console.print(f"Revoked '{name}'. Restart any running `baton serve` so it reloads the vault.")


# --- status / doctor --------------------------------------------------------

@app.command()
def status(ctx: typer.Context) -> None:
    """Show each key's health and quota usage."""
    from .ui import keys_table

    runtime = build_runtime(ctx)
    console.print(keys_table(runtime.pool))
    asyncio.run(runtime.client.aclose())


@app.command()
def doctor(ctx: typer.Context) -> None:
    """Check the installation for configuration and security problems."""
    home = prepare_environment()
    path = config_path(ctx)
    problems = 0

    def report(ok: bool, text: str, *, warn: bool = False) -> None:
        nonlocal problems
        if ok:
            console.print(f"[green] ok [/] {escape(text)}")
        elif warn:
            console.print(f"[yellow]warn[/] {escape(text)}")
        else:
            problems += 1
            console.print(f"[red]FAIL[/] {escape(text)}")

    config: BatonConfig | None = None
    try:
        config = load_config(path)
        report(True, f"configuration is valid ({len(config.keys)} key(s))")
    except ConfigError as exc:
        report(False, str(exc))

    vault_file = home / VAULT_FILE
    report(vault_file.is_file(), f"vault present at {vault_file}", warn=True)
    vault_mode = peek_vault_mode(vault_file) if vault_file.is_file() else None
    if vault_mode == "passphrase":
        report(False, "vault mode: legacy passphrase. It is converted automatically the next time a command "
                      "opens it (you are asked for the passphrase once).", warn=True)
    elif vault_mode == "keyfile":
        location = raw_key_location(home)
        report(location is not None, f"vault mode: keyfile, key stored in the {location or 'NOWHERE (key missing)'}")
    if os.name == "posix":
        mode = stat.S_IMODE(home.stat().st_mode)
        report(mode & 0o077 == 0, f"data directory is private (mode {oct(mode)})")
        for name in (VAULT_FILE, "master.key", "state.json"):
            target = home / name
            if target.is_file():
                report(stat.S_IMODE(target.stat().st_mode) & 0o077 == 0, f"{name} is readable by the owner only")
    if (home / "master.key").is_file():
        report(False, "vault key is stored in a file (keyfile mode) because no OS keychain is available here: "
                      "anyone who can read the data directory can decrypt the vault. Keep that directory private.",
               warn=True)

    if config is not None:
        for key in config.keys:
            if key.secret.startswith("env:"):
                report(bool(os.environ.get(key.secret[4:])), f"key '{key.id}': environment variable {key.secret[4:]} is set")
            if not (key.limits.rpd or key.limits.tpd):
                report(False, f"key '{key.id}' has no daily limit configured: planned handoff at "
                              f"{config.rotation.handoff_threshold:.0%} cannot trigger for it "
                              "(Baton will still fail over when the provider reports exhaustion)", warn=True)
        from ..proxy import is_loopback

        report(is_loopback(config.proxy.host) or config.proxy.allow_remote,
               f"proxy bind address {config.proxy.host} is loopback (or remote access was explicitly allowed)")
        if not is_loopback(config.proxy.host):
            report(False, "proxy is configured to listen on the network; put TLS in front of it", warn=True)

    local_env = Path.cwd() / ".env"
    if local_env.is_file() and (Path.cwd() / ".git").exists():
        ignored = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=Path.cwd(), capture_output=True, check=False)
        report(ignored.returncode == 0, ".env in this repository is git-ignored")
    if (Path.cwd() / ".githooks" / "pre-commit").is_file():
        hooks = subprocess.run(["git", "config", "core.hooksPath"], cwd=Path.cwd(), capture_output=True, text=True, check=False)
        report(hooks.stdout.strip() == ".githooks", "secret-scanning pre-commit hook is enabled "
                                                    "(run scripts/install-hooks if not)", warn=True)

    console.print("\n[green]No problems found.[/]" if not problems else f"\n[red]{problems} problem(s) found.[/]")
    if problems:
        raise typer.Exit(1)


# --- models -----------------------------------------------------------------

@app.command()
def models(
    ctx: typer.Context,
    fix: bool = typer.Option(False, "--fix", help="Rewrite pinned models the provider no longer offers."),
) -> None:
    """Check each key's configured models against what its provider offers right now."""
    from ..models import audit_pinned_models

    runtime = build_runtime(ctx)
    catalog = runtime.router.catalog

    async def inspect() -> tuple[list, dict[str, str | None]]:
        try:
            stale = await audit_pinned_models(runtime.pool, catalog, refresh=True)
            resolved: dict[str, str | None] = {}
            for key in runtime.pool.keys.values():
                if key.config.default_model.lower() in ("auto", "default", "baton"):
                    listed = await catalog.available(key)
                    resolved[key.id] = await catalog.best(key) if listed else None
            return stale, resolved
        finally:
            await runtime.aclose()

    stale, resolved = asyncio.run(inspect())
    table = Table(box=None, header_style="bold")
    for column in ("key", "provider", "models", "now"):
        table.add_column(column, overflow="fold")
    bad = {(item.key_id, item.model): item for item in stale}
    for key in runtime.pool.keys.values():
        shown = ", ".join(f"[red]{escape(m)}[/]" if (key.id, m) in bad else escape(m) for m in key.config.models)
        if key.id in resolved:
            now = escape(resolved[key.id]) if resolved[key.id] else "[yellow]list unavailable[/]"
        else:
            now = ", ".join(f"{escape(i.model)} -> {escape(i.suggestion or '?')}" for i in stale if i.key_id == key.id)
        table.add_row(key.id, key.config.provider, shown, now or "[green]ok[/]")
    console.print(table)
    if not stale:
        console.print("[green]Nothing to fix.[/]")
        return
    fixable = [item for item in stale if item.suggestion]
    if not fix:
        console.print(f"\n{len(stale)} pinned model(s) are no longer offered. Baton works around this automatically; "
                      "make it permanent with `baton models --fix` (or `/models --fix` in the chat).")
        return
    config = runtime.config
    for item in fixable:
        key = config.key(item.key_id)
        if key is not None:
            key.models = list(dict.fromkeys(item.suggestion if m == item.model else m for m in key.models))
    save_config(config_path(ctx), config)
    console.print(f"[green]Updated {len(fixable)} model(s) in the config.[/]")
    for item in stale:
        if not item.suggestion:
            console.print(f"[yellow]No replacement could be chosen for {item.key_id}: '{escape(item.model)}'. "
                          "Edit `models:` in the config.[/]")


# --- update -----------------------------------------------------------------

@app.command()
def update() -> None:
    """Update Baton to the latest version (git pull for a checkout, pip upgrade otherwise)."""
    from .update import run_update

    with console.status("[dim]checking for updates[/]", spinner="dots"):
        result = run_update()
    colour = "green" if result.ok else "red"
    console.print(f"[{colour}]{escape(result.message)}[/]")
    if result.changed:
        console.print("Restart Baton to use the new version.")
    if not result.ok:
        raise typer.Exit(1)


# --- serve ------------------------------------------------------------------

@app.command()
def serve(
    ctx: typer.Context,
    host: str | None = typer.Option(None, help="Bind address. Defaults to config (127.0.0.1)."),
    port: int | None = typer.Option(None, help="Port. Defaults to config (8787)."),
    expose: bool = typer.Option(False, "--expose", help="Required to bind a non-loopback address."),
    ssl_certfile: Path | None = typer.Option(None, help="TLS certificate (PEM)."),
    ssl_keyfile: Path | None = typer.Option(None, help="TLS private key (PEM)."),
) -> None:
    """Start the local OpenAI-compatible endpoint."""
    import uvicorn

    from ..proxy import create_app, is_loopback, make_token_verifier
    from ..proxy.app import PROXY_TOKEN_ENV

    runtime = build_runtime(ctx, console_logs=True)
    config = runtime.config.proxy
    bind_host = host or config.host
    bind_port = port or config.port

    if not is_loopback(bind_host):
        # Two independent switches (flag + config) so that neither a stray CLI
        # argument nor a copied config file alone can expose a team's keys.
        if not (expose and config.allow_remote):
            fail(f"refusing to listen on {bind_host}: exposing the endpoint to the network requires BOTH "
                 "`--expose` and `proxy.allow_remote: true` in the config.")
        stderr.print(
            "[bold red]WARNING[/] the endpoint is reachable from the network. Anyone holding a Baton token can spend "
            "your keys' quota.\n  - keep tokens secret and revoke unused ones (`baton token revoke`)\n"
            "  - use TLS (--ssl-certfile/--ssl-keyfile or a reverse proxy); without it tokens travel in clear text\n"
            "  - restrict access with a firewall"
        )
        if not ssl_certfile:
            stderr.print("[yellow]No TLS configured: tokens and prompts will be sent unencrypted.[/]")

    has_tokens = bool(runtime.vault and runtime.vault.token_names()) or bool(os.environ.get(PROXY_TOKEN_ENV))
    if not has_tokens:
        fail("no proxy tokens exist, so every request would be rejected. Create one with `baton token new`.")

    application = create_app(runtime, make_token_verifier(runtime.vault), bound_host=bind_host)
    scheme = "https" if ssl_certfile else "http"
    console.print(f"Baton endpoint: [bold]{scheme}://{bind_host}:{bind_port}/v1[/]  "
                  f"({len(runtime.pool.keys)} key(s) pooled)  Ctrl+C to stop")
    server = uvicorn.Server(uvicorn.Config(
        application,
        host=bind_host,
        port=bind_port,
        log_level="warning",
        access_log=False,          # access logs would record client addresses and paths for no benefit
        server_header=False,
        timeout_keep_alive=15,
        limit_concurrency=200,
        ssl_certfile=str(ssl_certfile) if ssl_certfile else None,
        ssl_keyfile=str(ssl_keyfile) if ssl_keyfile else None,
    ))

    async def run() -> None:
        try:
            await server.serve()
        finally:
            await runtime.aclose()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


# --- chat -------------------------------------------------------------------

@app.command()
def chat(
    ctx: typer.Context,
    model: str = typer.Option("auto", "--model", "-m", help="Model name or alias; 'auto' uses each key's default."),
    workspace: Path | None = typer.Option(None, "--workspace", "-w", help="Directory the agent may work in."),
    approval: str | None = typer.Option(None, "--approval", help="ask | auto_edit | auto"),
    prompt: str | None = typer.Option(None, "--prompt", "-p", help="Run one prompt non-interactively and exit."),
) -> None:
    """Start the terminal agent (default command)."""
    run_chat(ctx, model, workspace, approval, prompt)


def run_chat(ctx: typer.Context, model: str, workspace: Path | None, approval: str | None, prompt: str | None) -> None:
    from ..agent import Agent, Toolbox
    from .ui import HELP, TerminalUI

    runtime = build_runtime(ctx)
    agent_config = runtime.config.agent
    mode = approval or agent_config.approval
    if mode not in ("ask", "auto_edit", "auto"):
        fail("--approval must be one of: ask, auto_edit, auto")
    root_dir = (workspace or Path.cwd()).resolve()
    if not root_dir.is_dir():
        fail(f"workspace {root_dir} is not a directory")

    ui = TerminalUI(console, runtime.pool, show_handoffs=agent_config.show_handoffs)
    runtime.router.on_notice = ui.notice          # "model swapped", "provider busy, retrying in 4s..."
    interactive = prompt is None and is_interactive()

    async def approve(name: str, description: str) -> bool:
        if not interactive:
            return False  # nobody is there to ask: deny rather than guess
        answer = await ui.approve(name, description)
        if answer == "always":
            toolbox.remember_approval(name)
            return True
        return bool(answer)

    toolbox = Toolbox(
        root_dir,
        approve=approve,
        approval_mode=mode,
        command_timeout=agent_config.command_timeout,
        max_output_chars=agent_config.max_tool_output_chars,
        protected=(runtime.home,),
    )
    agent = Agent(runtime, toolbox, ui, model=model)
    # One long-lived loop driven turn by turn, so Ctrl+C can cancel a turn
    # without tearing down the HTTP client or the session.
    loop = asyncio.new_event_loop()

    def run_turn(text: str) -> bool:
        ui.start_waiting()
        task = loop.create_task(agent.run_turn(text))
        try:
            loop.run_until_complete(task)
            return True
        except KeyboardInterrupt:
            task.cancel()
            try:
                loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
            except BaseException:  # noqa: BLE001, S110 - the turn is being abandoned either way
                pass
            ui.on_turn_end()
            ui.info("turn cancelled")
        except AllKeysExhaustedError as exc:
            ui.error(str(exc))
            for reason in exc.reasons[:4]:
                ui.info(f"  {reason[:220]}")
            ui.info(f"Your message was not sent. {exhausted_hint(exc.kinds)}")
        except ProviderError as exc:
            ui.error(exc.message)
        except BatonError as exc:
            ui.error(str(exc))
        except Exception:  # noqa: BLE001 - the REPL must survive any single turn
            log.exception("unexpected error during agent turn")
            ui.error("unexpected error; details were written to the log file")
        return False

    def reload_runtime() -> None:
        """Pick up key changes (add/remove/enable/disable) without leaving the chat."""
        nonlocal runtime
        runtime.pool.save(force=True)
        try:
            fresh = build_runtime(ctx)
        except (typer.Exit, SystemExit):
            ui.info("keys were not reloaded; fix the problem above, or restart Baton")
            return
        old, runtime = runtime, fresh
        agent.runtime = fresh
        ui.pool = fresh.pool
        fresh.router.on_notice = ui.notice
        loop.run_until_complete(old.client.aclose())
        ui.info(f"keys reloaded: {len(fresh.pool.keys)} key(s) in the pool")

    def toolbar() -> str:
        key = agent.session.active_key or "no key used yet"
        return f" {key}  ·  model {agent.session.model}  ·  approval {toolbox.approval_mode}  ·  type / for commands"

    ui.set_toolbar(toolbar)

    def check_pinned_models() -> None:
        """Best effort, at most a few seconds: a config that pins a model the provider has retired
        is worked around right away (no failed first message) and reported once."""
        from ..models import audit_pinned_models

        try:
            stale = loop.run_until_complete(asyncio.wait_for(
                audit_pinned_models(runtime.pool, runtime.router.catalog), timeout=8))
        except Exception:  # noqa: BLE001 - never let a model check stop the chat from starting
            return
        for item in stale:
            if item.suggestion:
                runtime.router.catalog.remember_swap(item.key_id, item.model, item.suggestion)
            if not interactive:
                continue
            if item.suggestion:
                ui.info(f"key {item.key_id} pins model '{item.model}', which the provider no longer lists: using "
                        f"'{item.suggestion}' for now. Make it permanent with /models --fix")
            else:
                ui.error(f"key {item.key_id} pins model '{item.model}', which the provider no longer lists, and no "
                         "replacement could be chosen. Edit `models:` in the config, or set it to auto.")

    try:
        check_pinned_models()
        if prompt is not None:
            ok = run_turn(prompt)
            raise typer.Exit(0 if ok else 1)
        if not interactive:
            piped = sys.stdin.read().strip()
            raise typer.Exit(0 if piped and run_turn(piped) else 1)

        ui.banner(root_dir, model, mode)
        if mode == "auto":
            ui.error("approval mode 'auto': the agent may edit files and run shell commands without asking.")
        while True:
            try:
                text = ui.prompt()
            except EOFError:
                break
            except KeyboardInterrupt:
                console.print()
                ui.info("(Ctrl+D or /exit to quit)")
                continue
            if not text:
                continue
            text = normalize_input(text)
            if text.startswith("/"):
                command, _, argument = text.partition(" ")
                argument = argument.strip()
                if command in ("/exit", "/quit"):
                    break
                if command == "/help":
                    console.print(HELP)
                elif command == "/status":
                    ui.show_status()
                elif command in INLINE_COMMANDS:
                    try:
                        parts = shlex.split(argument)
                    except ValueError:
                        ui.error("could not parse that: check your quotes")
                        continue
                    run_inline(ctx, INLINE_COMMANDS[command](parts))
                    if (command == "/keys" and parts and parts[0] in ("add", "remove", "enable", "disable")) \
                            or (command == "/models" and "--fix" in parts):
                        reload_runtime()
                elif command == "/clear":
                    agent.reset()
                    ui.info("conversation cleared")
                elif command == "/model":
                    if not argument and can_use_menu():
                        names = runtime.pool.known_models()
                        argument = choose(f"Pick a model (now: {agent.session.model})",
                                          [(name, "auto (each key's default)" if name == "auto" else name)
                                           for name in names],
                                          default=agent.session.model)
                    if argument:
                        agent.session.model = argument
                    ui.info(f"model: {agent.session.model}")
                elif command == "/approval":
                    if argument in ("ask", "auto_edit", "auto"):
                        toolbox.approval_mode = argument
                    ui.info(f"approval: {toolbox.approval_mode}")
                elif command == "/handoff":
                    try:
                        outcome = loop.run_until_complete(agent.force_handoff())
                        ui.info(f"compacted {outcome.messages_before} -> {outcome.messages_after} messages "
                                f"(summary: {outcome.source})")
                    except (KeyboardInterrupt, BatonError) as exc:
                        ui.error(f"handoff not completed: {type(exc).__name__}")
                else:
                    ui.info("unknown command; /help lists them")
                continue
            run_turn(text)
            ui.footer(agent.session.active_key)
    finally:
        try:
            loop.run_until_complete(runtime.aclose())
            # Finalise any stream generators abandoned by a cancelled or failed
            # turn before the loop goes away (asyncio.run does this for us;
            # a hand-driven loop must do it itself).
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()


if __name__ == "__main__":
    app()
