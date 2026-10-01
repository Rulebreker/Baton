# Baton

Baton pools a team's LLM API keys, rotates through them automatically, and hands a task from one key to
the next without losing the thread. It ships two front ends on one engine:

- **A terminal agent** in the style of Claude Code: it reads, edits and runs things in your project.
- **A local OpenAI-compatible endpoint**: point any existing app at `http://127.0.0.1:8787/v1` with one
  Baton token, and Baton rotates the real keys behind it.

Supported providers: Google Gemini, OpenAI, Anthropic Claude, and anything that speaks the OpenAI API
(Groq, OpenRouter, Mistral, Ollama, LM Studio, vLLM, ...).

```
  your app / scripts ──► Baton proxy ─┐                          ┌─► key 1 (Gemini)
                         (one token)  ├─► router ─► key pool ────┼─► key 2 (Gemini)
  you ──► Baton terminal agent ───────┘   │                      ├─► key 3 (OpenAI)
                                          └─ handoff summaries   └─► key N ...
```

> **Check your providers' terms first.** Baton is meant for keys your team is entitled to use, such as
> one key per teammate or several paid projects. Some providers forbid creating extra accounts to get
> around free-tier limits. That is between you and the provider; Baton does not hide what it does.

## Contents

- [Quick start](#quick-start)
- [The terminal agent](#the-terminal-agent)
- [The proxy endpoint](#the-proxy-endpoint)
- [Configuration](#configuration)
- [How rotation, fallback and handoff work](#how-rotation-fallback-and-handoff-work)
- [Security model](#security-model)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Quick start

Requires Python 3.10 or newer and git.

**Windows (PowerShell)**

```powershell
git clone https://github.com/Rulebreker/Baton.git
cd Baton
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
baton init
```

If activation is blocked by the execution policy, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or skip activation and call
`.\.venv\Scripts\baton.exe` directly.

**Linux / macOS**

```bash
git clone https://github.com/Rulebreker/Baton.git
cd Baton
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
baton init
```

`baton init` is a short wizard. You pick how to protect the vault and your provider from a list (arrow keys
and Enter), then paste the API key (input is hidden) and optionally test it with one tiny request. Key
name, models and quota limits are chosen for you; add `--advanced` to set them yourself. Then:

```text
baton                 start the agent in the current directory
baton serve           start the OpenAI-compatible endpoint on 127.0.0.1:8787
baton status          key health and quota usage
baton doctor          check the setup for configuration and security problems
baton keys add        add another key        (also: list, remove, enable, disable, test)
baton token new NAME  create a proxy token   (also: list, revoke)
```

Optional: `pip install -e ".[keyring]"` stores the vault key in the OS keychain (Windows Credential
Manager, macOS Keychain, Secret Service) when you choose keyfile mode.

## The terminal agent

```text
> add input validation to the signup handler and run the tests
* read_file(path="app/handlers/signup.py")
  | 84 line(s)
* edit_file(path="app/handlers/signup.py", old_string=...)
  | edited app/handlers/signup.py
* run_command(command="pytest -q")
  | exit 0
Validation is in place and the 31 tests pass.
~ handed off gemini-alice -> gemini-bob (summary: model)
gemini-bob | gemini-3.8-flash | day 3% | 5/6 keys ready
```

The agent works inside one directory (the current one, or `--workspace PATH`). It can read, search,
write and edit files there and run shell commands. By default it **asks before every edit and every
command**; choose `--approval auto_edit` to stop asking about edits, or `--approval auto` to stop asking
altogether (only in a throwaway environment).

| Command | Effect |
| --- | --- |
| `/status` | key pool table with per-minute and per-day usage |
| `/handoff` | summarise the conversation now and continue from the summary |
| `/clear` | start a new conversation |
| `/model` | pick a model from a list with the arrow keys (or `/model NAME`; `auto` uses each key's default) |
| `/approval MODE` | `ask`, `auto_edit` or `auto` |
| `/exit` | quit (Ctrl+D also works). Ctrl+C cancels the current turn only |

One-shot use for scripts: `baton chat -p "summarise the TODOs in this repo"`. With no terminal attached
nobody can approve anything, so edits and commands are declined unless you pass `--approval auto`.

## The proxy endpoint

```text
baton serve
Baton endpoint: http://127.0.0.1:8787/v1  (6 key(s) pooled)  Ctrl+C to stop
```

Change two settings in an existing project, the base URL and the key:

```python
import os

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key=os.environ["BATON_TOKEN"])
reply = client.chat.completions.create(
    model="auto",                       # or a model/alias from your config
    messages=[{"role": "user", "content": "Hello"}],
)
```

```bash
curl http://127.0.0.1:8787/v1/chat/completions \
  -H "Authorization: Bearer $BATON_TOKEN" -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "Hello"}]}'
```

Most tools read `OPENAI_BASE_URL` and `OPENAI_API_KEY`, so setting those two variables is often enough.

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/chat/completions` | chat, tools/function calling, streaming (`"stream": true`) |
| `GET /v1/models` | models and aliases from your config, plus `auto` |
| `GET /v1/baton/status` | key health and usage (no secrets) |
| `GET /healthz` | liveness; no token needed, reveals nothing |

**Model names.** `auto` uses each key's default model. A model listed under a key is routed to keys that
offer it first. Any other name (for example a hard-coded `gpt-4o-mini` in an app you did not write) is
served by each key's default model, so existing projects work unchanged. Set
`rotation.unknown_model: reject` if you would rather get an error. Response headers `x-baton-key` and
`x-baton-attempts` tell you which key answered.

Not implemented: embeddings, images, audio, the Responses API, and `n > 1`.

## Configuration

Settings live in `~/.baton/config.yaml` (written by the wizard). Baton looks for, in order: `--config`,
`$BATON_CONFIG`, `~/.baton/config.yaml`. See [`baton.example.yaml`](baton.example.yaml) for every option
with comments, and [`.env.example`](.env.example) for the environment variables, which Baton reads from
`~/.baton/.env`.

Baton never reads a config or `.env` from the current directory. The agent is run inside arbitrary
project folders, and a file planted in a cloned repository must not be able to reconfigure it.

The settings that matter most:

| Setting | Default | Meaning |
| --- | --- | --- |
| `keys[].limits.rpd` / `tpd` | none | Daily request/token quota. **Needed for the planned 90% handoff.** |
| `keys[].limits.rpm` / `tpm` | none | Per-minute quota; Baton rotates around a key that is momentarily full. |
| `keys[].priority` | `0` | Lower numbers are used first. |
| `rotation.handoff_threshold` | `0.90` | Daily-quota fraction at which a session plans its handoff. |
| `rotation.hard_limit` | `0.98` | Past this a key takes no new requests. |
| `rotation.max_queue_wait` | `15` | Seconds to wait for a key to recover before failing closed. |
| `handoff.keep_recent_messages` | `6` | Newest messages carried verbatim through a handoff. |
| `proxy.host` | `127.0.0.1` | Bind address. See [network exposure](#network-exposure). |
| `proxy.rate_limit` | 120/min, burst 30 | Per client token. |
| `agent.approval` | `ask` | Whether edits and commands need confirmation. |

Free-tier limits change often, so Baton ships none built in. Enter your provider's current numbers in the
wizard. Without limits Baton still works: it learns from rate-limit headers and reacts to 429 responses,
but it can only hand off *after* a key runs dry instead of just before.

Data directory contents (`~/.baton`, override with `BATON_HOME`):

| File | Contents |
| --- | --- |
| `config.yaml` | settings; no secrets |
| `keys.vault` (+ `.bak`) | AES-256-GCM encrypted API keys and proxy-token hashes |
| `master.key` | vault key, **keyfile mode without a keychain only** |
| `state.json` | per-key usage counters, so daily quotas survive a restart |
| `logs/baton.log` | scrubbed, size-capped, rotated |
| `checkpoints/` | scrubbed full transcripts saved before each handoff |

## How rotation, fallback and handoff work

**Per-key accounting.** For every key Baton tracks requests and tokens in a sliding one-minute window and
in a UTC-day window, tokens reserved by in-flight requests (so concurrent requests cannot jointly
overshoot), what the provider reports in rate-limit headers, and health (cooldowns, a circuit breaker).

**Rotation.** Each request goes to the best usable key: one that offers the requested model, is not
nearly empty, and matches the session's current key if possible. When a key's per-minute window fills,
Baton simply uses another key and comes back when the window clears. That is lossless and needs no
summary.

**Fallback.** Every upstream failure is classified, and the class decides the reaction:

| Failure | Reaction |
| --- | --- |
| 429 rate limit | cool the key down (honouring `Retry-After`), retry on the next key |
| quota exhausted, billing (402) | park the key until it resets, retry on the next key |
| 401 / 403 | park the key (re-probed hourly), retry on the next key |
| 5xx, timeouts, network errors | strike against the circuit breaker, retry on the next key |
| 400 bad request | returned to the caller immediately; other keys would fail the same way |
| prompt too long | try a key with a larger context window, else compact the conversation |
| all keys unavailable | wait if a key recovers within `max_queue_wait`, otherwise **fail closed** with a clear error |

The retried request is the *same* request with the full conversation, so nothing is dropped. Attempts are
bounded (two passes over the pool), so Baton cannot loop forever.

**Summary handoff.** When the active key reaches 90% of its *daily* quota, the session plans its exit at
the next turn boundary:

1. The outgoing key, which still has the reserve between 90% and the hard limit, writes a structured note:
   goal, work done, current state, next steps, key decisions, files and context touched.
2. If that fails, another healthy key writes it. If no model can, Baton builds the note locally from the
   transcript. A handoff therefore always completes.
3. The full transcript is saved to `checkpoints/` (scrubbed), then replaced by the system prompt, the
   note, and the most recent messages verbatim.
4. The session continues on the next key.

Guarantees, each covered by tests: a handoff never happens between a tool call and its result; the note
and the checkpoint are scrubbed of keys and PII; the note re-enters the conversation as delimited *user*
content, never as a system prompt, so text that came from a web page or tool output cannot promote itself
to an instruction.

**Limits worth knowing.**

- A stream can fail over to another key only until the first byte reaches the caller. After that, the
  agent discards the partial reply and restarts it on another key; a proxy client receives an error event
  followed by `[DONE]`.
- The proxy forwards client conversations unchanged by default. Set `proxy.compact_on_handoff: true` to
  let it apply the summary handoff to long histories.
- Claude "thinking" blocks are not replayed. They are bound to the exact conversation that produced them,
  and a handoff rewrites history, so Baton sends a transcript without them. The model re-plans from the
  visible conversation.
- Daily windows reset at 00:00 UTC. A provider that resets at another time is handled by the hourly
  re-probe of parked keys.
- Run one Baton process per data directory. Two processes keep separate usage counters.
- Token counts used for reservations are estimates (about four characters per token). Actual usage
  reported by the provider replaces the estimate after each call.

## Security model

Baton holds credentials that cost money and runs model-chosen actions on your machine, so this section is
written for people who will change the code. If you add a feature, keep these properties intact.

**What Baton protects, and from what**

| Asset | Threat | Defence |
| --- | --- | --- |
| Provider API keys | committed to git | Keys never live in the repo or in YAML. `.gitignore` covers every secret path; the pre-commit hook blocks secret files and key-shaped strings even with `git add -f`; CI rescans the whole tree. |
| | read from disk | Vault encrypted with AES-256-GCM. Key derived from your passphrase with scrypt, or held in the OS keychain. Atomic writes, owner-only permissions on POSIX, one backup generation. |
| | leaked through logs, errors, summaries | Every loaded key is registered with a process-wide redactor. All log records (including third-party libraries' and tracebacks), error messages, handoff summaries, checkpoints and tool output pass through it. |
| | leaked in URLs | Keys are sent in headers only (including Gemini, which is usually shown with `?key=`). Redirects are never followed. Plain-HTTP upstreams are refused unless they are loopback. |
| | exfiltrated by the model | The agent cannot read `.env`, vaults, private keys or Baton's data directory; its shell has credential-like environment variables removed. |
| | redirected by a tampered config | Config and `.env` are read only from Baton's private directory, never from the project being worked on. Each vault key is bound, inside the encrypted vault, to the host it was added for; if the config later points it elsewhere, Baton refuses to use it. |
| Your quota (the proxy) | another machine using it | Binds to `127.0.0.1` by default. Listening elsewhere needs two switches and prints a warning. |
| | another local user or a web page | Every request needs a Baton token (256 bits, stored only as a SHA-256 hash, compared in constant time). `Host` is checked to defeat DNS rebinding; requests carrying a browser `Origin` are refused unless allow-listed; wildcard CORS is rejected by the config loader. |
| | abuse and guessing | Per-token rate limit, failed-auth lockout, body-size cap, strict request validation, an allow-list for pass-through parameters. |
| Your machine (the agent) | a model, or text it read, doing damage | Paths are confined to the workspace (symlinks resolved); writes into `.git` are refused; edits and commands need approval by default; "always allow" is never remembered for shell commands; commands have a timeout and their process tree is killed. |
| Conversation content | prompt injection through a handoff | The summariser is told to treat the transcript as data; the note returns as delimited user content, not as a system prompt. |

**What Baton does not protect against.** Be clear about these before relying on it.

- **Your prompts go to third parties.** Whatever you type, and whatever the agent reads from your files,
  is sent to the provider whose key is active. Baton scrubs known keys from tool output first, but it
  does not scrub PII or proprietary code from your prompts. Do not point the agent at data you may not
  share.
- **Anyone with a proxy token can spend the pool's quota.** Treat tokens like API keys and revoke with
  `baton token revoke`.
- **Keyfile mode is convenience, not strong protection.** If the vault key sits in `master.key`, anyone
  who can read the data directory can decrypt the vault. `baton doctor` warns about this. Use passphrase
  mode, or install the `keyring` extra.
- **Windows file permissions.** POSIX gets `0700`/`0600`. On Windows Baton relies on your user profile's
  ACLs; do not put `BATON_HOME` on a shared drive.
- **Memory.** Decrypted keys live in process memory while Baton runs. Python cannot reliably wipe
  strings, so this is "in memory only", not "zeroed after use".
- **Redaction is best effort for unknown formats.** Keys Baton has loaded are always caught by exact
  match. Other people's secrets are caught only if they match a known pattern.
- **`--approval auto` removes the human check.** A prompt-injected instruction could then run commands.
  Use it only in a disposable environment.
- **Model-run commands are not sandboxed.** An approved command runs with your user's rights. The
  workspace confinement applies to Baton's file tools, not to what a shell command does.

### Network exposure

`baton serve` refuses any non-loopback address unless **both** `proxy.allow_remote: true` is set in the
config and `--expose` is passed, so neither a copied config nor a stray flag exposes the pool by
accident. If you do expose it, put TLS in front (`--ssl-certfile`/`--ssl-keyfile` or a reverse proxy) and
firewall the port; without TLS, tokens and prompts cross the network in clear text.

### Rules for contributors

1. Never log, print or raise a string that may contain a key without passing it through
   `baton.redact.scrub`. Do not put secrets or full URLs with credentials into exception messages.
2. Never add a config field that can hold a secret. Config models forbid unknown fields on purpose.
3. Never accept a secret as a command-line argument; use a hidden prompt or stdin.
4. New agent tools must resolve paths through `Toolbox.resolve` and return errors as tool results.
5. New proxy endpoints must be wrapped in `guard(...)`.
6. Test fixtures build key-shaped strings at runtime; never paste a real-looking key into a file.
7. Run `scripts/install-hooks` once after cloning.

Report vulnerabilities privately: see [SECURITY.md](SECURITY.md).

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `All N API key(s) are exhausted, rate-limited or unhealthy` | The pool really is spent. `baton status` shows each key's state and when it retries. Add keys or wait. |
| A key shows `disabled` | The provider rejected it (401/403). Check the key, then `baton keys enable ID`. It is re-probed hourly anyway. |
| No planned handoff ever happens | The key has no `rpd`/`tpd` limit configured. `baton doctor` lists such keys. |
| `401 missing or invalid Baton token` | The client is sending its old provider key. Use the token from `baton token new`. |
| `403 invalid Host header` | The endpoint was reached through a name other than `localhost`/`127.0.0.1`. |
| `403 origin not allowed` | A browser app is calling the proxy. Add its origin to `proxy.cors_origins`. |
| Vault passphrase prompt in a service | Set `BATON_PASSPHRASE` in the service's environment. |
| Something odd | Look at `~/.baton/logs/baton.log`; it is scrubbed and safe to share after a quick read. |

## Contributing

```bash
pip install -e ".[dev]"
sh scripts/install-hooks.sh        # Windows: .\scripts\install-hooks.ps1
pytest                             # no network, no real sleeps
ruff check src tests scripts
python scripts/secret_scan.py --all
```

Layout:

```text
src/baton/
  config.py        validated settings (no secrets)
  vault.py         encrypted key store and proxy-token hashes
  redact.py        secret and PII scrubbing
  providers/       OpenAI, OpenAI-compatible, Anthropic and Gemini adapters
  pool.py          per-key quota accounting, health, candidate selection
  router.py        request routing and fallback
  handoff.py       summary handoff and its fallback chain
  session.py       planned handoff, compaction and stream restarts for a conversation
  proxy/           OpenAI-compatible HTTP endpoint and its abuse controls
  agent/           workspace-confined tools and the turn loop
  cli/             commands, setup wizard, terminal UI
tests/             rotation, fallback, handoff, security, proxy, agent, CLI
scripts/           secret scanner and hook installers
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Baton is released under the [MIT License](LICENSE).
