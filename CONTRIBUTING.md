# Contributing to Baton

Thanks for helping. Baton holds other people's API keys and runs model-chosen actions on their machines,
so stability and security come before features.

## Setup

```bash
git clone https://github.com/Rulebreker/Baton.git
cd Baton
python -m venv .venv            # then activate it
pip install -e ".[dev]"
sh scripts/install-hooks.sh     # Windows: .\scripts\install-hooks.ps1
```

The hook installer points `core.hooksPath` at `.githooks/`, which runs `scripts/secret_scan.py` on every
commit. Please do not skip it with `--no-verify`.

## Before you open a pull request

```bash
pytest
ruff check src tests scripts
python scripts/secret_scan.py --all
```

- Tests must not use the network or sleep for real. Use the `Harness`, `FakeProvider` and `Clock` helpers
  in `tests/conftest.py`, and `httpx.MockTransport` for adapters.
- Changes to rotation, fallback or handoff need a test that fails without the change.
- Read the "Rules for contributors" in the [README](README.md#rules-for-contributors).
- Keep commits small and use conventional prefixes (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).

## Things that are easy to get wrong

- **Error classification** lives in `providers/base.py::classify_http_error`. A new provider quirk
  belongs there, not in a special case in the router.
- **The router never retries a 400.** If you find yourself wanting to, the request is being built wrong.
- **History is only rewritten at a turn boundary.** Anything that edits `session.messages` must check
  `pending_tool_calls` first.
- **Pool methods are synchronous on purpose.** They are atomic within the event loop; adding an `await`
  inside one introduces a race.
- **Adding a provider**: implement `Provider.complete` and `Provider.stream` to translate to and from the
  OpenAI shape, send the key in a header, register it in `providers/__init__.py`, and add adapter tests
  against a mock transport, including its error bodies.

## Fake secrets in tests

Never write a key-shaped literal into a file, even an obviously fake one: scanners (ours and GitHub's)
cannot tell the difference. Build them at runtime:

```python
fake = "sk-" + "aB3" * 14
```
