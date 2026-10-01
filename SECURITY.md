# Security policy

## Reporting a vulnerability

Please do **not** open a public issue for a security problem.

Use GitHub's private reporting instead: on the repository page open **Security → Advisories → Report a
vulnerability**. Include what you found, how to reproduce it, and what an attacker could gain. If you
believe a real credential was exposed, say so, but do not include the credential itself.

You should get an acknowledgement within a few days. Fixes for confirmed issues are released as soon as
they are ready, with credit to the reporter unless you prefer otherwise.

## Scope

In scope: anything that lets someone obtain a pooled API key or proxy token, use the proxy without a
valid token, make the agent read or write outside its workspace or read protected files, get a secret
into a log, error message, handoff summary or checkpoint, or commit a secret past the pre-commit hook.

Out of scope: the limits documented under "What Baton does not protect against" in the
[README](README.md#security-model), and vulnerabilities in upstream LLM providers.

## If you leak a key

1. Revoke the key at the provider immediately. Removing it from git history is not enough: assume it was
   copied the moment it was pushed.
2. Remove it from Baton with `baton keys remove ID` and add the replacement with `baton keys add`.
3. If a proxy token leaked, run `baton token revoke NAME` and restart `baton serve`.
