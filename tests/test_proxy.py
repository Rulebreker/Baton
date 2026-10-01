"""The local OpenAI-compatible endpoint: compatibility, auth, limits, hardening."""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from baton.errors import ErrorKind
from baton.providers.base import StreamChunk, Usage
from baton.proxy import create_app, is_loopback, make_token_verifier
from baton.proxy.guard import AuthLockout, TokenBucket
from baton.redact import get_redactor
from baton.runtime import Runtime
from baton.vault import Vault
from conftest import Harness, err, make_config, secret_for

BASE = "http://127.0.0.1:8787"
BODY = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]}


class Proxy:
    def __init__(self, tmp_path, count: int = 3, **sections) -> None:
        self.h = Harness(make_config(count, **sections))
        h = self.h
        # Production registers every loaded key with the process-wide redactor
        # (runtime.collect_secrets); mirror that for the fake keys.
        for index in range(count):
            get_redactor().register_secret(secret_for(index), f"key:k{index}")
        self.runtime = Runtime(h.config, tmp_path, h.pool, h.router, h.handoff, h.relay, client=None)  # type: ignore[arg-type]
        self.vault = Vault.create(tmp_path / "keys.vault", passphrase="correct horse battery", scrypt_n=2**10)
        self.token = self.vault.create_token("web-app")
        self.ticks = 0.0
        app = create_app(self.runtime, make_token_verifier(self.vault), clock=lambda: self.ticks)
        self.client = TestClient(app, base_url=BASE)
        self.auth = {"Authorization": f"Bearer {self.token}"}

    def post(self, body=None, **kwargs):
        headers = kwargs.pop("headers", self.auth)
        return self.client.post("/v1/chat/completions", json=BODY if body is None else body, headers=headers, **kwargs)


@pytest.fixture
def proxy(tmp_path):
    def build(count: int = 3, **sections) -> Proxy:
        return Proxy(tmp_path, count, **sections)

    return build


# --- compatibility ----------------------------------------------------------

def test_chat_completion_has_openai_shape(proxy):
    p = proxy()
    response = p.post()
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion" and data["id"].startswith("chatcmpl-")
    assert data["choices"][0]["message"] == {"role": "assistant", "content": "ok"}
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert response.headers["x-baton-key"] == "k0"
    assert response.headers["cache-control"] == "no-store"


def test_unknown_client_model_is_mapped_to_each_keys_default(proxy):
    p = proxy()
    p.post()
    assert p.h.provider.calls[0][1].model == "fake-model"      # not the client's "gpt-4o-mini"


def test_rotation_is_invisible_to_the_client(proxy):
    p = proxy()
    p.h.provider.script(0, err(ErrorKind.RATE_LIMIT, retry_after=60))
    response = p.post()
    assert response.status_code == 200 and response.headers["x-baton-key"] == "k1"
    assert response.headers["x-baton-attempts"] == "2"
    assert p.post().headers["x-baton-key"] == "k1"             # client stays on the key that worked


def test_streaming_emits_openai_sse_frames(proxy):
    p = proxy()
    call = {"index": 0, "id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    p.h.provider.stream_scripts[secret_for(0)] = [[
        StreamChunk(text="Hel"), StreamChunk(text="lo"), StreamChunk(tool_calls=[call]),
        StreamChunk(finish_reason="tool_calls", usage=Usage(7, 3)),
    ]]
    body = {**BODY, "stream": True, "stream_options": {"include_usage": True}}
    with p.client.stream("POST", "/v1/chat/completions", json=body, headers=p.auth) as response:
        assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in response.iter_lines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    frames = [json.loads(line[6:]) for line in lines[:-1]]
    assert frames[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert "".join(f["choices"][0]["delta"].get("content", "") for f in frames if f["choices"]) == "Hello"
    assert any(f["choices"] and f["choices"][0]["delta"].get("tool_calls") for f in frames)
    assert [f["choices"][0]["finish_reason"] for f in frames if f["choices"]][-1] == "tool_calls"
    assert frames[-1]["usage"]["total_tokens"] == 10


def test_stream_fails_over_before_first_byte(proxy):
    p = proxy()
    p.h.provider.stream_scripts[secret_for(0)] = [[err(ErrorKind.TRANSIENT)]]
    with p.client.stream("POST", "/v1/chat/completions", json={**BODY, "stream": True}, headers=p.auth) as response:
        assert response.status_code == 200 and response.headers["x-baton-key"] == "k1"
        assert "ok" in "".join(response.iter_lines())


def test_mid_stream_failure_is_reported_in_band_and_terminated_cleanly(proxy):
    p = proxy(1)
    leaked = secret_for(0)
    p.h.provider.stream_scripts[secret_for(0)] = [[
        StreamChunk(text="partial"),
        err(ErrorKind.TRANSIENT).__class__(ErrorKind.TRANSIENT, f"upstream died, key was {leaked}"),
    ]]
    with p.client.stream("POST", "/v1/chat/completions", json={**BODY, "stream": True}, headers=p.auth) as response:
        text = "\n".join(response.iter_lines())
    assert '"code": "stream_interrupted"' in text and text.rstrip().endswith("data: [DONE]")
    assert leaked not in text


def test_models_and_status_endpoints(proxy):
    p = proxy(2)
    models = p.client.get("/v1/models", headers=p.auth).json()
    assert {"auto", "fake-model"} <= {m["id"] for m in models["data"]}
    status = p.client.get("/v1/baton/status", headers=p.auth)
    assert [row["id"] for row in status.json()["keys"]] == ["k0", "k1"]
    assert secret_for(0) not in status.text


# --- failure reporting ------------------------------------------------------

def test_all_keys_exhausted_returns_503_with_retry_after(proxy):
    p = proxy(2)
    for index in range(2):
        p.h.provider.always(index, err(ErrorKind.QUOTA_EXHAUSTED, retry_after=900))
    response = p.post()
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "all_keys_exhausted"
    assert int(response.headers["retry-after"]) > 800
    stream = p.post({**BODY, "stream": True})
    assert stream.status_code == 503                           # proper status, not a broken 200 stream


def test_upstream_bad_request_is_passed_through_scrubbed(proxy):
    p = proxy(1)
    p.h.provider.script(0, err(ErrorKind.BAD_REQUEST, status=400).__class__(
        ErrorKind.BAD_REQUEST, f"bad tool schema (auth was Bearer {secret_for(0)})", status=400))
    response = p.post()
    assert response.status_code == 400 and response.json()["error"]["type"] == "invalid_request_error"
    assert secret_for(0) not in response.text


@pytest.mark.parametrize("body", [
    [], {}, {"messages": []}, {"messages": "hi"}, {"messages": [{"role": "wizard", "content": "x"}]},
    {"messages": [{"role": "user", "content": 5}]}, {**BODY, "temperature": 9}, {**BODY, "n": 3},
    {**BODY, "stream": "yes"}, {**BODY, "max_tokens": -1}, {**BODY, "model": "m" * 500},
    {"messages": [{"role": "tool", "content": "x"}]}, {**BODY, "tools": "nope"},
])
def test_malformed_requests_are_rejected_before_any_key_is_used(proxy, body):
    p = proxy()
    response = p.post(body)
    assert response.status_code == 400 and response.json()["error"]["type"] == "invalid_request_error"
    assert not p.h.provider.calls


def test_invalid_json_and_oversized_bodies(proxy):
    p = proxy(proxy={"max_body_bytes": 2048})
    bad = p.client.post("/v1/chat/completions", content=b"{not json", headers=p.auth)
    assert bad.status_code == 400
    huge = {**BODY, "messages": [{"role": "user", "content": "x" * 5000}]}
    assert p.post(huge).status_code == 413
    assert not p.h.provider.calls


def test_unknown_routes_and_methods_answer_json(proxy):
    p = proxy()
    assert p.client.get("/v1/embeddings", headers=p.auth).status_code == 404
    assert p.client.get("/v1/chat/completions", headers=p.auth).status_code == 405
    assert p.client.get("/nope").json()["error"]["code"] == "not_found"


# --- authentication and abuse controls --------------------------------------

@pytest.mark.parametrize("headers", [
    {}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic abc"}, {"x-api-key": "nope"},
    {"Authorization": "Bearer " + "x" * 5000},
])
def test_requests_without_a_valid_token_are_rejected(proxy, headers):
    p = proxy()
    response = p.post(headers=headers)
    assert response.status_code == 401 and response.headers["www-authenticate"] == "Bearer"
    assert not p.h.provider.calls


def test_x_api_key_header_and_revocation(proxy):
    p = proxy()
    assert p.post(headers={"x-api-key": p.token}).status_code == 200
    p.vault.revoke_token("web-app")
    assert p.post().status_code == 401


def test_env_token_is_accepted_only_when_long_enough(tmp_path, monkeypatch):
    long_token = "ci-" + "t0k3n" * 6
    monkeypatch.setenv("BATON_PROXY_TOKEN", long_token)
    assert make_token_verifier(None)(long_token) == "env"
    assert make_token_verifier(None)("something-else") is None
    monkeypatch.setenv("BATON_PROXY_TOKEN", "short")
    assert make_token_verifier(None)("short") is None


def test_healthz_needs_no_token_and_reveals_nothing(proxy):
    p = proxy()
    response = p.client.get("/healthz")
    assert response.status_code == 200 and response.json() == {"status": "ok"}


def test_per_client_rate_limit(proxy):
    p = proxy(proxy={"rate_limit": {"requests_per_minute": 60, "burst": 3}})
    assert [p.post().status_code for _ in range(4)] == [200, 200, 200, 429]
    limited = p.post()
    assert limited.json()["error"]["code"] == "rate_limit_exceeded" and "retry-after" in limited.headers
    p.ticks += 2.0                                             # two seconds refill two tokens
    assert [p.post().status_code for _ in range(3)] == [200, 200, 429]


def test_repeated_bad_tokens_are_locked_out_but_valid_clients_are_not(proxy):
    p = proxy(proxy={"auth_fail_limit": 3, "auth_lockout_seconds": 60})
    bad = {"Authorization": "Bearer wrong"}
    assert [p.post(headers=bad).status_code for _ in range(4)] == [401, 401, 401, 429]
    assert p.post().status_code == 200                         # same address, valid token: unaffected
    p.ticks += 61
    assert p.post(headers=bad).status_code == 401


def test_dns_rebinding_host_header_is_rejected(proxy):
    p = proxy()
    for host in ("evil.example", "evil.example:8787", "192.168.1.5:8787"):
        response = p.post(headers={**p.auth, "Host": host})
        assert response.status_code == 403 and response.json()["error"]["code"] == "forbidden_host"
    for host in ("localhost:8787", "127.0.0.1", "[::1]:8787"):
        assert p.post(headers={**p.auth, "Host": host}).status_code == 200


def test_cross_site_browser_requests_are_rejected_unless_origin_is_allowed(proxy):
    p = proxy(proxy={"cors_origins": ["http://localhost:3000"]})
    evil = p.post(headers={**p.auth, "Origin": "https://evil.example"})
    assert evil.status_code == 403 and "access-control-allow-origin" not in evil.headers
    good = p.post(headers={**p.auth, "Origin": "http://localhost:3000"})
    assert good.status_code == 200 and good.headers["access-control-allow-origin"] == "http://localhost:3000"
    preflight = p.client.options("/v1/chat/completions", headers={"Origin": "http://localhost:3000"})
    assert preflight.status_code == 204
    assert p.client.options("/v1/chat/completions", headers={"Origin": "https://evil.example"}).status_code == 403


def test_extra_parameters_are_allow_listed(proxy):
    p = proxy()
    p.post({**BODY, "seed": 7, "user": "u1", "base_url": "https://evil.example", "api_key": "x"})
    extra = p.h.provider.calls[0][1].extra
    assert extra == {"seed": 7, "user": "u1"}


# --- helpers ----------------------------------------------------------------

def test_is_loopback():
    assert all(is_loopback(h) for h in ("127.0.0.1", "localhost", "::1", "[::1]", "127.8.9.1"))
    assert not any(is_loopback(h) for h in ("0.0.0.0", "192.168.1.2", "example.com", "::"))  # noqa: S104


def test_token_bucket_and_lockout_are_bounded():
    now = [0.0]
    bucket = TokenBucket(60, 1, clock=lambda: now[0])
    for index in range(12_000):
        bucket.allow(f"client-{index}")
    assert len(bucket._buckets) <= 10_000
    lockout = AuthLockout(2, 10, clock=lambda: now[0])
    for index in range(12_000):
        lockout.record_failure(f"10.0.{index // 250}.{index % 250}")
    assert len(lockout._failures) <= 10_000
