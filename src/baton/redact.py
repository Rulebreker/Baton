"""Scrubbing of secrets and PII.

Every string that leaves the process boundary in a form a human or another
service may keep (log lines, error messages, handoff summaries, checkpoints,
tool output fed back to a model) goes through `Redactor.scrub`.

Two layers:

1. Exact matches for every secret Baton has loaded. This is the layer that
   matters most: it catches keys with unusual formats (self-hosted gateways,
   new providers) that no regex would recognise.
2. Pattern matches for well-known credential formats and, optionally, PII.
"""

from __future__ import annotations

import re
import threading
from typing import Any

REDACTED = "[REDACTED]"

_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private-key", re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----.*?-----END (?:[A-Z ]+ )?PRIVATE KEY-----", re.S)),
    ("anthropic-key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}")),
    ("openrouter-key", re.compile(r"sk-or-(?:v1-)?[A-Za-z0-9]{24,}")),
    ("openai-key", re.compile(r"sk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{20,}")),
    ("google-key", re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
    ("google-key", re.compile(r"\bAQ\.[A-Za-z0-9_\-]{30,}")),   # newer Google API key format
    ("groq-key", re.compile(r"gsk_[A-Za-z0-9]{24,}")),
    ("hf-token", re.compile(r"hf_[A-Za-z0-9]{24,}")),
    ("xai-key", re.compile(r"xai-[A-Za-z0-9]{24,}")),
    ("baton-token", re.compile(r"baton-[A-Za-z0-9_\-]{24,}")),
    ("github-token", re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}")),
    ("aws-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("slack-token", re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}")),
]

# Patterns that keep a prefix so the scrubbed line stays readable.
_KEEP_PREFIX_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("bearer", re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/\-]{8,}=*")),
    ("auth-header", re.compile(r"(?i)\b((?:x-api-key|x-goog-api-key|api-key|authorization)\s*[:=]\s*[\"']?)(?!bearer\b)[^\s\"',;]{8,}")),
    ("url-credential", re.compile(r"(?i)([?&](?:key|api_key|apikey|token|access_token)=)[^&\s\"']{6,}")),
    ("url-userinfo", re.compile(r"(?i)(\b[a-z][a-z0-9+.\-]*://[^/\s:@]+:)[^@\s/]+(?=@)")),
    ("assignment", re.compile(
        r"(?i)(\b[\w\-.]*(?:api[_\-]?key|secret|passw(?:or)?d|passphrase|access[_\-]?token|auth[_\-]?token)[\w\-.]*"
        r"\s*[:=]\s*[\"']?)[^\s\"',;]{8,}"
    )),
]

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ \-]?){13,19}\b")
# International (+CC ...) or clearly formatted numbers only: bare digit runs
# are far more often IDs, timestamps or token counts than phone numbers.
_PHONE = re.compile(r"(?<![\w.])(?:\+\d{1,3}[ \-.]?)(?:\(?\d{2,4}\)?[ \-.]?){2,4}\d{2,4}(?![\w])")


def _luhn_ok(digits: str) -> bool:
    total, alternate = 0, False
    for ch in reversed(digits):
        value = ord(ch) - 48
        if alternate:
            value *= 2
            if value > 9:
                value -= 9
        total += value
        alternate = not alternate
    return total % 10 == 0


def _card_sub(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    if 13 <= len(digits) <= 19 and _luhn_ok(digits) and len(set(digits)) > 1:
        return "[REDACTED:card]"
    return match.group(0)


class Redactor:
    """Thread-safe scrubber. One shared instance is used process-wide."""

    # Shorter values would cause absurd collateral redaction ("a" -> everywhere).
    MIN_SECRET_LENGTH = 8

    def __init__(self, *, pii: bool = True) -> None:
        self.pii = pii
        self._lock = threading.Lock()
        self._secrets: dict[str, str] = {}
        self._exact: re.Pattern[str] | None = None

    def register_secret(self, value: str | None, label: str = "secret") -> None:
        """Remember a live secret so it is scrubbed wherever it shows up."""
        if not value or len(value) < self.MIN_SECRET_LENGTH:
            return
        with self._lock:
            self._secrets[value] = label
            # Longest first so a secret that contains another is replaced whole.
            ordered = sorted(self._secrets, key=len, reverse=True)
            self._exact = re.compile("|".join(re.escape(item) for item in ordered))

    def scrub(self, text: Any, *, pii: bool | None = None) -> str:
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        if not text:
            return text
        exact, secrets = self._exact, self._secrets
        if exact is not None:
            text = exact.sub(lambda m: f"[REDACTED:{secrets.get(m.group(0), 'secret')}]", text)
        for label, pattern in _SECRET_PATTERNS:
            text = pattern.sub(f"[REDACTED:{label}]", text)
        for _label, pattern in _KEEP_PREFIX_PATTERNS:
            text = pattern.sub(lambda m: m.group(1) + REDACTED, text)
        if self.pii if pii is None else pii:
            text = _EMAIL.sub("[REDACTED:email]", text)
            text = _SSN.sub("[REDACTED:ssn]", text)
            text = _CARD.sub(_card_sub, text)
            text = _PHONE.sub("[REDACTED:phone]", text)
        return text

    def scrub_obj(self, obj: Any, *, pii: bool | None = None) -> Any:
        """Recursively scrub strings inside JSON-like structures (returns a copy)."""
        if isinstance(obj, str):
            return self.scrub(obj, pii=pii)
        if isinstance(obj, dict):
            return {key: self.scrub_obj(value, pii=pii) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self.scrub_obj(item, pii=pii) for item in obj]
        return obj


_default = Redactor()


def get_redactor() -> Redactor:
    return _default


def scrub(text: Any, *, pii: bool | None = None) -> str:
    return _default.scrub(text, pii=pii)


def mask_secret(value: str) -> str:
    """Display form for a key in the UI: enough to recognise, useless to steal."""
    if len(value) <= 10:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"
