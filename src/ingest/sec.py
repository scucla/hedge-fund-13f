"""Shared helpers for talking to sec.gov politely."""

import os
import time

import requests
from dotenv import load_dotenv

# The SEC allows 10 requests/second; stay well under it.
MIN_SECONDS_BETWEEN_REQUESTS = 0.2

_last_request_at = 0.0


def user_agent() -> str:
    load_dotenv()
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if "@" not in ua:
        raise RuntimeError(
            "Set SEC_USER_AGENT to 'Your Name <you@example.com>' in .env; the SEC requires it."
        )
    return ua


def get(url: str, **kwargs) -> requests.Response:
    """GET with the SEC User-Agent and a simple rate limit."""
    global _last_request_at
    wait = MIN_SECONDS_BETWEEN_REQUESTS - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()
    resp = requests.get(url, headers={"User-Agent": user_agent()}, timeout=60, **kwargs)
    resp.raise_for_status()
    return resp


def get_head(url: str, max_bytes: int) -> bytes:
    """The first `max_bytes` of a document. sec.gov ignores Range headers, so
    stream and stop reading; merger proxies run to several MB and the terms are
    on the first pages."""
    global _last_request_at
    wait = MIN_SECONDS_BETWEEN_REQUESTS - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()
    with requests.get(url, headers={"User-Agent": user_agent()}, timeout=60, stream=True) as resp:
        resp.raise_for_status()
        out = bytearray()
        for chunk in resp.iter_content(65536):
            out += chunk
            if len(out) >= max_bytes:
                break
    return bytes(out[:max_bytes])
