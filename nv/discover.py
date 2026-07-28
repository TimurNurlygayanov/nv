"""Auto-discovery of Ollama servers — explicit candidates only, no scanning.

Checks, in order: the OLLAMA_HOST environment variable, previously
configured hosts, and localhost (127.0.0.1). Each candidate is verified
with /api/tags. nv never port-scans the network.
"""
from __future__ import annotations

import json
import os
import urllib.request

OLLAMA_PORT = 11434
PROBE_TIMEOUT = 3

# local/LAN probes must bypass any corporate HTTP(S)_PROXY, or every
# candidate "fails" and discovery reports no server found
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def probe(host_url: str) -> list[str] | None:
    """Returns model names if host_url is a live Ollama server, else None."""
    try:
        with _OPENER.open(host_url.rstrip("/") + "/api/tags",
                          timeout=PROBE_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m.get("name", "") for m in data.get("models", [])]
    except (OSError, ValueError):
        return None


def _normalize(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url:
        return ""
    if not url.startswith("http"):
        url = f"http://{url}"
    if "//" not in url:  # malformed like "http:11434" — not a usable candidate
        return ""
    host_part = url.split("//", 1)[1]
    if not host_part:
        return ""
    if ":" not in host_part:
        url += f":{OLLAMA_PORT}"
    return url


def scan(extra_hosts: list[str] | None = None,
         on_status=None) -> list[tuple[str, list[str]]]:
    """Check known candidate hosts. Returns [(host_url, [model, ...]), ...]."""
    found: list[tuple[str, list[str]]] = []
    seen: set[str] = set()

    def note(msg: str) -> None:
        if on_status:
            on_status(msg)

    candidates: list[str] = []
    env_host = os.environ.get("OLLAMA_HOST", "")
    if env_host:
        candidates.append(env_host)
    candidates.extend(extra_hosts or [])
    candidates.extend([f"http://127.0.0.1:{OLLAMA_PORT}",
                       f"http://localhost:{OLLAMA_PORT}"])

    for raw in candidates:
        url = _normalize(raw)
        if not url or url in seen:
            continue
        seen.add(url)
        note(f"checking {url}...")
        models = probe(url)
        if models is not None:
            found.append((url, models))

    return found
