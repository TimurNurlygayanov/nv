"""Rolling generation-speed stats per host+model, used for job estimates.

Every Ollama response includes token counts and durations; we keep an
exponential moving average in ~/.nv-perf.json, so estimates calibrate
themselves after the first real calls.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

PERF_FILE = Path.home() / ".nv-perf.json"
_ALPHA = 0.3  # EMA weight of the newest measurement
_LOCK = threading.Lock()  # team mode records from parallel threads


def _load() -> dict:
    try:
        data = json.loads(PERF_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def record(host: str, model: str, chunk: dict) -> None:
    """Update speed stats from a final Ollama stream chunk."""
    ec, ed = chunk.get("eval_count"), chunk.get("eval_duration")
    pc, pd = chunk.get("prompt_eval_count"), chunk.get("prompt_eval_duration")
    if not ec or not ed:
        return
    gen_tps = ec / (ed / 1e9)
    prompt_tps = pc / (pd / 1e9) if pc and pd else None

    def ema(old, new):
        return new if not old else old * (1 - _ALPHA) + new * _ALPHA

    with _LOCK:
        data = _load()
        key = f"{host}|{model}"
        cur = data.get(key, {})
        if not isinstance(cur, dict):
            cur = {}
        cur["gen_tps"] = ema(cur.get("gen_tps"), gen_tps)
        if prompt_tps:
            cur["prompt_tps"] = ema(cur.get("prompt_tps"), prompt_tps)
        cur["n"] = cur.get("n", 0) + 1
        data[key] = cur
        try:  # atomic replace: a crash must not truncate the stats file
            fd, tmp = tempfile.mkstemp(dir=str(PERF_FILE.parent),
                                       prefix=".nv-perf.", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json.dumps(data))
            os.replace(tmp, PERF_FILE)
        except OSError:
            pass


def speeds(host: str, model: str) -> tuple[float, float] | None:
    """(prompt_tokens_per_sec, gen_tokens_per_sec) or None if never measured."""
    cur = _load().get(f"{host}|{model}") or {}
    gen = cur.get("gen_tps")
    if not gen:
        return None
    # prompt processing is usually much faster than generation; if we never
    # measured it, assume 8x as a conservative default
    return cur.get("prompt_tps") or gen * 8, gen


def humanize(seconds: float) -> str:
    if seconds < 90:
        return f"~{max(1, round(seconds))}s"
    if seconds < 5400:
        return f"~{seconds / 60:.1f} min"
    return f"~{seconds / 3600:.1f} h"
