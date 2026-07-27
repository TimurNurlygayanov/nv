"""Persistent state under .nv/: chat history between runs and project notes.

Notes are an append-only knowledge file (.nv/notes.md) with durable facts
about the project — env vars, gotchas, flaky tests. They are injected into
every agent's system prompt, so knowledge accumulates across sessions.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

SESSION_FILE = ".nv/session.json"
NOTES_FILE = ".nv/notes.md"


def _nv_dir(root: Path) -> Path:
    d = root / ".nv"
    d.mkdir(exist_ok=True)
    return d


# ---- chat history ----------------------------------------------------

def save_history(root: Path, agent_kind: str, messages: list[dict]) -> None:
    data = {
        "agent": agent_kind,
        "time": datetime.datetime.now().isoformat(timespec="minutes"),
        "messages": [m for m in messages if m.get("role") != "system"],
    }
    _nv_dir(root)
    try:
        (root / SESSION_FILE).write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def load_history(root: Path) -> dict | None:
    try:
        data = json.loads((root / SESSION_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        return data
    return None


# ---- project notes ---------------------------------------------------

def add_note(root: Path, text: str) -> None:
    text = " ".join(text.split())
    if not text:
        return
    stamp = datetime.date.today().isoformat()
    path = _nv_dir(root) / "notes.md"
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"- [{stamp}] {text}\n")


def load_notes(root: Path, limit: int = 3000) -> str:
    try:
        text = (root / NOTES_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if len(text) > limit:
        # keep the most recent notes, starting at a line boundary
        tail = text[-limit:]
        nl = tail.find("\n")
        text = "[...older notes trimmed...]\n" + (tail[nl + 1:] if nl >= 0 else tail)
    return text
