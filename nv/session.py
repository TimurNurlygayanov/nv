"""Persistent state: chat history between runs and project notes.

State lives OUTSIDE the repo, in <system tmp>/nv-state/<project>-<hash>/,
so it can never be committed by accident. Notes are an append-only
knowledge file (notes.md) with durable facts about the project — env vars,
gotchas, flaky tests. They are injected into every agent's system prompt,
so knowledge accumulates across sessions.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import tempfile
from pathlib import Path


def _migrate_legacy(root: Path, d: Path) -> None:
    """One-time copy from the old in-repo .nv/ folder."""
    legacy = root / ".nv"
    if not legacy.is_dir():
        return
    for name in ("session.json", "notes.md", "checkpoint.json"):
        src, dst = legacy / name, d / name
        if src.is_file() and not dst.exists():
            try:
                dst.write_bytes(src.read_bytes())
            except OSError:
                pass


def state_dir(root: Path) -> Path:
    """Per-project state dir under the system temp folder (outside the repo)."""
    resolved = str(Path(root).resolve())
    key = hashlib.sha1(resolved.lower().encode("utf-8")).hexdigest()[:12]
    d = Path(tempfile.gettempdir()) / "nv-state" / f"{Path(root).name}-{key}"
    d.mkdir(parents=True, exist_ok=True)
    _migrate_legacy(Path(root), d)
    return d


# ---- chat history ----------------------------------------------------

def save_history(root: Path, agent_kind: str, messages: list[dict]) -> None:
    data = {
        "agent": agent_kind,
        "time": datetime.datetime.now().isoformat(timespec="minutes"),
        "messages": [m for m in messages if m.get("role") != "system"],
    }
    try:
        (state_dir(root) / "session.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def load_history(root: Path) -> dict | None:
    try:
        data = json.loads(
            (state_dir(root) / "session.json").read_text(encoding="utf-8"))
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
    path = state_dir(root) / "notes.md"
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"- [{stamp}] {text}\n")


def load_notes(root: Path, limit: int = 3000) -> str:
    try:
        text = (state_dir(root) / "notes.md").read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if len(text) > limit:
        # keep the most recent notes, starting at a line boundary
        tail = text[-limit:]
        nl = tail.find("\n")
        text = "[...older notes trimmed...]\n" + (tail[nl + 1:] if nl >= 0 else tail)
    return text
