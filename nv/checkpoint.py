"""Task checkpoints for /undo.

Before each task nv snapshots the working tree as an unreferenced git
commit object (via write-tree/commit-tree — HEAD, branches and the user's
staging area are left untouched). /undo restores files to that snapshot.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

CHECKPOINT_FILE = ".nv/checkpoint.json"


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          timeout=60)


def is_repo(root: Path) -> bool:
    return _git(root, "rev-parse", "--git-dir").returncode == 0


def create(root: Path) -> str | None:
    """Snapshot the working tree; returns the checkpoint commit hash."""
    if not is_repo(root):
        return None
    _git(root, "add", "-A")
    tree = _git(root, "write-tree").stdout.strip()
    _git(root, "reset", "-q")
    if not tree:
        return None
    args = ["commit-tree", tree, "-m", "nv checkpoint"]
    head = _git(root, "rev-parse", "--verify", "HEAD")
    if head.returncode == 0:
        args += ["-p", head.stdout.strip()]
    commit = _git(root, *args).stdout.strip()
    if not commit:
        return None
    nv_dir = root / ".nv"
    nv_dir.mkdir(exist_ok=True)
    (root / CHECKPOINT_FILE).write_text(
        json.dumps({"commit": commit}), encoding="utf-8")
    return commit


def load_last(root: Path) -> str | None:
    try:
        data = json.loads((root / CHECKPOINT_FILE).read_text(encoding="utf-8"))
        commit = data.get("commit", "")
    except (OSError, ValueError):
        return None
    if commit and _git(root, "cat-file", "-e", commit).returncode == 0:
        return commit
    return None


def diff_stat(root: Path, commit: str) -> str:
    """Summary of what changed since the checkpoint (new files included)."""
    _git(root, "add", "-A", "-N")
    stat = _git(root, "diff", "--stat", commit).stdout.strip()
    _git(root, "reset", "-q")
    return stat


def changed_lines_since(root: Path, commit: str) -> int:
    """Total inserted+deleted lines since the checkpoint."""
    stat = diff_stat(root, commit)
    total = 0
    m = re.search(r"(\d+) insertion", stat)
    if m:
        total += int(m.group(1))
    m = re.search(r"(\d+) deletion", stat)
    if m:
        total += int(m.group(1))
    return total


def restore(root: Path, commit: str) -> str:
    """Revert the working tree to the checkpoint state."""
    _git(root, "add", "-A")
    current = set(_git(root, "ls-files").stdout.splitlines())
    at_checkpoint = set(
        _git(root, "ls-tree", "-r", "--name-only", commit).stdout.splitlines())
    _git(root, "checkout", commit, "--", ".")
    removed = 0
    for name in sorted(current - at_checkpoint):
        try:
            (root / name).unlink()
            removed += 1
        except OSError:
            pass
    _git(root, "reset", "-q")
    parts = [f"restored to checkpoint {commit[:10]}"]
    if removed:
        parts.append(f"removed {removed} file(s) created after it")
    return ", ".join(parts)
