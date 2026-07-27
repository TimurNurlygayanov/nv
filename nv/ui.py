"""Console UI helpers: colors, streaming output, confirmations, diff rendering.

All interactive prompts go through a global lock so parallel agents
never fight over stdin.
"""
from __future__ import annotations

import difflib
import os
import sys
import threading

# Enable ANSI escape codes on Windows terminals.
if os.name == "nt":
    os.system("")

# Windows consoles often default to a legacy codepage (cp1251/cp866) that
# cannot print box-drawing chars or non-ASCII model output.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
BLUE = "\x1b[34m"
MAGENTA = "\x1b[35m"
CYAN = "\x1b[36m"

_io_lock = threading.RLock()

# semantic color roles — the theme module rewrites these at runtime
DEFAULT_CODES = {
    "prompt": BOLD + GREEN,
    "banner": BOLD + CYAN,
    "info": DIM,
    "warn": YELLOW,
    "error": RED,
    "tool": MAGENTA,
    "diff_add": GREEN,
    "diff_del": RED,
}
CODES = dict(DEFAULT_CODES)


def c(role: str) -> str:
    return CODES.get(role, "")


def out(text: str = "", color: str = "", end: str = "\n") -> None:
    with _io_lock:
        sys.stdout.write(f"{color}{text}{RESET if color else ''}{end}")
        sys.stdout.flush()


def stream_token(token: str) -> None:
    sys.stdout.write(token)
    sys.stdout.flush()


def banner(text: str) -> None:
    out(f"\n{c('banner')}── {text} {'─' * max(0, 60 - len(text))}{RESET}")


def info(text: str) -> None:
    out(text, c("info"))


def warn(text: str) -> None:
    out(text, c("warn"))


def error(text: str) -> None:
    out(text, c("error"))


def render_diff(old: str, new: str, path: str, max_lines: int = 120) -> None:
    lines = list(difflib.unified_diff(
        old.splitlines(), new.splitlines(),
        fromfile=f"a/{path}", tofile=f"b/{path}", lineterm=""))
    shown = lines[:max_lines]
    with _io_lock:
        for line in shown:
            if line.startswith("+") and not line.startswith("+++"):
                out(line, c("diff_add"))
            elif line.startswith("-") and not line.startswith("---"):
                out(line, c("diff_del"))
            elif line.startswith("@@"):
                out(line, CYAN)
            else:
                out(line)
        if len(lines) > max_lines:
            out(f"... {len(lines) - max_lines} more diff lines hidden ...", c("info"))


def colorize_git_diff(diff_text: str) -> None:
    with _io_lock:
        for line in diff_text.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                out(line, c("diff_add"))
            elif line.startswith("-") and not line.startswith("---"):
                out(line, c("diff_del"))
            elif line.startswith(("@@", "diff --git")):
                out(line, CYAN)
            else:
                out(line)


class Confirmer:
    """Serialized y/n confirmations. 'a' approves everything for the rest
    of the current task (resettable)."""

    def __init__(self) -> None:
        self.approve_all = False
        self._lock = threading.RLock()

    def reset(self) -> None:
        self.approve_all = False

    def ask(self, prompt: str, agent_name: str = "",
            force: bool = False) -> tuple[bool, str]:
        """Returns (approved, user_feedback). force=True ignores yes-to-all —
        used for changes that must always be looked at (e.g. large diffs)."""
        with self._lock:
            if self.approve_all and not force:
                return True, ""
            tag = f"[{agent_name}] " if agent_name else ""
            with _io_lock:
                out(f"\n{BOLD}{YELLOW}{tag}{prompt}{RESET}")
                out(f"{DIM}  y = yes | n = no | a = yes to all (this task) | "
                    f"or type feedback for the agent{RESET}")
                try:
                    answer = input(f"{BOLD}  approve? {RESET}").strip()
                except EOFError:
                    return False, "user aborted"
            low = answer.lower()
            if low in ("y", "yes", "д", "да"):
                return True, ""
            if low == "a":
                self.approve_all = True
                return True, ""
            if low in ("n", "no", "н", "нет", ""):
                return False, ""
            return False, answer  # anything else = rejection with feedback


CONFIRM = Confirmer()
