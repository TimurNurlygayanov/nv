"""Console UI helpers: colors, streaming output, confirmations, diff rendering.

All interactive prompts go through a global lock so parallel agents
never fight over stdin.
"""
from __future__ import annotations

import difflib
import os
import re
import sys
import threading

# Line editing in input(): without readline, arrow keys on unix terminals
# print escape codes (^[[D) instead of moving the cursor. Also enables
# up-arrow history. Windows consoles do this natively; readline is absent
# there and the import just fails quietly.
try:
    import readline  # noqa: F401
    _HAS_READLINE = True
except ImportError:
    _HAS_READLINE = False

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def prompt_str(text: str) -> str:
    """Prepare a colored prompt for input(): readline must be told the ANSI
    codes are zero-width (\\x01..\\x02), or cursor movement miscounts."""
    if not _HAS_READLINE:
        return text
    return _ANSI.sub(lambda m: "\x01" + m.group(0) + "\x02", text)

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

# no escape codes when piped/redirected or when the user sets NO_COLOR
COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

RESET = "\x1b[0m" if COLOR else ""
BOLD = "\x1b[1m" if COLOR else ""
DIM = "\x1b[2m" if COLOR else ""
RED = "\x1b[31m" if COLOR else ""
GREEN = "\x1b[32m" if COLOR else ""
YELLOW = "\x1b[33m" if COLOR else ""
BLUE = "\x1b[34m" if COLOR else ""
MAGENTA = "\x1b[35m" if COLOR else ""
CYAN = "\x1b[36m" if COLOR else ""

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
    return CODES.get(role, "") if COLOR else ""


def out(text: str = "", color: str = "", end: str = "\n") -> None:
    with _io_lock:
        sys.stdout.write(f"{color}{text}{RESET if color else ''}{end}")
        sys.stdout.flush()


def stream_token(token: str) -> None:
    sys.stdout.write(token)
    sys.stdout.flush()


def stream_thinking(token: str) -> None:
    """Stream the model's thinking dimmed, so long thinks show progress
    instead of a silent console."""
    sys.stdout.write(f"{DIM}{token}{RESET}" if COLOR else token)
    sys.stdout.flush()


def completion_matches(text: str, begidx: int, cwd: str,
                       commands: list[str]) -> list[str]:
    """Candidates for tab completion. First word: known commands and slash
    commands. Arguments: files/dirs relative to the current shell cwd."""
    if begidx == 0:
        low = text.lower()
        return sorted(c for c in commands if c.lower().startswith(low))
    t = text.strip("\"'").replace("\\", "/")
    d, _, prefix = t.rpartition("/")
    from pathlib import Path
    folder = Path(cwd) / d if d else Path(cwd)
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    out = []
    for n in names:
        if n.lower().startswith(prefix.lower()):
            full = (d + "/" if d else "") + n
            if (folder / n).is_dir():
                full += "/"
            out.append(full)
    return sorted(out)


def setup_completion(commands: list[str], get_cwd) -> None:
    """Enable tab completion in input() (no-op without readline)."""
    if not _HAS_READLINE:
        return

    def complete(text: str, state: int):
        try:
            matches = completion_matches(
                text, readline.get_begidx(), get_cwd(), commands)
        except Exception:  # a completer must never crash the prompt
            matches = []
        return matches[state] if state < len(matches) else None

    readline.set_completer(complete)
    readline.set_completer_delims(" \t")
    readline.parse_and_bind("tab: complete")


def init_history(path) -> None:
    """Persist input history across sessions (no-op without readline)."""
    if not _HAS_READLINE:
        return
    import atexit
    try:
        readline.read_history_file(str(path))
    except OSError:
        pass
    readline.set_history_length(500)

    def _save() -> None:
        try:
            readline.write_history_file(str(path))
        except OSError:
            pass
    atexit.register(_save)


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


def flush_stdin() -> None:
    """Discard keystrokes typed while the model was busy (streaming/thinking)
    so they do not accidentally answer the next prompt."""
    try:
        if os.name == "nt":
            import msvcrt
            while msvcrt.kbhit():
                msvcrt.getwch()
        elif sys.stdin.isatty():
            import termios
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except (ImportError, OSError, ValueError):
        pass


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
                flush_stdin()
                out(f"\n{BOLD}{YELLOW}{tag}{prompt}{RESET}")
                out(f"{DIM}  y = yes | n = no | a = yes to all (this task) | "
                    f"or type feedback for the agent{RESET}")
                while True:
                    try:
                        answer = input(prompt_str(f"{BOLD}  approve? {RESET}")).strip()
                    except (EOFError, KeyboardInterrupt):
                        return False, "user aborted"
                    if answer:  # bare Enter is ambiguous — ask again
                        break
            low = answer.lower()
            if low in ("y", "yes", "д", "да"):
                return True, ""
            if low == "a":
                self.approve_all = True
                return True, ""
            if low in ("n", "no", "н", "нет"):
                return False, ""
            return False, answer  # anything else = rejection with feedback


CONFIRM = Confirmer()
