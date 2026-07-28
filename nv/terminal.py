"""Terminal passthrough: run shell commands in the same console as the chat.

- `!cmd` runs a command directly (no model, no confirmation — the user
  typed it themselves). Bare lines that clearly look like commands
  (`ls`, `git status`, `env | grep X`) run the same way.
- `!!description` asks the model to build the command from natural
  language; the user confirms or corrects it before it runs.
- Captured output goes into a buffer that is attached as context to the
  NEXT model prompt — no switching between terminals.
- Interactive programs (vim, less, ...) run attached to the console.

These are the user's own commands: the agent sandbox/denylist does not
apply. Commands start in the project root; `cd` is handled as a builtin
and persists for later commands (the agent's own sandbox stays pinned
to the project root).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from nv import ui
from nv.config import Config
from nv.ollama import OllamaClient, OllamaError, strip_thinking

# programs that need the real console (output cannot be captured)
INTERACTIVE = {"vim", "vi", "nvim", "nano", "less", "more", "top", "htop",
               "ssh", "cmd", "powershell", "pwsh", "wsl"}
REPLS = {"python", "py", "ipython", "node"}  # interactive only with no args

# common commands + PowerShell aliases that `which` may not resolve
KNOWN = {"ls", "dir", "cat", "type", "cp", "copy", "mv", "move", "rm", "del",
         "cd",
         "echo", "pwd", "gci", "gc", "sls", "git", "python", "py", "pip",
         "uv", "npm", "npx", "node", "pytest", "tox", "ruff", "mypy",
         "docker", "kubectl", "helm", "env", "set", "grep", "find",
         "findstr", "tree", "where", "which", "head", "tail", "wc", "make",
         "go", "cargo", "dotnet", "java", "mvn", "gradle"}

# natural-language filler words -> the line is a prompt, not a command
_NL_WORDS = {"the", "a", "an", "to", "in", "on", "of", "for", "all", "and",
             "or", "with", "from", "into", "please", "me", "my", "it", "this",
             "that", "what", "why", "how", "can", "should", "file", "files",
             "код", "все", "в", "и", "для", "по", "это", "мне", "файл",
             "файлы", "что", "как", "почему", "сделай", "запусти", "покажи"}


# args that may follow a bare command without an explicit ! prefix
SUBCOMMANDS = {
    "git": {"status", "diff", "log", "show", "branch", "stash", "blame",
            "tag", "describe", "shortlog", "worktree"},
    "docker": {"ps", "images", "info", "version", "stats"},
    "kubectl": {"get", "describe", "version", "config", "logs", "top"},
    "npm": {"ls", "list", "outdated", "test", "version", "audit"},
    "pip": {"list", "show", "freeze", "check"},
    "cargo": {"build", "test", "check", "clippy", "fmt"},
    "go": {"build", "test", "vet", "version", "env"},
    "helm": {"list", "ls", "status", "version"},
}


def _arg_ok(first: str, tok: str) -> bool:
    t = tok.lower()
    return (t.startswith("-") or t.isdigit()
            or any(ch in t for ch in "./\\=:*|>\"'")
            or t in SUBCOMMANDS.get(first, set()))


def looks_like_command(line: str, cwd: str = "") -> bool:
    """Conservative: a bare line auto-runs only when the first word is a
    known command AND every argument looks like a flag/path/number/known
    subcommand, or names something that exists in cwd. 'make coverage
    report' or 'go faster' must NOT execute — anything ambiguous goes to
    the model instead (use ! to force)."""
    tokens = line.split()
    if not tokens or line.startswith(("/", "!")):
        return False
    first = tokens[0].lower()
    if first not in KNOWN and not shutil.which(tokens[0]):
        return False
    if any(t.lower().strip(",.?") in _NL_WORDS for t in tokens[1:]):
        return False
    if first == "cd":  # its argument is a dir name by definition;
        return len(tokens) <= 2  # the cd builtin errors clearly if wrong
    base = Path(cwd or ".")

    def exists(tok: str) -> bool:
        try:
            return (base / tok.strip("\"'")).exists()
        except OSError:
            return False

    return all(_arg_ok(first, t) or exists(t) for t in tokens[1:])


class TerminalBuffer:
    """Recent terminal activity, attached to the next model prompt."""

    def __init__(self) -> None:
        self.entries: list[tuple[str, str]] = []
        self.cwd: str = ""  # user's shell cwd; "" = project root

    def add(self, cmd: str, output: str) -> None:
        self.entries.append((cmd, (output or "").strip()[-3000:]))
        self.entries = self.entries[-5:]

    def wrap(self, prompt: str) -> str:
        if not self.entries:
            return prompt
        blob = "\n\n".join(f"$ {c}\n{o}".strip() for c, o in self.entries)
        self.entries = []
        return ("TERMINAL CONTEXT — commands the user just ran in this "
                "console, with their output (reference material, already "
                "executed):\n```\n" + blob + "\n```\n\n" + prompt)


def _decode(data: bytes | None) -> str:
    """Console programs on Windows often emit the OEM codepage, not UTF-8."""
    if not data:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return data.decode("oem", errors="replace")  # Windows only
        except LookupError:
            return data.decode("utf-8", errors="replace")


def _shell_name(cfg: Config) -> str:
    shell = getattr(cfg, "shell", "")
    if shell:
        return shell
    return "powershell" if os.name == "nt" else "sh"


def _shell_args(cfg: Config, cmd: str):
    shell = _shell_name(cfg)
    if shell == "powershell":
        return ["powershell", "-NoProfile", "-Command", cmd], False
    if shell == "bash":
        return ["bash", "-lc", cmd], False
    return cmd, True  # cmd.exe on Windows / sh elsewhere


def cwd_label(cfg: Config, buffer: TerminalBuffer) -> str:
    """Short label of the user's shell cwd, '' when at the project root."""
    here = buffer.cwd
    if not here or os.path.abspath(here) == os.path.abspath(cfg.root):
        return ""
    rel = os.path.relpath(here, cfg.root)
    return here if rel.startswith("..") else rel


def _change_dir(cfg: Config, cmd: str, buffer: TerminalBuffer) -> tuple[int, str]:
    target = cmd.split(maxsplit=1)[1].strip().strip("\"'") if " " in cmd else ""
    base = buffer.cwd or cfg.root
    new = (cfg.root if not target or target == "~"
           else os.path.abspath(os.path.join(base, os.path.expanduser(target))))
    if not os.path.isdir(new):
        msg = f"no such directory: {new}"
        ui.error(msg)
        buffer.add(cmd, msg)
        return 1, msg
    buffer.cwd = new
    ui.info(new)
    buffer.add(cmd, new)
    return 0, new


def run(cfg: Config, cmd: str, buffer: TerminalBuffer) -> tuple[int, str]:
    """Run a user command; print its output; record it in the buffer.
    Returns (exit_code, captured_output)."""
    tokens = cmd.split()
    if not tokens:
        return 0, ""
    first = tokens[0].lower().rsplit(".", 1)[0]
    if first == "cd" and len(tokens) <= 2:
        return _change_dir(cfg, cmd, buffer)
    cwd = buffer.cwd or cfg.root
    args, use_shell = _shell_args(cfg, cmd)

    if first in INTERACTIVE or (first in REPLS and len(tokens) == 1):
        ui.info("[interactive program — output will not be captured]")
        try:
            subprocess.run(args, shell=use_shell, cwd=cwd)
        except KeyboardInterrupt:
            pass
        buffer.add(cmd, "[interactive session, output not captured]")
        return 0, ""

    try:
        proc = subprocess.run(
            args, shell=use_shell, cwd=cwd, capture_output=True,
            timeout=max(300, cfg.command_timeout))
    except subprocess.TimeoutExpired:
        ui.error("command timed out")
        buffer.add(cmd, "[timed out]")
        return 1, "[timed out]"
    except KeyboardInterrupt:
        ui.warn("^C")
        return 130, ""
    except OSError as e:
        ui.error(str(e))
        return 1, str(e)

    out = _decode(proc.stdout)
    if proc.stderr:
        out += ("\n" if out else "") + _decode(proc.stderr)
    if out.strip():
        ui.out(out.rstrip("\n"))
    if proc.returncode != 0:
        ui.warn(f"exit code {proc.returncode}")
        out += f"\n[exit code {proc.returncode}]"
    buffer.add(cmd, out)
    return proc.returncode, out


_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def synthesize(cfg: Config, request: str, error_ctx: str = "") -> str | None:
    """Natural language -> one shell command, confirmed by the user.
    Returns the approved command, or None."""
    shell = _shell_name(cfg)
    client = OllamaClient(cfg.host, cfg.model, num_ctx=min(cfg.num_ctx, 8192),
                          temperature=0.1, num_predict=200)
    messages = [
        {"role": "system", "content":
            f"You convert a request into exactly ONE {shell} command line "
            f"(the OS is {'Windows' if os.name == 'nt' else 'unix'}; the "
            "command runs in the project root). Output ONLY the command — "
            "no code fences, no explanation, no comments."},
        {"role": "user", "content":
            request + (f"\n\nIt must fix this failure:\n{error_ctx}"
                       if error_ctx else "")},
    ]
    for _ in range(4):
        try:
            reply = client.chat(messages)
        except OllamaError as e:
            ui.error(str(e))
            return None
        cmd = strip_thinking(reply.get("content") or "").strip()
        cmd = _FENCE.sub("", cmd).strip().splitlines()[0].strip() if cmd else ""
        if not cmd:
            ui.warn("the model returned no command")
            return None
        ui.out(f"  $ {cmd}", ui.BOLD)
        try:
            answer = input("run it? [y/n or type a correction] ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        low = answer.lower()
        if low in ("y", "yes", "д", "да"):
            return cmd
        if low in ("n", "no", "", "н", "нет"):
            return None
        messages.append({"role": "assistant", "content": cmd})
        messages.append({"role": "user", "content":
                         f"Correction: {answer}. Output only the fixed command."})
    ui.warn("too many corrections — cancelled")
    return None
