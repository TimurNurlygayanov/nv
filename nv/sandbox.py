"""Sandbox: the agent may only see and touch the folder where nv started.

- Every file path is resolved and must stay inside the root folder.
- Writing into .git/ is blocked.
- Shell commands are checked against a denylist (network access, package
  installation, escaping the folder, destructive system commands).
"""
from __future__ import annotations

import re
from pathlib import Path


class SandboxError(Exception):
    pass


class Sandbox:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve(self, path: str, for_write: bool = False) -> Path:
        raw = (path or ".").strip().strip('"').strip("'")
        p = Path(raw)
        candidate = (p if p.is_absolute() else self.root / p).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise SandboxError(
                f"path '{path}' is outside the working folder — access denied")
        if for_write and ".git" in candidate.relative_to(self.root).parts:
            raise SandboxError("writing inside .git is not allowed")
        return candidate

    # -- command guard -------------------------------------------------

    _DENY_PATTERNS = [
        # network access
        r"\bcurl\b", r"\bwget\b", r"\bInvoke-WebRequest\b", r"\bInvoke-RestMethod\b",
        r"\biwr\b", r"\birm\b", r"\bssh\b", r"\bscp\b", r"\bftp\b", r"\btelnet\b",
        r"\bnc\b", r"\bncat\b", r"\bping\b", r"\bnslookup\b",
        r"https?://",
        # package installation
        r"\bpip3?\s+install\b", r"\bnpm\s+(install|i|add|ci)\b", r"\byarn\s+(add|install)\b",
        r"\bpnpm\s+(add|install|i)\b", r"\bpoetry\s+(add|install)\b", r"\buv\s+(pip|add|sync)\b",
        r"\bapt(-get)?\b", r"\bchoco\b", r"\bwinget\b", r"\bscoop\b", r"\bbrew\b",
        r"\bcargo\s+install\b", r"\bgo\s+(install|get)\b", r"\bdotnet\s+add\b",
        r"\bgem\s+install\b", r"\bconda\s+install\b",
        # pushing / publishing
        r"\bgit\s+push\b", r"\bgit\s+remote\b", r"\bgit\s+fetch\b", r"\bgit\s+pull\b",
        r"\bgit\s+clone\b", r"\bnpm\s+publish\b", r"\btwine\b",
        # escaping the sandbox / destructive
        r"\bcd\s+\.\.", r"\bcd\s+[a-zA-Z]:", r"\bcd\s+[/\\]", r"~[/\\]",
        r"\brm\s+-rf\s+[/\\]", r"\bformat\b", r"\bdel\s+[a-zA-Z]:", r"\breg\s",
        r"\bregedit\b", r"\bschtasks\b", r"\bshutdown\b", r"\bnet\s+user\b",
        r"\bRemove-Item\b.*-Recurse.*[a-zA-Z]:", r"\bsc\s+(stop|delete|config)\b",
    ]

    def check_command(self, command: str) -> None:
        cmd = command.strip()
        for pattern in self._DENY_PATTERNS:
            if re.search(pattern, cmd, re.IGNORECASE):
                raise SandboxError(
                    f"command blocked by sandbox (matched '{pattern}'): "
                    "no network, package installs, git remotes, or paths "
                    "outside the working folder")
        # absolute paths in the command must point inside root
        for m in re.finditer(r"[a-zA-Z]:[\\/][^\s\"']*", cmd):
            p = Path(m.group(0))
            try:
                resolved = p.resolve()
            except OSError:
                continue
            if resolved != self.root and self.root not in resolved.parents:
                raise SandboxError(
                    f"command references path outside the working folder: {m.group(0)}")
