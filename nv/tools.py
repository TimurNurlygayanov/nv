"""Agent tools: sandboxed file operations, search, and shell commands.

Every mutating tool (write_file, edit_file, run_command) asks the user
for confirmation and shows a preview first.
"""
from __future__ import annotations

import difflib
import fnmatch
import os
import re
import subprocess
import tempfile
from pathlib import Path

from nv.config import Config, save_global
from nv.sandbox import Sandbox, SandboxError
from nv import checkpoint, session, ui

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv",
             ".idea", ".vscode", "dist", "build", ".mypy_cache",
             ".pytest_cache", ".nv", "*.egg-info"}

TEXT_EXT_HINT = {".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".md", ".txt",
                 ".yml", ".yaml", ".toml", ".cfg", ".ini", ".html", ".css",
                 ".sh", ".ps1", ".bat", ".go", ".rs", ".java", ".c", ".cpp",
                 ".h", ".sql", ".xml", ".csv", ".env"}


def _skip(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in SKIP_DIRS)


def _changed_lines(old: str, new: str) -> int:
    return sum(1 for ln in difflib.unified_diff(
        old.splitlines(), new.splitlines(), lineterm="")
        if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---")))


def _read_raw(p: Path) -> str:
    """Read without newline translation, so CRLF files stay detectable."""
    with open(p, encoding="utf-8", errors="replace", newline="") as f:
        return f.read()


def _detect_eol(raw: str) -> str:
    crlf = raw.count("\r\n")
    lf = raw.count("\n") - crlf
    return "\r\n" if crlf > lf else "\n"


def _write_atomic(p: Path, content: str, eol: str) -> None:
    """Write via temp file + os.replace (a crash never truncates the target),
    re-applying the file's original line-ending style."""
    data = content.replace("\r\n", "\n")
    if eol == "\r\n":
        data = data.replace("\n", "\r\n")
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name + ".",
                               suffix=".nv-tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(data)
        os.replace(tmp, p)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class Toolbox:
    def __init__(self, cfg: Config, agent_name: str = "agent") -> None:
        self.cfg = cfg
        self.sandbox = Sandbox(cfg.root)
        self.agent_name = agent_name

    # ---- read-only tools (no confirmation) ---------------------------

    def _walk(self, base: Path):
        """Files under base, pruning skipped dirs BEFORE descending (rglob
        would materialize .git/node_modules first and stall on big repos)."""
        if base.is_file():
            yield base
            return
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if not _skip(d))
            for fn in sorted(filenames):
                if not _skip(fn):
                    yield Path(dirpath) / fn

    def list_files(self, path: str = ".", max_entries: int = 200) -> str:
        base = self.sandbox.resolve(path)
        if base.is_file():
            return str(base.relative_to(self.cfg.root))
        entries: list[str] = []
        for p in self._walk(base):
            entries.append(str(p.relative_to(self.cfg.root)).replace("\\", "/"))
            if len(entries) >= max_entries:
                entries.append(f"[... more than {max_entries} files, "
                               "use list_files on a subfolder or search]")
                break
        return "\n".join(entries) or "(empty folder)"

    def read_file(self, path: str, start_line: int = 1, max_lines: int = 0) -> str:
        p = self.sandbox.resolve(path)
        if not p.is_file():
            return f"ERROR: file not found: {path}. Use list_files or search to find the right path."
        limit = max_lines or self.cfg.max_read_lines
        limit = min(limit, self.cfg.max_read_lines)
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        start = max(1, int(start_line))
        chunk = lines[start - 1:start - 1 + limit]
        numbered = "\n".join(f"{i}| {ln}" for i, ln in enumerate(chunk, start))
        total = len(lines)
        header = f"[{path}: lines {start}-{start + len(chunk) - 1} of {total}]"
        if start + len(chunk) - 1 < total:
            header += f" (call read_file with start_line={start + len(chunk)} for more)"
        return f"{header}\n{numbered}" if chunk else f"[{path}: empty or start_line beyond end, total {total} lines]"

    def search(self, query: str, path: str = ".", use_regex: bool = False) -> str:
        base = self.sandbox.resolve(path)
        try:
            pattern = re.compile(query if use_regex else re.escape(query), re.IGNORECASE)
        except re.error as e:
            return f"ERROR: bad regex: {e}"
        hits: list[str] = []
        for p in self._walk(base):
            rel = p.relative_to(self.cfg.root)
            try:
                size = p.stat().st_size
                if size > 2_000_000:
                    continue  # no text worth line-searching is this big
                if p.suffix.lower() not in TEXT_EXT_HINT and size > 512_000:
                    continue
                for i, line in enumerate(
                        p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if pattern.search(line):
                        hits.append(f"{str(rel).replace(chr(92), '/')}:{i}: {line.strip()[:160]}")
                        if len(hits) >= self.cfg.max_search_hits:
                            hits.append("[... more results truncated, refine the query]")
                            return "\n".join(hits)
            except OSError:
                continue
        return "\n".join(hits) or f"no matches for '{query}'"

    def git_diff(self, stat_only: bool = False) -> str:
        def _git(*args):
            return subprocess.run(["git", *args], cwd=self.cfg.root,
                                  capture_output=True, text=True,
                                  encoding="utf-8", errors="replace",
                                  timeout=30)
        if _git("rev-parse", "--git-dir").returncode != 0:
            return "(not a git repository — no diff available)"
        args = ["diff"] + (["--stat"] if stat_only else [])
        result = _git(*args)
        if result.returncode != 0:
            return f"ERROR: git diff failed: {result.stderr.strip()[:300]}"
        out = result.stdout.strip()
        # list untracked files WITHOUT touching the user's index
        untracked = _git("ls-files", "--others", "--exclude-standard")
        names = [n for n in untracked.stdout.splitlines() if n.strip()][:50]
        if names:
            out += ("\n\n" if out else "") + "untracked new files:\n" + \
                   "\n".join("  " + n for n in names)
        return out or "(no changes)"

    # ---- mutating tools (confirmation required) ----------------------

    def write_file(self, path: str, content: str) -> str:
        try:
            p = self.sandbox.resolve(path, for_write=True)
        except SandboxError as e:
            return f"ERROR: {e}"
        old, raw, eol = "", None, "\n"
        if p.is_file():
            raw = _read_raw(p)
            eol = _detect_eol(raw)
            old = raw.replace("\r\n", "\n")
            if old == content.replace("\r\n", "\n"):
                return f"no changes: {path} already has this exact content"
        ui.banner(f"{self.agent_name} wants to write {path}")
        ui.render_diff(old, content, path)
        changed = _changed_lines(old, content)
        big = changed > self.cfg.max_diff_lines
        if big:
            ui.warn(f"LARGE CHANGE: {changed} changed lines "
                    f"(minimal-change limit is {self.cfg.max_diff_lines}) — "
                    "needs explicit approval")
        ok, feedback = ui.CONFIRM.ask(f"write {path}?", self.agent_name,
                                      force=big)
        if not ok:
            hint = (f" The change was too large ({changed} lines > "
                    f"{self.cfg.max_diff_lines}): make the smallest change "
                    "that solves the task, or split it into steps."
                    if big else " Do not retry the same change.")
            return f"REJECTED by user.{(' User says: ' + feedback) if feedback else hint}"
        p.parent.mkdir(parents=True, exist_ok=True)
        if raw is not None and p.is_file() and _read_raw(p) != raw:
            return (f"ERROR: {path} changed on disk while waiting for "
                    "approval — re-read the file and retry")
        _write_atomic(p, content, eol)
        return f"written: {path} ({len(content.splitlines())} lines)"

    def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        try:
            p = self.sandbox.resolve(path, for_write=True)
        except SandboxError as e:
            return f"ERROR: {e}"
        if not p.is_file():
            return f"ERROR: file not found: {path}"
        raw = _read_raw(p)
        eol = _detect_eol(raw)
        text = raw.replace("\r\n", "\n")
        old_text = old_text.replace("\r\n", "\n")
        new_text = new_text.replace("\r\n", "\n")
        count = text.count(old_text)
        if count == 0:
            return ("ERROR: old_text not found in file. Re-read the file with "
                    "read_file and copy the exact text including whitespace.")
        if count > 1:
            return (f"ERROR: old_text appears {count} times. Include more "
                    "surrounding lines to make it unique.")
        new_content = text.replace(old_text, new_text, 1)
        ui.banner(f"{self.agent_name} wants to edit {path}")
        ui.render_diff(text, new_content, path)
        changed = _changed_lines(text, new_content)
        big = changed > self.cfg.max_diff_lines
        if big:
            ui.warn(f"LARGE CHANGE: {changed} changed lines "
                    f"(minimal-change limit is {self.cfg.max_diff_lines}) — "
                    "needs explicit approval")
        ok, feedback = ui.CONFIRM.ask(f"edit {path}?", self.agent_name,
                                      force=big)
        if not ok:
            hint = (f" The change was too large ({changed} lines > "
                    f"{self.cfg.max_diff_lines}): make the smallest change "
                    "that solves the task, or split it into steps."
                    if big else " Do not retry the same change.")
            return f"REJECTED by user.{(' User says: ' + feedback) if feedback else hint}"
        if _read_raw(p) != raw:
            return (f"ERROR: {path} changed on disk while waiting for "
                    "approval — re-read the file and retry")
        _write_atomic(p, new_content, eol)
        return f"edited: {path}"

    def run_command(self, command: str) -> str:
        try:
            self.sandbox.check_command(command)
        except SandboxError as e:
            return f"ERROR: {e}"
        ui.banner(f"{self.agent_name} wants to run a command")
        ui.out(f"  $ {command}", ui.BOLD)
        ok, feedback = ui.CONFIRM.ask("run this command?", self.agent_name)
        if not ok:
            return f"REJECTED by user. {('User says: ' + feedback) if feedback else 'Do not retry the same command.'}"
        try:
            result = subprocess.run(
                command, shell=True, cwd=self.cfg.root, capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=self.cfg.command_timeout)
        except subprocess.TimeoutExpired:
            return f"ERROR: command timed out after {self.cfg.command_timeout}s"
        cap = self.cfg.max_tool_output
        stdout = result.stdout[-cap:] if result.stdout else ""
        stderr = result.stderr[-cap:] if result.stderr else ""
        parts = [f"exit code: {result.returncode}"]
        if stdout:
            parts.append(f"stdout:\n{stdout}")
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        return "\n".join(parts)

    def analyze_files(self, path: str, goal: str) -> str:
        """Deep analysis of big files/folders via the ingest pipeline.
        The pipeline itself asks the user to confirm file selection and
        long jobs, so no extra gate is needed here."""
        try:
            p = self.sandbox.resolve(path or ".")
        except SandboxError as e:
            return f"ERROR: {e}"
        from nv import ingest  # local import: ingest imports tools' constants
        from nv.ollama import OllamaError
        try:
            res = ingest.interactive_pipeline(self.cfg, p, goal)
        except OllamaError as e:
            return f"ERROR: {e}"
        if res is None:
            return ("cancelled: the user declined the analysis or nothing "
                    "was found. Do not retry; ask the user how to proceed.")
        text, summarized, source = res
        kind = "digest" if summarized else "content"
        return f"[{source} — {kind}]\n{text}"

    def undo_changes(self) -> str:
        cp = checkpoint.load_last(self.cfg.root)
        if not cp:
            return "no checkpoint found — nothing to undo"
        stat = checkpoint.diff_stat(self.cfg.root, cp)
        if not stat:
            return "nothing to undo — working tree matches the checkpoint"
        ui.banner(f"{self.agent_name} wants to revert to the last checkpoint")
        ui.out(stat)
        ok, feedback = ui.CONFIRM.ask("revert the working tree?", self.agent_name)
        if not ok:
            return f"REJECTED by user. {('User says: ' + feedback) if feedback else ''}"
        return checkpoint.restore(self.cfg.root, cp)

    def configure_ui(self, changes: dict) -> str:
        """Apply console-appearance changes (announced, not confirmed —
        instantly visible and reversible with /theme reset)."""
        from nv import theme  # theme imports ollama; keep tools import light
        applied: list[str] = []
        personality = changes.pop("personality", None)
        if personality is not None:
            self.cfg.personality = str(personality).strip()
            applied.append(f"answer personality: "
                           f"{self.cfg.personality or 'neutral (off)'}")
        patch, errors = theme.validate(changes)
        if not patch and not applied:
            return ("ERROR: no valid changes. " + "; ".join(errors)
                    if errors else "ERROR: nothing to change")
        if patch:
            applied += theme.apply(patch)
            self.cfg.theme.update(patch)
        save_global(self.cfg)
        result = "applied: " + "; ".join(applied)
        if errors:
            result += " | ignored: " + "; ".join(errors)
        return result + " (saved; /theme reset and /style off restore defaults)"

    def save_note(self, text: str) -> str:
        """Append a durable fact to the project notes (announced, not
        confirmed — stored outside the repo, never touches project code)."""
        text = text.strip()
        if not text:
            return "ERROR: empty note"
        session.add_note(self.cfg.root, text)
        ui.out(f"[{self.agent_name}] noted: {text[:140]}", ui.CYAN)
        return "note saved to project notes"

    # ---- dispatch ----------------------------------------------------

    def call(self, name: str, args: dict) -> str:
        try:
            if name == "list_files":
                return self.list_files(str(args.get("path", ".")))
            if name == "read_file":
                return self.read_file(str(args.get("path", "")),
                                      int(args.get("start_line", 1) or 1),
                                      int(args.get("max_lines", 0) or 0))
            if name == "search":
                return self.search(str(args.get("query", "")),
                                   str(args.get("path", ".")),
                                   bool(args.get("use_regex", False)))
            if name == "git_diff":
                return self.git_diff(bool(args.get("stat_only", False)))
            if name == "write_file":
                return self.write_file(str(args.get("path", "")),
                                       str(args.get("content", "")))
            if name == "edit_file":
                return self.edit_file(str(args.get("path", "")),
                                      str(args.get("old_text", "")),
                                      str(args.get("new_text", "")))
            if name == "run_command":
                return self.run_command(str(args.get("command", "")))
            if name == "save_note":
                return self.save_note(str(args.get("text", "")))
            if name == "analyze_files":
                return self.analyze_files(str(args.get("path", ".")),
                                          str(args.get("goal", "")))
            if name == "undo_changes":
                return self.undo_changes()
            if name == "configure_ui":
                return self.configure_ui(
                    {k: v for k, v in args.items() if k != "name"})
            return f"ERROR: unknown tool '{name}'"
        except SandboxError as e:
            return f"ERROR: {e}"
        except Exception as e:  # keep the agent loop alive on any tool bug
            return f"ERROR: tool '{name}' failed: {e}"


# JSON schemas sent to the model -------------------------------------

def tool_schemas(names: list[str]) -> list[dict]:
    all_schemas = {
        "list_files": {
            "description": "List files in the project (or a subfolder). "
                           "Recursive, capped. Use before guessing any path.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "folder, default '.'"}},
                "required": []},
        },
        "read_file": {
            "description": "Read part of a file with line numbers. Read only "
                           "the range you need; call again with start_line to continue.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "description": "1-based, default 1"},
                "max_lines": {"type": "integer", "description": "default/max 200"}},
                "required": ["path"]},
        },
        "search": {
            "description": "Search for text across project files. Returns "
                           "file:line: matches. Use this instead of reading many files.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"},
                "path": {"type": "string", "description": "limit to a subfolder"},
                "use_regex": {"type": "boolean"}},
                "required": ["query"]},
        },
        "git_diff": {
            "description": "Show current uncommitted git changes.",
            "parameters": {"type": "object", "properties": {
                "stat_only": {"type": "boolean", "description": "summary only"}},
                "required": []},
        },
        "write_file": {
            "description": "Create a new file or fully overwrite a small file. "
                           "For existing files prefer edit_file. User must approve.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}},
                "required": ["path", "content"]},
        },
        "edit_file": {
            "description": "Replace exact text in a file (must match exactly one "
                           "occurrence, copy it from read_file output without the "
                           "line-number prefix). User must approve.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"}},
                "required": ["path", "old_text", "new_text"]},
        },
        "run_command": {
            "description": "Run a shell command inside the project folder "
                           "(tests, linters, python scripts). No network, no "
                           "package installs. User must approve.",
            "parameters": {"type": "object", "properties": {
                "command": {"type": "string"}},
                "required": ["command"]},
        },
        "analyze_files": {
            "description": "Deep-analyze a big file or a whole folder (docs, "
                           "logs, specs) that is too large to read directly. "
                           "Relevant files are auto-selected and content is "
                           "distilled chunk by chunk into a digest focused on "
                           "the goal. Use for tasks like 'generate user "
                           "stories from the docs' or 'why does this huge CI "
                           "log fail'. Long jobs are estimated and confirmed "
                           "by the user.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string",
                         "description": "file or folder, e.g. 'docs' or 'ci.log'"},
                "goal": {"type": "string",
                         "description": "what to look for / what the digest is for"}},
                "required": ["path", "goal"]},
        },
        "undo_changes": {
            "description": "Revert the working tree to the checkpoint taken "
                           "before the current task. Call when the user asks "
                           "to undo/revert/rollback the changes. User must "
                           "approve.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        "configure_ui": {
            "description": "Change the UX of this chat console when the user "
                           "asks: colors, font size, or the assistant's "
                           "answer personality. Colors are names (green, "
                           "black, cyan, orange...) or #rrggbb. Include only "
                           "what the user asked to change.",
            "parameters": {"type": "object", "properties": {
                "fg": {"type": "string", "description": "default text color"},
                "bg": {"type": "string", "description": "background color"},
                "prompt": {"type": "string", "description": "nv> prompt color"},
                "banner": {"type": "string", "description": "section header color"},
                "info": {"type": "string", "description": "status line color"},
                "warn": {"type": "string"},
                "error": {"type": "string"},
                "tool": {"type": "string", "description": "tool-call line color"},
                "diff_add": {"type": "string"},
                "diff_del": {"type": "string"},
                "font_size": {"type": "string",
                              "description": "8..72, or relative like '+2'/'-4'"},
                "personality": {"type": "string",
                                "description": "answer style, e.g. 'playful "
                                               "and lightly flirty', 'dry "
                                               "sarcasm'; empty string = off"}},
                "required": []},
        },
        "save_note": {
            "description": "Save ONE short durable fact about the project to "
                           "the shared notes (env var meaning, gotcha, flaky "
                           "test, non-obvious behavior). Not for task progress.",
            "parameters": {"type": "object", "properties": {
                "text": {"type": "string", "description": "one sentence"}},
                "required": ["text"]},
        },
    }
    return [{"type": "function",
             "function": {"name": n, **all_schemas[n]}}
            for n in names if n in all_schemas]


CODER_TOOLS = ["list_files", "read_file", "search", "git_diff",
               "write_file", "edit_file", "run_command", "save_note",
               "analyze_files", "undo_changes", "configure_ui"]
READONLY_TOOLS = ["list_files", "read_file", "search", "git_diff"]
