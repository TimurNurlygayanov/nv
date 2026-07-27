"""Configuration for nv: Ollama host, model, context limits.

Priority: CLI flags > project .nv.json > ~/.nv.json > defaults.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

GLOBAL_CONFIG = Path.home() / ".nv.json"
PROJECT_CONFIG_NAME = ".nv.json"

DEFAULTS = {
    "host": "http://localhost:11434",
    "model": "qwen3-coder:30b",
    "review_model": "",          # empty -> use "model"
    "num_ctx": 16384,
    "temperature": 0.2,
    "max_steps": 40,             # max agent iterations per task
    "plan_first": True,          # architect plans + user approval before coding
    "max_read_lines": 200,       # max lines returned by read_file per call
    "max_search_hits": 40,
    "max_tool_output": 4000,     # chars of tool output kept in history
    "command_timeout": 120,      # seconds for run_command
    "history_char_budget": 45000,  # ~ chars before history compaction kicks in
    "confirm_over_seconds": 60,  # ask before jobs estimated longer than this
    "shell": "",                 # shell for !commands: powershell/cmd/bash ('' = auto)
    "theme": {},                 # console theme, managed via /theme or the agent
    "personality": "",           # answer style for chat agents, e.g. "flirty"
    "max_answer_tokens": 2048,   # hard cap on model output per reply (num_predict)
    "max_diff_lines": 60,        # bigger changes always need explicit approval
}


@dataclass
class Config:
    host: str = DEFAULTS["host"]
    model: str = DEFAULTS["model"]
    review_model: str = DEFAULTS["review_model"]
    num_ctx: int = DEFAULTS["num_ctx"]
    temperature: float = DEFAULTS["temperature"]
    max_steps: int = DEFAULTS["max_steps"]
    plan_first: bool = DEFAULTS["plan_first"]
    max_read_lines: int = DEFAULTS["max_read_lines"]
    max_search_hits: int = DEFAULTS["max_search_hits"]
    max_tool_output: int = DEFAULTS["max_tool_output"]
    command_timeout: int = DEFAULTS["command_timeout"]
    history_char_budget: int = DEFAULTS["history_char_budget"]
    confirm_over_seconds: int = DEFAULTS["confirm_over_seconds"]
    shell: str = DEFAULTS["shell"]
    theme: dict = field(default_factory=dict)
    personality: str = DEFAULTS["personality"]
    max_answer_tokens: int = DEFAULTS["max_answer_tokens"]
    max_diff_lines: int = DEFAULTS["max_diff_lines"]
    root: Path = field(default_factory=Path.cwd)

    @property
    def reviewer_model(self) -> str:
        return self.review_model or self.model


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def load_config(root: Path) -> Config:
    data = dict(DEFAULTS)
    data.update(_read_json(GLOBAL_CONFIG))
    data.update(_read_json(root / PROJECT_CONFIG_NAME))
    known = {k: v for k, v in data.items() if k in DEFAULTS}
    known["theme"] = dict(known.get("theme") or {})  # never share the default
    cfg = Config(**known)
    cfg.root = root.resolve()
    return cfg


def save_global(cfg: Config) -> None:
    data = {k: v for k, v in asdict(cfg).items() if k in DEFAULTS}
    GLOBAL_CONFIG.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_agents_md(root: Path, limit: int = 4000) -> str:
    """Project rules file injected into agent system prompts."""
    for name in ("AGENTS.md", "agents.md", "CLAUDE.md"):
        p = root / name
        if p.is_file():
            text = p.read_text(encoding="utf-8", errors="replace").strip()
            if len(text) > limit:
                text = text[:limit] + "\n[...truncated...]"
            return text
    return ""
