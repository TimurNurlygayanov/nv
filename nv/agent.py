"""The agent loop: prompt -> model -> tool calls -> results -> repeat.

Includes history compaction so long sessions do not overflow the small
context window of local models.
"""
from __future__ import annotations

import json

from nv import ui
from nv.config import Config
from nv.ollama import OllamaClient, OllamaError, strip_thinking
from nv.prompts import AGENT_CONFIGS, build_system_prompt
from nv.tools import Toolbox, tool_schemas


class Agent:
    def __init__(self, cfg: Config, kind: str = "coder", name: str = "",
                 agents_md: str = "", model: str = "", quiet: bool = False) -> None:
        self.cfg = cfg
        self.kind = kind
        self.name = name or kind
        self.quiet = quiet
        spec = AGENT_CONFIGS[kind]
        self.client = OllamaClient(
            host=cfg.host,
            model=model or cfg.model,
            num_ctx=cfg.num_ctx,
            temperature=spec.get("temperature", cfg.temperature),
        )
        self.toolbox = Toolbox(cfg, agent_name=self.name)
        self.schemas = tool_schemas(spec["tools"])
        listing = self.toolbox.list_files(".", max_entries=40)
        self.system = build_system_prompt(kind, agents_md, listing)
        self.messages: list[dict] = [{"role": "system", "content": self.system}]

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system}]

    # ---- context compaction -----------------------------------------

    def _history_size(self) -> int:
        return sum(len(m.get("content") or "") for m in self.messages)

    def _compact(self) -> None:
        """Truncate old tool outputs first; if still too big, drop oldest turns."""
        budget = self.cfg.history_char_budget
        if self._history_size() <= budget:
            return
        # 1) shrink old tool results (keep the last 4 messages intact)
        for m in self.messages[1:-4]:
            content = m.get("content") or ""
            if m.get("role") == "tool" and len(content) > 300:
                m["content"] = content[:300] + "\n[...old tool output truncated...]"
        # 2) drop oldest non-system messages while over budget
        while self._history_size() > budget and len(self.messages) > 6:
            del self.messages[1]
        # 3) last resort: truncate even recent tool outputs
        if self._history_size() > budget:
            for m in self.messages[1:]:
                content = m.get("content") or ""
                if m.get("role") == "tool" and len(content) > 1000:
                    m["content"] = content[:1000] + "\n[...truncated...]"
        if not self.quiet:
            ui.info(f"[{self.name}] history compacted to ~{self._history_size()} chars")

    # ---- main loop ---------------------------------------------------

    def run(self, task: str) -> str:
        self.messages.append({"role": "user", "content": task})
        final = ""
        for step in range(1, self.cfg.max_steps + 1):
            self._compact()
            try:
                reply = self.client.chat(
                    self.messages,
                    tools=self.schemas,
                    on_token=None if self.quiet else ui.stream_token,
                    on_thinking=None,
                )
            except OllamaError as e:
                ui.error(f"\n[{self.name}] {e}")
                return f"ERROR: {e}"

            content = strip_thinking(reply.get("content") or "")
            tool_calls = reply.get("tool_calls") or []
            self.messages.append({
                "role": "assistant",
                "content": reply.get("content") or "",
                "tool_calls": tool_calls or None,
            })

            if not tool_calls:
                if not self.quiet:
                    ui.out()  # newline after streamed text
                final = content
                break

            for tc in tool_calls:
                fn = tc.get("function") or {}
                tname = fn.get("name", "")
                targs = fn.get("arguments") or {}
                if isinstance(targs, str):
                    try:
                        targs = json.loads(targs)
                    except ValueError:
                        targs = {}
                if not self.quiet:
                    brief = {k: (v[:80] + "…" if isinstance(v, str) and len(v) > 80 else v)
                             for k, v in targs.items()}
                    ui.out(f"\n[{self.name}] step {step}: {tname}({json.dumps(brief, ensure_ascii=False)})",
                           ui.MAGENTA)
                result = self.toolbox.call(tname, targs)
                if not self.quiet:
                    preview = result if len(result) <= 400 else result[:400] + "…"
                    ui.info(preview)
                cap = self.cfg.max_tool_output
                if len(result) > cap:
                    result = result[:cap] + "\n[...output truncated...]"
                self.messages.append({"role": "tool", "content": result,
                                      "tool_name": tname})
        else:
            final = f"stopped: reached max_steps={self.cfg.max_steps} without finishing"
            ui.warn(f"[{self.name}] {final}")
        return final
