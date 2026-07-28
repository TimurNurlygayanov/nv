"""The agent loop: prompt -> model -> tool calls -> results -> repeat.

Includes history compaction so long sessions do not overflow the small
context window of local models.
"""
from __future__ import annotations

import json

from nv import session, ui
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
            num_predict=cfg.max_answer_tokens,
        )
        self.toolbox = Toolbox(cfg, agent_name=self.name)
        self.schemas = tool_schemas(spec["tools"])
        self.stop_event = None  # set by team mode for cooperative Ctrl+C
        self._agents_md = agents_md
        self._listing = self.toolbox.list_files(".", max_entries=40)
        self.system = ""
        self._refresh_system()
        self.messages: list[dict] = [{"role": "system", "content": self.system}]

    def _refresh_system(self) -> None:
        """Rebuild the system prompt so notes and personality changes made
        mid-session take effect on the next run."""
        self.system = build_system_prompt(
            self.kind, self._agents_md, self._listing,
            session.load_notes(self.cfg.root),
            personality=self.cfg.personality)

    def reset(self) -> None:
        self._refresh_system()
        self.messages = [{"role": "system", "content": self.system}]

    # ---- context compaction -----------------------------------------

    def _history_size(self) -> int:
        return sum(len(m.get("content") or "") for m in self.messages)

    def _compact(self) -> None:
        """Shrink history over budget. Order matters: old tool outputs first,
        then old assistant/tool turns; USER messages (the original issue and
        the feedback) survive longest — dropping them is exactly how an
        agent 'forgets' what it was fixing mid-conversation."""
        budget = self.cfg.history_char_budget
        if self._history_size() <= budget:
            return
        # 1) shrink old tool results (keep the last 4 messages intact)
        for m in self.messages[1:-4]:
            content = m.get("content") or ""
            if m.get("role") == "tool" and len(content) > 300:
                m["content"] = content[:300] + "\n[...old tool output truncated...]"
        # 2) drop oldest assistant/tool messages, keeping user messages;
        #    a dropped assistant tool call takes its tool results with it —
        #    an orphaned tool result confuses small models
        i = 1
        while (self._history_size() > budget and len(self.messages) > 6
               and i < len(self.messages) - 4):
            if self.messages[i].get("role") == "user":
                i += 1
                continue
            del self.messages[i]
            while (i < len(self.messages) - 1
                   and self.messages[i].get("role") == "tool"):
                del self.messages[i]
        # 3) trim huge user messages (keep the head — that's where the ask
        #    lives; pasted logs/tracebacks tail off)
        for m in self.messages[1:-4]:
            if self._history_size() <= budget:
                break
            content = m.get("content") or ""
            if m.get("role") == "user" and len(content) > 2000:
                m["content"] = content[:2000] + "\n[...input trimmed...]"
        # 4) last resort: drop oldest turns regardless of role
        while self._history_size() > budget and len(self.messages) > 6:
            del self.messages[1]
            while (len(self.messages) > 6
                   and self.messages[1].get("role") == "tool"):
                del self.messages[1]
        # 5) and if even that failed: truncate recent tool outputs
        if self._history_size() > budget:
            for m in self.messages[1:]:
                content = m.get("content") or ""
                if m.get("role") == "tool" and len(content) > 1000:
                    m["content"] = content[:1000] + "\n[...truncated...]"
        if not self.quiet:
            ui.info(f"[{self.name}] history compacted to ~{self._history_size()} chars")

    # ---- main loop ---------------------------------------------------

    def run(self, task: str) -> str:
        self._refresh_system()
        self.messages[0] = {"role": "system", "content": self.system}
        self.messages.append({"role": "user", "content": task})
        final = ""
        for step in range(1, self.cfg.max_steps + 1):
            if self.stop_event is not None and self.stop_event.is_set():
                return "stopped by user (Ctrl+C)"
            self._compact()
            try:
                reply = self.client.chat(
                    self.messages,
                    tools=self.schemas,
                    on_token=None if self.quiet else ui.stream_token,
                    on_thinking=None if self.quiet else ui.stream_thinking,
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
                           ui.c("tool"))
                result = self.toolbox.call(tname, targs)
                if not self.quiet:
                    preview = result if len(result) <= 400 else result[:400] + "…"
                    ui.info(preview)
                cap = self.cfg.max_tool_output
                if tname == "analyze_files":  # digests are the whole point
                    cap = max(cap, 12000)
                if len(result) > cap:
                    result = result[:cap] + "\n[...output truncated...]"
                self.messages.append({"role": "tool", "content": result,
                                      "tool_name": tname})
        else:
            final = f"stopped: reached max_steps={self.cfg.max_steps} without finishing"
            ui.warn(f"[{self.name}] {final}")
        return final
