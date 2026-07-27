"""Minimal Ollama chat client (stdlib only) with streaming and tool calls.

Works with the /api/chat endpoint. Supports:
- streaming tokens (callback)
- native tool calling (qwen3-coder etc.)
- fallback: parsing a JSON tool call out of plain text, for models that
  do not emit structured tool_calls reliably
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Callable, Optional

from nv import perf


class OllamaError(Exception):
    pass


class OllamaClient:
    def __init__(self, host: str, model: str, num_ctx: int = 16384,
                 temperature: float = 0.2, timeout: int = 600,
                 num_predict: int = 0) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout = timeout
        self.num_predict = num_predict

    def _options(self) -> dict:
        options = {"num_ctx": self.num_ctx, "temperature": self.temperature}
        if self.num_predict:
            options["num_predict"] = self.num_predict
        return options

    def _post(self, path: str, body: dict):
        req = urllib.request.Request(
            self.host + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.URLError as e:
            raise OllamaError(
                f"cannot reach Ollama at {self.host}: {e.reason}. "
                "Set the host with:  /host http://<ip>:11434") from e

    def ping(self) -> list[str]:
        """Returns list of available model names."""
        try:
            with urllib.request.urlopen(self.host + "/api/tags", timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return [m.get("name", "") for m in data.get("models", [])]
        except (urllib.error.URLError, ValueError, OSError) as e:
            raise OllamaError(f"cannot reach Ollama at {self.host}: {e}") from e

    def chat(self, messages: list[dict], tools: Optional[list[dict]] = None,
             on_token: Optional[Callable[[str], None]] = None,
             on_thinking: Optional[Callable[[str], None]] = None) -> dict:
        """Returns {'role': 'assistant', 'content': str, 'tool_calls': list}."""
        body = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": self._options(),
        }
        if tools:
            body["tools"] = tools

        content_parts: list[str] = []
        tool_calls: list[dict] = []
        with self._post("/api/chat", body) as resp:
            for line in resp:
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError:
                    continue
                if chunk.get("error"):
                    raise OllamaError(chunk["error"])
                msg = chunk.get("message") or {}
                if msg.get("thinking") and on_thinking:
                    on_thinking(msg["thinking"])
                piece = msg.get("content") or ""
                if piece:
                    content_parts.append(piece)
                    if on_token:
                        on_token(piece)
                for tc in msg.get("tool_calls") or []:
                    tool_calls.append(tc)
                if chunk.get("done"):
                    try:
                        perf.record(self.host, self.model, chunk)
                    except Exception:
                        pass
                    break

        content = "".join(content_parts)
        if not tool_calls:
            fallback = parse_text_tool_call(content)
            if fallback:
                tool_calls = [fallback]
        return {"role": "assistant", "content": content, "tool_calls": tool_calls}


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def parse_text_tool_call(content: str) -> Optional[dict]:
    """Fallback for models that print a tool call as JSON text instead of
    using structured tool_calls. Looks for {"name": ..., "arguments": {...}}."""
    text = strip_thinking(content)
    # find candidate JSON objects in code fences first, then raw
    candidates = re.findall(r"```(?:json|tool)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not candidates:
        m = re.search(r"\{[^{}]*\"(?:name|tool)\"\s*:.*\}", text, re.DOTALL)
        if m:
            candidates = [m.group(0)]
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("tool")
        args = obj.get("arguments") or obj.get("args") or obj.get("parameters")
        if name and isinstance(args, dict):
            return {"function": {"name": name, "arguments": args}}
    return None
