"""Self-review before showing an answer to the user.

A cheap tool-less verifier compares the ORIGINAL user prompt with what the
model produced (a plan, or the final result) and votes OK / RETRY. Catches
the failure modes of small local models: forgetting the task mid-way,
answering a different question, claiming work is impossible, or reporting
"done" while the diff is empty.

Fail-open by design: if the verifier itself errors or rambles, the pipeline
continues as if the check passed — it must never block the user.
"""
from __future__ import annotations

from nv import ui
from nv.config import Config
from nv.ollama import OllamaClient, OllamaError, strip_thinking

_PLAN_RULES = """You review an implementation PLAN against the USER TASK.
Reply RETRY if the plan: ignores the task or solves a different one, claims
the work cannot be done, plans manual user actions instead of code changes,
or is not actually a plan (a greeting, an apology, a question back)."""

_RESULT_RULES = """You review an agent's final ANSWER against the USER TASK.
Reply RETRY if the answer: ignores or misunderstands the task, answers a
different question, claims changes cannot be made, or reports code changes
while DIFF STAT below is empty. An empty diff is FINE when the task is a
question or discussion that needs no changes."""


def check(cfg: Config, task: str, output: str, stage: str,
          diff_stat: str = "") -> str:
    """Verify output against the original task. Returns '' when the output
    passes (or the check is disabled/broken), else a short retry reason."""
    if not cfg.self_check or not output.strip():
        return ""
    rules = _PLAN_RULES if stage == "plan" else _RESULT_RULES
    client = OllamaClient(cfg.host, cfg.reviewer_model,
                          num_ctx=min(cfg.num_ctx, 8192),
                          temperature=0.0, num_predict=120)
    user = f"USER TASK:\n{task[:2000]}\n\n{stage.upper()}:\n{output[:3000]}"
    if stage != "plan":
        user += f"\n\nDIFF STAT of changes made:\n{diff_stat[:600] or '(empty)'}"
    try:
        reply = client.chat([
            {"role": "system", "content": rules + """
Answer with EXACTLY one line and nothing else:
OK
or
RETRY: <short concrete reason>
Judge only whether the output serves the task. Style, brevity and wording
are NOT reasons to retry. When in doubt, answer OK."""},
            {"role": "user", "content": user},
        ])
    except OllamaError as e:
        ui.info(f"(self-review skipped: {e})")
        return ""
    verdict = strip_thinking(reply.get("content") or "").strip()
    first = verdict.splitlines()[0] if verdict else "OK"
    if first.upper().startswith("RETRY"):
        return first.split(":", 1)[1].strip() if ":" in first else "off-target"
    return ""
