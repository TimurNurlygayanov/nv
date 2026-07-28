"""End-to-end pipeline test: a long fix-review-feedback loop.

Simulates the real daily flow against a scripted fake Ollama transport:
the user shares a failing-test issue, the architect plans, the user
approves, the executor edits a real file via tool calls, self-review
rejects once and the agent fixes it, then the user sends many rounds of
feedback under a deliberately tiny history budget.

The core assertion: after all of that, the ORIGINAL issue text is still
present in the messages sent to the model — the pipeline must not forget
what it was fixing, no matter how long the loop runs.

Run with:  python -m unittest tests.test_feedback_loop -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nv import agent as agent_mod
from nv import cli, ui
from nv.agent import Agent
from nv.config import Config

ISSUE = (
    "Fix ISSUE-12345: payment tests fail when discount is 100%.\n"
    "FAILED tests/test_payment.py::test_full_discount\n"
    "Traceback (most recent call last):\n"
    '  File "payment_calc.py", line 2, in calc\n'
    "    return price / (100 - discount)\n"
    "ZeroDivisionError: division by zero"
)

FEEDBACK = [
    "also handle discount above 100, should return 0 too",
    "please add a comment explaining why we clamp",
    "actually rename the guard variable to clamped",
    "run through the logic once more, I think rounding is off",
    "revert the rename, keep the comment",
    "make the comment shorter",
    "ok now explain what the final version does",
]


def say(text: str) -> dict:
    return {"role": "assistant", "content": text, "tool_calls": []}


def tool(name: str, **args) -> dict:
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": args}}]}


class FakeOllama:
    """Scripted stand-in for OllamaClient: pops pre-planned replies and
    records every full message list the 'model' was shown."""
    replies: list[dict] = []
    calls: list[list[dict]] = []

    def __init__(self, *a, **k):
        pass

    def chat(self, messages, tools=None, on_token=None, on_thinking=None):
        FakeOllama.calls.append([dict(m) for m in messages])
        if not FakeOllama.replies:
            raise AssertionError("script exhausted — unexpected extra model call")
        return FakeOllama.replies.pop(0)


class FeedbackLoopTest(unittest.TestCase):
    def setUp(self):
        FakeOllama.replies = []
        FakeOllama.calls = []
        ui.CONFIRM.reset()

    def test_long_feedback_loop_keeps_original_issue(self):
        cfg = Config()
        cfg.root = Path(tempfile.mkdtemp()).resolve()  # as load_config does
        cfg.plan_first = True
        cfg.self_check = True
        cfg.self_check_retries = 2
        # tight but possible: the ~3k system prompt + user messages fit,
        # the ~6k of assistant chatter does not — compaction must trigger
        cfg.history_char_budget = 7000
        (cfg.root / "payment_calc.py").write_text(
            "def calc(price, discount):\n"
            "    return price / (100 - discount)\n", encoding="utf-8")

        # ---- scripted conversation -----------------------------------
        script = [
            # turn 1: architect plans
            say("PLAN:\n1. edit payment_calc.py — guard the 100% discount case"),
            # executor: reads, edits (real file ops), reports
            tool("read_file", path="payment_calc.py"),
            tool("edit_file", path="payment_calc.py",
                 old_text="    return price / (100 - discount)",
                 new_text="    if discount >= 100:\n"
                          "        return 0\n"
                          "    return price / (100 - discount)"),
            say("fixed: guarded the 100% discount case in payment_calc.py"),
            # self-review rejects once -> executor amends
            say("amended: discount > 100 also returns 0 now"),
        ]
        # turns 2..8: feedback rounds (architect says NO_PLAN, executor
        # answers; padding makes history overflow the tiny budget)
        for i, _ in enumerate(FEEDBACK):
            script.append(say("NO_PLAN"))
            script.append(say(f"round {i + 1} done. " + "details " * 120))
        FakeOllama.replies = script

        # check order: plan check (pass), result check (reject once), result
        # re-check (pass), then one passing result check per feedback round
        selfcheck_verdicts = ["", "fix ignores discounts above 100", ""] + \
                             [""] * len(FEEDBACK)

        with mock.patch.object(agent_mod, "OllamaClient", FakeOllama), \
             mock.patch.object(ui.CONFIRM, "ask", return_value=(True, "")), \
             mock.patch.object(cli.selfcheck, "check",
                               side_effect=selfcheck_verdicts):
            executor = Agent(cfg, "coder", agents_md="")
            cli._run_task(cfg, executor, ISSUE, "")
            for fb in FEEDBACK:
                cli._run_task(cfg, executor, fb, "")

        # ---- the fix actually landed in the file ---------------------
        result = (cfg.root / "payment_calc.py").read_text(encoding="utf-8")
        self.assertIn("if discount >= 100:", result)

        # ---- self-review retry carried the original task -------------
        retry_msgs = [m["content"] for call in FakeOllama.calls for m in call
                      if m.get("role") == "user"
                      and "SELF-REVIEW rejected" in (m.get("content") or "")]
        self.assertTrue(retry_msgs, "self-review retry never reached the model")
        self.assertIn("ISSUE-12345", retry_msgs[0])

        # ---- core assertion: the model NEVER lost the original issue -
        for n, call_msgs in enumerate(FakeOllama.calls, 1):
            # skip architect calls (fresh agents, own short history)
            if not any(m.get("role") == "user" and
                       "ISSUE-12345" in (m.get("content") or "")
                       for m in call_msgs):
                # executor calls must always contain the issue; architect
                # calls after turn 1 legitimately don't (NO_PLAN turns)
                texts = " ".join((m.get("content") or "") for m in call_msgs)
                self.assertIn("nv-architect", call_msgs[0]["content"],
                              f"call {n}: executor lost the original issue "
                              f"(history was compacted away): {texts[:200]}")

        # ---- compaction really happened AND kept every user message --
        final = FakeOllama.calls[-1]
        sent_assistant_texts = {m["content"] for m in final
                                if m.get("role") == "assistant"}
        self.assertLess(
            len(sent_assistant_texts), len(FEEDBACK) + 2,
            "history never compacted — budget was not exercised")
        user_texts = " ".join((m.get("content") or "") for m in final
                              if m.get("role") == "user")
        self.assertIn("ISSUE-12345", user_texts)
        for fb in FEEDBACK[-3:]:  # at minimum the recent feedback survives
            self.assertIn(fb[:30], user_texts)

        # ---- history structure stays sane after compaction -----------
        self.assertEqual(final[0]["role"], "system")
        self.assertNotEqual(final[1].get("role"), "tool",
                            "orphaned tool result at history start")

    def test_compaction_prefers_dropping_assistant_over_user(self):
        cfg = Config()
        cfg.root = Path(tempfile.mkdtemp()).resolve()
        # realistic: bigger than the system prompt, smaller than the chatter
        cfg.history_char_budget = 4200
        with mock.patch.object(agent_mod, "OllamaClient", FakeOllama):
            a = Agent(cfg, "coder", agents_md="")
        a.messages = [a.messages[0]]
        a.messages.append({"role": "user", "content": "THE-ORIGINAL-TASK " * 10})
        for i in range(6):
            a.messages.append({"role": "assistant", "content": f"blah {i} " * 60,
                               "tool_calls": None})
            a.messages.append({"role": "user", "content": f"feedback {i}"})
        a._compact()
        roles = [m["role"] for m in a.messages]
        joined = " ".join(m.get("content") or "" for m in a.messages)
        self.assertIn("THE-ORIGINAL-TASK", joined)
        self.assertIn("feedback 5", joined)
        self.assertLess(roles.count("assistant"), 6)


if __name__ == "__main__":
    unittest.main()
