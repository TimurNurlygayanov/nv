"""Pipeline robustness tests: stray keypresses, plan preservation,
smalltalk bypass, and the self-review/retry loop.

Run with:  python -m unittest tests.test_pipeline -v
No Ollama server needed — agents and the verifier are faked.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nv import cli, selfcheck, ui
from nv.config import Config
from nv.ollama import OllamaError


class FakeAgent:
    """Stands in for nv.agent.Agent: returns scripted replies, records prompts."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.kind = "coder"
        self.messages: list[dict] = []
        self.toolbox = SimpleNamespace(git_diff=lambda stat_only=False: "")

    def run(self, task: str) -> str:
        self.prompts.append(task)
        return self.replies.pop(0)


def make_cfg(**over) -> Config:
    cfg = Config()
    cfg.root = Path(tempfile.mkdtemp())
    cfg.self_check = False
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


class RunPlannedTests(unittest.TestCase):
    def setUp(self):
        ui.CONFIRM.reset()

    def _planned(self, cfg, planner, executor, task="fix the bug", answers=()):
        with mock.patch.object(cli, "Agent", return_value=planner), \
             mock.patch.object(ui, "flush_stdin"), \
             mock.patch("builtins.input", side_effect=list(answers)):
            return cli.run_planned(cfg, executor, task, "")

    def test_stray_enters_then_yes_keeps_plan(self):
        """Accidental Enters at 'approve?' must re-ask, not cancel the plan."""
        planner = FakeAgent(["PLAN:\n1. edit foo.py — fix the bug"])
        executor = FakeAgent(["done"])
        out = self._planned(make_cfg(), planner, executor,
                            answers=["", "", "y"])
        self.assertEqual(out, "done")
        self.assertEqual(len(executor.prompts), 1)
        self.assertIn("PLAN:", executor.prompts[0])
        self.assertIn("TASK: fix the bug", executor.prompts[0])

    def test_prose_revision_keeps_previous_plan(self):
        """A question typed as feedback must not overwrite the plan."""
        planner = FakeAgent(["PLAN:\n1. edit foo.py — fix the bug",
                             "Because the bug is in foo.py."])  # prose answer
        executor = FakeAgent(["done"])
        self._planned(make_cfg(), planner, executor,
                      answers=["why foo.py?", "y"])
        self.assertEqual(len(planner.prompts), 2)  # task + revise
        self.assertIn("PLAN:\n1. edit foo.py", executor.prompts[0])

    def test_prose_instead_of_plan_executes_directly(self):
        """Architect chatter without PLAN: must never reach 'approve?'."""
        planner = FakeAgent(["Hello! I'm ready to help you."])
        executor = FakeAgent(["answered"])
        no_input = mock.Mock(side_effect=AssertionError("approve? was shown"))
        with mock.patch.object(cli, "Agent", return_value=planner), \
             mock.patch("builtins.input", no_input):
            out = cli.run_planned(make_cfg(), executor, "hello there friend", "")
        self.assertEqual(out, "answered")
        self.assertEqual(executor.prompts, ["hello there friend"])

    def test_plan_selfcheck_retry_replans_with_original_task(self):
        """A rejected plan is retried; the corrected plan is what executes."""
        planner = FakeAgent(["PLAN:\n1. delete everything — wrong.py",
                             "PLAN:\n1. edit right.py — actual fix"])
        executor = FakeAgent(["done"])
        cfg = make_cfg(self_check=True, self_check_retries=2)
        with mock.patch.object(cli.selfcheck, "check",
                               side_effect=["plan solves a different task", ""]) as chk:
            self._planned(cfg, planner, executor, answers=["y"])
        self.assertEqual(chk.call_count, 2)
        self.assertIn("ORIGINAL TASK: fix the bug", planner.prompts[1])
        self.assertIn("right.py", executor.prompts[0])


class RunTaskTests(unittest.TestCase):
    def setUp(self):
        ui.CONFIRM.reset()

    def test_smalltalk_skips_checkpoint_and_planning(self):
        agent = FakeAgent(["Hi! How can I help?"])
        with mock.patch.object(cli.checkpoint, "create",
                               side_effect=AssertionError("checkpoint for chat")), \
             mock.patch.object(cli, "run_planned",
                               side_effect=AssertionError("planned chat")):
            cli._run_task(make_cfg(), agent, "hello", "")
        self.assertEqual(agent.prompts, ["hello"])

    def test_result_selfcheck_retries_with_original_task(self):
        """An off-target result is sent back to the agent with the task."""
        agent = FakeAgent(["I refactored bar.py instead.", "fixed foo.py"])
        agent.kind = "ask"  # skip git checkpoint machinery in this test
        cfg = make_cfg(self_check=True, self_check_retries=2, plan_first=False)
        with mock.patch.object(cli.selfcheck, "check",
                               side_effect=["ignores the task", ""]):
            cli._run_task(cfg, agent, "fix the crash in foo.py", "")
        self.assertEqual(len(agent.prompts), 2)
        self.assertIn("SELF-REVIEW rejected", agent.prompts[1])
        self.assertIn("ORIGINAL TASK: fix the crash in foo.py", agent.prompts[1])

    def test_transport_errors_are_not_retried(self):
        agent = FakeAgent(["ERROR: connection refused"])
        agent.kind = "ask"
        cfg = make_cfg(self_check=True, plan_first=False)
        with mock.patch.object(cli.selfcheck, "check",
                               side_effect=AssertionError("checked an ERROR")):
            cli._run_task(cfg, agent, "fix the crash", "")
        self.assertEqual(len(agent.prompts), 1)


class SelfCheckTests(unittest.TestCase):
    def _verdict(self, content):
        client = mock.Mock()
        client.chat.return_value = {"content": content}
        cfg = make_cfg(self_check=True)
        with mock.patch.object(selfcheck, "OllamaClient", return_value=client):
            return selfcheck.check(cfg, "fix foo", "PLAN:\n1. edit foo", "plan")

    def test_ok_passes(self):
        self.assertEqual(self._verdict("OK"), "")

    def test_retry_returns_reason(self):
        self.assertEqual(self._verdict("RETRY: plan ignores the task"),
                         "plan ignores the task")

    def test_rambling_verdict_fails_open(self):
        self.assertEqual(self._verdict("Well, the plan looks reasonable..."), "")

    def test_verifier_error_fails_open(self):
        client = mock.Mock()
        client.chat.side_effect = OllamaError("server down")
        cfg = make_cfg(self_check=True)
        with mock.patch.object(selfcheck, "OllamaClient", return_value=client):
            self.assertEqual(
                selfcheck.check(cfg, "fix foo", "PLAN: 1. x", "plan"), "")

    def test_disabled_check_is_noop(self):
        with mock.patch.object(selfcheck, "OllamaClient",
                               side_effect=AssertionError("client built")):
            self.assertEqual(
                selfcheck.check(make_cfg(), "fix foo", "whatever", "plan"), "")


class CommandDetectionTests(unittest.TestCase):
    def test_natural_language_never_auto_runs(self):
        from nv import terminal
        for line in ("make coverage report", "go faster", "find duplicate models",
                     "cat and summarize logs", "git rewrite everything"):
            self.assertFalse(terminal.looks_like_command(line), line)

    def test_real_commands_still_detected(self):
        from nv import terminal
        for line in ("ls", "ls -la", "git status", "git log --oneline",
                     "pwd", "cd ..", "cd partner-tests", "cat nv/cli.py",
                     "pip freeze", "git checkout master", "git pull",
                     "git commit -m fix", "npm install", "npm run build",
                     "make test", "docker compose up -d"):
            self.assertTrue(terminal.looks_like_command(line), line)

    def test_bare_arg_naming_existing_path_is_a_command(self):
        import tempfile as tf
        from nv import terminal
        root = Path(tf.mkdtemp())
        (root / "README").write_text("x", encoding="utf-8")
        (root / "src").mkdir()
        self.assertTrue(terminal.looks_like_command("cat README", str(root)))
        self.assertTrue(terminal.looks_like_command("ls src", str(root)))
        self.assertFalse(
            terminal.looks_like_command("cat something-imaginary", str(root)))


class FileWriteTests(unittest.TestCase):
    def _toolbox(self, root):
        from nv.tools import Toolbox
        cfg = make_cfg()
        cfg.root = root
        tb = Toolbox(cfg)
        return tb

    def test_edit_preserves_crlf_and_is_approved(self):
        import tempfile as tf
        root = Path(tf.mkdtemp())
        f = root / "a.py"
        f.write_bytes(b"line1\r\nline2\r\nline3\r\n")
        tb = self._toolbox(root)
        with mock.patch.object(ui.CONFIRM, "ask", return_value=(True, "")):
            out = tb.edit_file("a.py", "line2", "changed2")
        self.assertIn("edited", out)
        self.assertEqual(f.read_bytes(), b"line1\r\nchanged2\r\nline3\r\n")

    def test_edit_detects_concurrent_change(self):
        import tempfile as tf
        root = Path(tf.mkdtemp())
        f = root / "a.txt"
        f.write_text("one\ntwo\n", encoding="utf-8")

        def approve_and_race(*a, **k):
            f.write_text("one\nTWO-changed-outside\n", encoding="utf-8")
            return True, ""

        tb = self._toolbox(root)
        with mock.patch.object(ui.CONFIRM, "ask", side_effect=approve_and_race):
            out = tb.edit_file("a.txt", "two", "three")
        self.assertIn("changed on disk", out)
        self.assertIn("TWO-changed-outside", f.read_text(encoding="utf-8"))


class CompletionTests(unittest.TestCase):
    def setUp(self):
        import tempfile as tf
        self.root = Path(tf.mkdtemp())
        (self.root / "partner-tests").mkdir()
        (self.root / "payments").mkdir()
        (self.root / "README.md").write_text("x", encoding="utf-8")
        (self.root / "partner-tests" / "conftest.py").write_text(
            "x", encoding="utf-8")

    def test_first_word_completes_commands(self):
        cmds = ["/help", "/host", "git", "grep"]
        self.assertEqual(ui.completion_matches("/h", 0, str(self.root), cmds),
                         ["/help", "/host"])
        self.assertEqual(ui.completion_matches("gr", 0, str(self.root), cmds),
                         ["grep"])

    def test_argument_completes_paths_with_dir_slash(self):
        got = ui.completion_matches("pa", 3, str(self.root), [])
        self.assertEqual(got, ["partner-tests/", "payments/"])
        got = ui.completion_matches("RE", 3, str(self.root), [])
        self.assertEqual(got, ["README.md"])

    def test_argument_completes_inside_subdir(self):
        got = ui.completion_matches("partner-tests/con", 3, str(self.root), [])
        self.assertEqual(got, ["partner-tests/conftest.py"])

    def test_unreadable_folder_is_silent(self):
        self.assertEqual(
            ui.completion_matches("nope/xx", 3, str(self.root), []), [])


class DiscoverTests(unittest.TestCase):
    def test_normalize_malformed_hosts(self):
        from nv import discover
        self.assertEqual(discover._normalize("http:11434"), "")
        self.assertEqual(discover._normalize("http://"), "")
        self.assertEqual(discover._normalize("192.168.1.5"),
                         "http://192.168.1.5:11434")
        self.assertEqual(discover._normalize("http://x:1234"), "http://x:1234")


if __name__ == "__main__":
    unittest.main()
