"""nv console: chat REPL with agent mode, team mode and git-diff shortcut."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from nv import (__version__, checkpoint, discover, ingest, selfcheck, session,
                terminal, theme, ui)
from nv.agent import Agent
from nv.config import Config, load_agents_md, load_config, save_global
from nv.ollama import OllamaClient, OllamaError
from nv.prompts import AGENT_CONFIGS
from nv.team import run_team

HELP = """
terminal passthrough (no model involved, output lands in the chat context):
  !<command>       run a shell command right here, e.g. !git log --oneline
  !!<description>  the model builds the command, you confirm, it runs
  bare commands (`ls`, `git status`, `env`) are auto-detected and run too;
  if a command fails, nv offers to let the model fix it

commands:
  /ask <question>  quick read-only answer (no changes possible, no prompts)
  /paste [question]  paste multi-line text (stack trace, log); finish with END
  /load <file-or-folder> [question]  feed files to the agent: relevant files
                   are auto-selected, big inputs summarized chunk by chunk,
                   long jobs estimated and confirmed before starting
  /diff            show current git diff (colored)
  /diff stat       show git diff --stat summary
  /undo            revert the working tree to before the last task
  /minimize        run the minimizer agent: shrink the current diff without
                   changing behavior (auto-offered after oversized tasks)
  /resume          continue the conversation from the previous session
  /note <fact>     save a fact to project notes; /note alone shows them
  /theme <words>   restyle the console ("green text on black, bigger font");
                   /theme shows the current theme, /theme reset restores it
  /style <words>   answer personality, e.g. /style playful and lightly flirty
                   /style shows the current one, /style off = neutral
  /plan on|off     toggle the planning step before coding (default: on)
  /plan <task>     force plan -> approve -> execute for this task
  /team N <task>   run N agents in parallel on the task + separate review
  /review          run the reviewer agent on the current git diff
  /agents          list built-in agent configs
  /agent <name>    switch the chat agent (coder / writer / reviewer)
  /scan            check OLLAMA_HOST / localhost for an Ollama server
  /model <name>    set the model (saved)
  /host <url>      set the Ollama host, e.g. /host http://192.168.1.50:11434
  /config          show current config
  /new             reset conversation history
  /help            this help
  /exit            quit

anything else is sent to the agent. It will ask confirmation before every
file change and every command.
"""


def _print_config(cfg: Config) -> None:
    ui.out(f"  host: {cfg.host}")
    ui.out(f"  model: {cfg.model}   reviewer model: {cfg.reviewer_model}")
    ui.out(f"  num_ctx: {cfg.num_ctx}   max_steps: {cfg.max_steps}   "
           f"plan_first: {'on' if cfg.plan_first else 'off'}")
    ui.out(f"  max_answer_tokens: {cfg.max_answer_tokens}   "
           f"max_diff_lines: {cfg.max_diff_lines}   "
           f"style: {cfg.personality or 'neutral'}")
    ui.out(f"  root (sandbox): {cfg.root}")


def _pick(prompt: str, options: list[str], default: int = 0) -> str:
    if len(options) == 1:
        return options[0]
    ui.out(prompt, ui.BOLD)
    for i, opt in enumerate(options, 1):
        marker = " (default)" if i - 1 == default else ""
        ui.out(f"  {i}. {opt}{marker}")
    try:
        answer = input("  choose number: ").strip()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if answer.isdigit() and 1 <= int(answer) <= len(options):
        return options[int(answer) - 1]
    return options[default]


def _preferred_model(models: list[str], current: str) -> int:
    """Index of the best default: keep current if present, else a coder model."""
    if current in models:
        return models.index(current)
    for i, m in enumerate(models):
        if "coder" in m.lower():
            return i
    return 0


def run_discovery(cfg: Config) -> bool:
    """Scan for Ollama servers, let the user pick host + model, save config."""
    found = discover.scan(extra_hosts=[cfg.host], on_status=ui.info)
    if not found:
        ui.warn("no Ollama server found (checked OLLAMA_HOST, the saved host "
                "and localhost — nv does not scan the network).\n"
                "for a remote desktop: set OLLAMA_HOST=0.0.0.0 there, restart "
                "Ollama, then here run  /host http://<desktop-ip>:11434  or "
                "set OLLAMA_HOST=http://<desktop-ip>:11434")
        return False
    for url, models in found:
        ui.out(f"  found {url} — models: {', '.join(models) or '(none)'}", ui.GREEN)
    hosts_with_models = [f for f in found if f[1]] or found
    host = _pick("select Ollama server:", [f[0] for f in hosts_with_models])
    models = dict(found)[host]
    if models:
        model = _pick("select model:", models, _preferred_model(models, cfg.model))
        cfg.model = model
    cfg.host = host
    save_global(cfg)
    ui.info(f"saved: host={cfg.host} model={cfg.model}")
    return True


def _check_connection(cfg: Config, offer_scan: bool = True) -> None:
    try:
        models = OllamaClient(cfg.host, cfg.model).ping()
        ui.info(f"connected to {cfg.host} — models: {', '.join(models) or '(none)'}")
        if models and cfg.model not in models:
            ui.warn(f"model '{cfg.model}' is not on this server")
            model = _pick("select model:", models,
                          _preferred_model(models, cfg.model))
            cfg.model = model
            save_global(cfg)
            ui.info(f"model set to {cfg.model} (saved)")
        return
    except OllamaError as e:
        ui.warn(str(e))
    if not offer_scan:
        return
    try:
        answer = input(ui.prompt_str(
            f"{ui.BOLD}look for Ollama on OLLAMA_HOST / localhost? "
            f"[Y/n] {ui.RESET}")).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return
    if answer in ("", "y", "yes", "д", "да"):
        run_discovery(cfg)


def run_planned(cfg: Config, executor: Agent, task: str, agents_md: str) -> str:
    """Pipeline: architect makes a plan -> user approves/revises -> executor
    follows the approved plan step by step."""
    ui.banner("planning")
    planner = Agent(cfg, "architect", name="architect", agents_md=agents_md)
    plan = planner.run(task)
    for _ in range(max(0, cfg.self_check_retries)):
        if "NO_PLAN" in plan[:200] or "PLAN:" not in plan:
            break
        reason = selfcheck.check(cfg, task, plan, "plan")
        if not reason:
            break
        ui.warn(f"self-review: {reason} — replanning")
        plan = planner.run(
            "REVIEW REJECTED your plan: " + reason + ". Re-read the ORIGINAL "
            "TASK below and output a corrected full PLAN (or NO_PLAN).\n"
            "ORIGINAL TASK: " + task)
    if "NO_PLAN" in plan[:200] or "PLAN:" not in plan:
        # either an explicit NO_PLAN or the model answered in prose
        # (greeting, question, ...) — never offer that for approval
        ui.info("no plan needed — executing directly")
        return executor.run(task)
    for _ in range(5):
        ok, feedback = ui.CONFIRM.ask("execute this plan?")
        if ok:
            return executor.run(
                "Execute this task following the approved plan below, step by "
                "step and in order. Say which step number you are on. Do not "
                "add work beyond the plan. If a step turns out to be wrong or "
                "impossible, stop and explain instead of improvising. You have "
                "full write access: make every change yourself with your "
                "tools. The plan was written by a read-only planning agent — "
                "if it says changes cannot be made or must be applied "
                "manually by the user, ignore that and implement them.\n\n"
                f"TASK: {task}\n\n{plan}")
        if not feedback:
            ui.warn("plan cancelled")
            return ""
        ui.banner("revising plan")
        revised = planner.run(
            "Revise the plan based on this user feedback (output the full "
            "updated PLAN again, or NO_PLAN if nothing is needed): " + feedback)
        if "NO_PLAN" in revised[:200]:
            ui.warn("plan cancelled")
            return ""
        if "PLAN:" in revised:
            plan = revised
        else:
            ui.info("(answer only — the previous plan is unchanged)")
    ui.warn("too many plan revisions — cancelled")
    return ""


def _read_multiline() -> str:
    ui.info("paste your text; finish with a line containing only END "
            "(or Ctrl+Z + Enter)")
    lines: list[str] = []
    while True:
        try:
            row = input()
        except (EOFError, KeyboardInterrupt):
            break
        if row.strip() == "END":
            break
        lines.append(row)
    return "\n".join(lines).strip()


def run_minimizer(cfg: Config, agents_md: str, task: str = "") -> None:
    minimizer = Agent(cfg, "minimizer", agents_md=agents_md)
    try:
        summary = minimizer.run(
            "Minimize the current git diff: remove everything not strictly "
            "needed, keep behavior identical."
            + (f"\nThe diff came from this task: {task}" if task else "")
            + "\nStart with the git_diff tool.")
        ui.banner("minimizer")
        ui.out(summary)
    except KeyboardInterrupt:
        ui.warn("\nminimizer interrupted")


# greetings / acknowledgements: no checkpoint, no planning — just chat
_SMALLTALK = re.compile(
    r"^(hi|hello|hey|yo|good\s+(morning|afternoon|evening)|thanks|thank\s+you|"
    r"ok|okay|привет|здравствуй(те)?|добр(ый|ое)\s+(утро|день|вечер)|спасибо|ок)"
    r"[\s!.?)]*$", re.IGNORECASE)


def _run_task(cfg: Config, agent: Agent, task: str, agents_md: str,
              planned: bool = True) -> None:
    """One chat turn: checkpoint -> (plan ->) run -> minimize-offer ->
    persist history. Ctrl+C aborts the turn, not the REPL."""
    ui.CONFIRM.reset()
    chat = bool(_SMALLTALK.match(task.strip()))
    cp = None
    if not chat and agent.kind in ("coder", "writer"):
        cp = checkpoint.create(cfg.root)
        if cp:
            ui.info("checkpoint saved (/undo reverts this task)")
    try:
        if planned and not chat and cfg.plan_first and agent.kind == "coder":
            result = run_planned(cfg, agent, task, agents_md)
        else:
            result = agent.run(task)
        # self-review: does the final answer actually serve the original
        # prompt? Skip for chat, cancelled plans and transport errors.
        if result and not chat and not result.startswith(("ERROR:", "stopped:")):
            for _ in range(max(0, cfg.self_check_retries)):
                reason = selfcheck.check(
                    cfg, task, result, "result",
                    diff_stat=agent.toolbox.git_diff(stat_only=True))
                if not reason:
                    break
                ui.warn(f"self-review: {reason} — asking the agent to fix it")
                result = agent.run(
                    "SELF-REVIEW rejected your previous answer: " + reason +
                    ". Re-read the ORIGINAL TASK below, check the actual state "
                    "of the files, and produce a correct result.\n"
                    "ORIGINAL TASK: " + task)
        if cp:
            grown = checkpoint.changed_lines_since(cfg.root, cp)
            if grown > cfg.max_diff_lines:
                ok, _ = ui.CONFIRM.ask(
                    f"the task changed {grown} lines (limit "
                    f"{cfg.max_diff_lines}) — run the minimizer agent to "
                    "shrink the diff?", force=True)
                if ok:
                    run_minimizer(cfg, agents_md, task)
    except KeyboardInterrupt:
        ui.warn("\ninterrupted — back to the prompt (history kept, "
                "/undo reverts applied changes)")
    session.save_history(cfg.root, agent.kind, agent.messages)


def _handle_load(cfg: Config, agent: Agent, path_str: str, question: str,
                 agents_md: str) -> None:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = cfg.root / p
    try:
        res = ingest.interactive_pipeline(cfg, p, question)
    except KeyboardInterrupt:
        ui.warn("\nload interrupted")
        return
    except OllamaError as e:
        ui.error(str(e))
        return
    if res is None:
        return
    text, summarized, source = res
    _run_task(cfg, agent, _input_task(text, source, question, summarized),
              agents_md, planned=False)


def _split_path_arg(rest: str) -> tuple[str, str]:
    """Split '<path> [question]' honoring quotes around paths with spaces."""
    rest = rest.strip()
    if rest[:1] in ("'", '"'):
        end = rest.find(rest[0], 1)
        if end > 0:
            return rest[1:end], rest[end + 1:].strip()
    parts = rest.split(maxsplit=1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def _input_task(text: str, source: str, question: str, summarized: bool) -> str:
    q = question or ("Analyze this input: explain the key errors/failures "
                     "and their most likely causes.")
    label = "summarized digest" if summarized else "content"
    # fence longer than any backtick run inside, so content can't escape it
    run = max((len(m) for m in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, run + 1)
    return f"{q}\n\nINPUT {label} from {source}:\n{fence}\n{text}\n{fence}"


def repl(cfg: Config) -> None:
    agents_md = load_agents_md(cfg.root)
    if cfg.theme:
        theme.apply(cfg.theme, quiet=True)
    ui.out(f"{ui.BOLD}{ui.CYAN}nv v{__version__}{ui.RESET} — local LLM agent console")
    ui.info(f"sandbox: {cfg.root}")
    if agents_md:
        ui.info("loaded project rules from AGENTS.md")
    notes = session.load_notes(cfg.root)
    if notes:
        ui.info(f"loaded project notes ({notes.count(chr(10)) + 1} lines)")
    _check_connection(cfg)
    prev = session.load_history(cfg.root)
    if prev:
        ui.info(f"previous session found ({prev.get('time', '?')}, "
                f"{len(prev['messages'])} messages) — /resume to continue")
    ui.info("type /help for commands\n")

    agent = Agent(cfg, "coder", agents_md=agents_md)
    ask_agent: Agent | None = None
    term_buf = terminal.TerminalBuffer()
    ui.init_history(session.state_dir(cfg.root) / "input-history")

    def run_shell(cmd: str) -> None:
        rc, out = terminal.run(cfg, cmd, term_buf)
        if rc in (0, 130):
            return
        try:
            answer = input(ui.prompt_str(
                f"{ui.BOLD}fix this command with the model? "
                f"[y/N] {ui.RESET}")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        if answer in ("y", "yes", "д", "да"):
            fixed = terminal.synthesize(
                cfg, f"This command failed, fix it: {cmd}",
                error_ctx=out[-1500:])
            if fixed:
                terminal.run(cfg, fixed, term_buf)

    while True:
        try:
            ui.flush_stdin()
            loc = terminal.cwd_label(cfg, term_buf)
            line = input(ui.prompt_str(
                f"{ui.c('prompt')}nv{' ' + loc if loc else ''}>"
                f"{ui.RESET} ")).strip()
        except EOFError:
            ui.out("\nbye")
            return
        except KeyboardInterrupt:
            ui.out()
            ui.info("(use /exit to quit)")
            continue
        if not line:
            continue

        if line.startswith("!!"):
            request = line[2:].strip()
            if not request:
                ui.warn("usage: !!<what the command should do>")
                continue
            cmd = terminal.synthesize(cfg, request)
            if cmd:
                run_shell(cmd)
            continue
        if line.startswith("!"):
            cmd = line[1:].strip()
            if cmd:
                run_shell(cmd)
            continue
        if terminal.looks_like_command(line):
            ui.info(f"[shell] {line}")
            run_shell(line)
            continue

        if line.startswith("/"):
            parts = line.split(maxsplit=2)
            cmd = parts[0].lower()

            if cmd in ("/exit", "/quit", "/q"):
                return
            if cmd == "/help":
                ui.out(HELP)
            elif cmd == "/diff":
                stat = len(parts) > 1 and parts[1] == "stat"
                diff = agent.toolbox.git_diff(stat_only=stat)
                ui.colorize_git_diff(diff)
            elif cmd == "/ask":
                if len(parts) < 2:
                    ui.warn("usage: /ask <question>")
                    continue
                question = term_buf.wrap(line.split(maxsplit=1)[1])
                if ask_agent is None:
                    ask_agent = Agent(cfg, "ask", agents_md=agents_md)
                try:
                    ask_agent.run(question)
                except KeyboardInterrupt:
                    ui.warn("\ninterrupted")
            elif cmd == "/paste":
                question = line.split(maxsplit=1)[1] if len(parts) > 1 else ""
                text = _read_multiline()
                if not text:
                    ui.warn("nothing pasted")
                    continue
                _run_task(cfg, agent,
                          _input_task(text, "pasted text", question, False),
                          agents_md, planned=False)
            elif cmd == "/load":
                if len(parts) < 2:
                    ui.warn("usage: /load <file-or-folder> [question] "
                            "(quote paths with spaces)")
                    continue
                lpath, lquestion = _split_path_arg(line.split(maxsplit=1)[1])
                _handle_load(cfg, agent, lpath, lquestion, agents_md)
            elif cmd == "/minimize":
                ui.CONFIRM.reset()
                run_minimizer(cfg, agents_md)
            elif cmd == "/undo":
                cp = checkpoint.load_last(cfg.root)
                if not cp:
                    ui.warn("no checkpoint found (one is saved before each task)")
                    continue
                stat = checkpoint.diff_stat(cfg.root, cp)
                if not stat:
                    ui.info("nothing to undo — working tree matches the checkpoint")
                    continue
                ui.banner("changes since the checkpoint")
                ui.out(stat)
                ok, _ = ui.CONFIRM.ask("revert the working tree to the checkpoint?")
                if ok:
                    ui.info(checkpoint.restore(cfg.root, cp))
            elif cmd == "/resume":
                prev = session.load_history(cfg.root)
                if not prev:
                    ui.warn("no previous session found")
                    continue
                kind = prev.get("agent", "coder")
                if kind in AGENT_CONFIGS and kind != agent.kind:
                    agent = Agent(cfg, kind, agents_md=agents_md)
                agent.reset()
                agent.messages.extend(prev["messages"])
                ui.info(f"restored {len(prev['messages'])} messages from "
                        f"{prev.get('time', '?')} (agent: {agent.kind})")
            elif cmd == "/theme":
                arg = line.split(maxsplit=1)[1].strip() if len(parts) > 1 else ""
                if not arg:
                    ui.out(theme.describe(cfg.theme))
                elif arg.lower() == "reset":
                    theme.reset()
                    cfg.theme = {}
                    save_global(cfg)
                else:
                    try:
                        patch = theme.from_prompt(cfg, arg)
                    except KeyboardInterrupt:
                        ui.warn("\ninterrupted")
                        continue
                    if patch:
                        theme.apply(patch)
                        cfg.theme.update(patch)
                        save_global(cfg)
                        ui.info("theme saved (/theme reset restores defaults)")
            elif cmd == "/style":
                arg = line.split(maxsplit=1)[1].strip() if len(parts) > 1 else ""
                if not arg:
                    ui.out(cfg.personality or
                           "(neutral — set one with /style <description>)")
                else:
                    cfg.personality = "" if arg.lower() in ("off", "none",
                                                            "reset") else arg
                    save_global(cfg)
                    ui.info(f"answer style: {cfg.personality or 'neutral'} "
                            "(saved, applies from the next message)")
            elif cmd == "/note":
                if len(parts) > 1:
                    session.add_note(cfg.root, line.split(maxsplit=1)[1])
                    ui.info("note saved to project notes (injected into "
                            "agents from their next start)")
                else:
                    notes = session.load_notes(cfg.root)
                    ui.out(notes or "(no notes yet — /note <fact> to add one)")
            elif cmd == "/plan":
                arg = line.split(maxsplit=1)[1] if len(line.split(maxsplit=1)) > 1 else ""
                if arg.lower() in ("on", "off"):
                    cfg.plan_first = arg.lower() == "on"
                    save_global(cfg)
                    ui.info(f"planning step is now {'ON' if cfg.plan_first else 'OFF'} (saved)")
                elif arg:
                    if agent.kind != "coder":
                        agent = Agent(cfg, "coder", agents_md=agents_md)
                    ui.CONFIRM.reset()
                    if checkpoint.create(cfg.root):
                        ui.info("checkpoint saved (/undo reverts this task)")
                    try:
                        run_planned(cfg, agent, arg, agents_md)
                    except KeyboardInterrupt:
                        ui.warn("\ninterrupted — back to the prompt")
                    session.save_history(cfg.root, agent.kind, agent.messages)
                else:
                    ui.warn("usage: /plan on | /plan off | /plan <task>")
            elif cmd == "/team":
                if len(parts) < 3 or not parts[1].isdigit():
                    ui.warn("usage: /team 3 <task description>")
                    continue
                ui.CONFIRM.reset()
                if checkpoint.create(cfg.root):
                    ui.info("checkpoint saved (/undo reverts this team run)")
                try:
                    run_team(cfg, parts[2], int(parts[1]), agents_md)
                except KeyboardInterrupt:
                    ui.warn("\nteam run interrupted")
            elif cmd == "/review":
                ui.CONFIRM.reset()
                reviewer = Agent(cfg, "reviewer", agents_md=agents_md,
                                 model=cfg.reviewer_model)
                try:
                    report = reviewer.run("Review the current git diff. "
                                          "Start with the git_diff tool.")
                    ui.banner("review report")
                    ui.out(report)
                except KeyboardInterrupt:
                    ui.warn("\nreview interrupted")
            elif cmd == "/agents":
                for name, spec in AGENT_CONFIGS.items():
                    ui.out(f"  {name:<10} — {spec['description']}")
            elif cmd == "/agent":
                name = parts[1] if len(parts) > 1 else ""
                if name not in AGENT_CONFIGS:
                    ui.warn(f"unknown agent, choose from: {', '.join(AGENT_CONFIGS)}")
                    continue
                agent = Agent(cfg, name, agents_md=agents_md)
                ui.info(f"switched to agent '{name}' (history reset)")
            elif cmd == "/scan":
                try:
                    if run_discovery(cfg):
                        agent = Agent(cfg, agent.kind, agents_md=agents_md)
                except KeyboardInterrupt:
                    ui.warn("\nscan interrupted")
            elif cmd == "/model":
                if len(parts) < 2:
                    ui.warn("usage: /model qwen3-coder:30b")
                    continue
                cfg.model = parts[1]
                save_global(cfg)
                agent = Agent(cfg, agent.kind, agents_md=agents_md)
                ui.info(f"model set to {cfg.model} (saved)")
            elif cmd == "/host":
                if len(parts) < 2:
                    ui.warn("usage: /host http://192.168.1.50:11434")
                    continue
                cfg.host = parts[1].rstrip("/")
                save_global(cfg)
                agent = Agent(cfg, agent.kind, agents_md=agents_md)
                _check_connection(cfg)
                ui.info("host saved")
            elif cmd == "/config":
                _print_config(cfg)
            elif cmd == "/new":
                agent.reset()
                ui.CONFIRM.reset()
                ui.info("conversation reset")
            else:
                ui.warn(f"unknown command {cmd}, try /help")
            continue

        # normal prompt -> (plan ->) agent iterates until done
        _run_task(cfg, agent, term_buf.wrap(line), agents_md)


def main() -> None:
    # stream encoding is configured in nv.ui at import time (utf-8 with
    # errors="replace") — do not reconfigure here, it would reset the
    # error handler to strict and crash on stray surrogates mid-stream
    parser = argparse.ArgumentParser(
        prog="nv", description="local LLM agent console (Ollama)")
    parser.add_argument("prompt", nargs="*", help="one-shot task (omit for REPL)")
    parser.add_argument("--host", help="Ollama host, e.g. http://192.168.1.50:11434")
    parser.add_argument("--model", help="model name, e.g. qwen3-coder:30b")
    parser.add_argument("--ctx", type=int, help="context window (num_ctx)")
    parser.add_argument("--team", type=int, metavar="N",
                        help="run the one-shot prompt with N parallel agents + review")
    parser.add_argument("--no-plan", action="store_true",
                        help="skip the planning step for this run")
    parser.add_argument("--scan", action="store_true",
                        help="check OLLAMA_HOST / localhost for Ollama and exit")
    parser.add_argument("--version", action="version", version=f"nv {__version__}")
    args = parser.parse_args()

    cfg = load_config(Path.cwd())
    # CLI flags are per-run overrides; only /host and /model persist
    if args.host:
        cfg.host = args.host.rstrip("/")
    if args.model:
        cfg.model = args.model
    if args.ctx:
        cfg.num_ctx = args.ctx

    if args.scan:
        run_discovery(cfg)
        return

    if args.prompt:
        task = " ".join(args.prompt)
        agents_md = load_agents_md(cfg.root)
        if args.team:
            try:
                run_team(cfg, task, args.team, agents_md)
            except KeyboardInterrupt:
                ui.warn("\ninterrupted")
        else:
            # same pipeline as the REPL: checkpoint, plan, self-review,
            # history — so /undo and /resume work for one-shot runs too
            executor = Agent(cfg, "coder", agents_md=agents_md)
            _run_task(cfg, executor, task, agents_md,
                      planned=not args.no_plan)
        return

    repl(cfg)


if __name__ == "__main__":
    main()
