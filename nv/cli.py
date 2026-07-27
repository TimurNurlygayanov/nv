"""nv console: chat REPL with agent mode, team mode and git-diff shortcut."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nv import __version__, discover, ui
from nv.agent import Agent
from nv.config import Config, load_agents_md, load_config, save_global
from nv.ollama import OllamaClient, OllamaError
from nv.prompts import AGENT_CONFIGS
from nv.team import run_team

HELP = """
commands:
  /diff            show current git diff (colored)
  /diff stat       show git diff --stat summary
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
    except EOFError:
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
        answer = input(f"{ui.BOLD}look for Ollama on OLLAMA_HOST / localhost? "
                       f"[Y/n] {ui.RESET}").strip().lower()
    except EOFError:
        return
    if answer in ("", "y", "yes", "д", "да"):
        run_discovery(cfg)


def run_planned(cfg: Config, executor: Agent, task: str, agents_md: str) -> str:
    """Pipeline: architect makes a plan -> user approves/revises -> executor
    follows the approved plan step by step."""
    ui.banner("planning")
    planner = Agent(cfg, "architect", name="architect", agents_md=agents_md)
    plan = planner.run(task)
    if "NO_PLAN" in plan[:200]:
        ui.info("no plan needed — executing directly")
        return executor.run(task)
    for _ in range(5):
        ok, feedback = ui.CONFIRM.ask("execute this plan?")
        if ok:
            return executor.run(
                "Execute this task following the approved plan below, step by "
                "step and in order. Say which step number you are on. Do not "
                "add work beyond the plan. If a step turns out to be wrong or "
                "impossible, stop and explain instead of improvising.\n\n"
                f"TASK: {task}\n\n{plan}")
        if not feedback:
            ui.warn("plan cancelled")
            return ""
        ui.banner("revising plan")
        plan = planner.run(
            "Revise the plan based on this user feedback (output the full "
            "updated PLAN again, or NO_PLAN if nothing is needed): " + feedback)
        if "NO_PLAN" in plan[:200]:
            ui.warn("plan cancelled")
            return ""
    ui.warn("too many plan revisions — cancelled")
    return ""


def repl(cfg: Config) -> None:
    agents_md = load_agents_md(cfg.root)
    ui.out(f"{ui.BOLD}{ui.CYAN}nv v{__version__}{ui.RESET} — local LLM agent console")
    ui.info(f"sandbox: {cfg.root}")
    if agents_md:
        ui.info("loaded project rules from AGENTS.md")
    _check_connection(cfg)
    ui.info("type /help for commands\n")

    agent = Agent(cfg, "coder", agents_md=agents_md)

    while True:
        try:
            line = input(f"{ui.BOLD}{ui.GREEN}nv>{ui.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            ui.out("\nbye")
            return
        if not line:
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
            elif cmd == "/plan":
                arg = line.split(maxsplit=1)[1] if len(line.split(maxsplit=1)) > 1 else ""
                if arg.lower() in ("on", "off"):
                    cfg.plan_first = arg.lower() == "on"
                    save_global(cfg)
                    ui.info(f"planning step is now {'ON' if cfg.plan_first else 'OFF'} (saved)")
                elif arg:
                    ui.CONFIRM.reset()
                    run_planned(cfg, agent, arg, agents_md)
                else:
                    ui.warn("usage: /plan on | /plan off | /plan <task>")
            elif cmd == "/team":
                if len(parts) < 3 or not parts[1].isdigit():
                    ui.warn("usage: /team 3 <task description>")
                    continue
                ui.CONFIRM.reset()
                run_team(cfg, parts[2], int(parts[1]), agents_md)
            elif cmd == "/review":
                ui.CONFIRM.reset()
                reviewer = Agent(cfg, "reviewer", agents_md=agents_md,
                                 model=cfg.reviewer_model)
                report = reviewer.run("Review the current git diff. "
                                      "Start with the git_diff tool.")
                ui.banner("review report")
                ui.out(report)
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
                if run_discovery(cfg):
                    agent = Agent(cfg, agent.kind, agents_md=agents_md)
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
        ui.CONFIRM.reset()
        if cfg.plan_first and agent.kind == "coder":
            run_planned(cfg, agent, line, agents_md)
        else:
            agent.run(line)


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

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
    if args.host:
        cfg.host = args.host.rstrip("/")
        save_global(cfg)
    if args.model:
        cfg.model = args.model
        save_global(cfg)
    if args.ctx:
        cfg.num_ctx = args.ctx

    if args.scan:
        run_discovery(cfg)
        return

    if args.prompt:
        task = " ".join(args.prompt)
        agents_md = load_agents_md(cfg.root)
        if args.team:
            run_team(cfg, task, args.team, agents_md)
        else:
            executor = Agent(cfg, "coder", agents_md=agents_md)
            if cfg.plan_first and not args.no_plan:
                run_planned(cfg, executor, task, agents_md)
            else:
                executor.run(task)
        return

    repl(cfg)


if __name__ == "__main__":
    main()
