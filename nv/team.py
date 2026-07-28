"""Team mode: planner splits the task, worker agents run in parallel,
a separate reviewer agent reviews the resulting git diff."""
from __future__ import annotations

import json
import re
import threading

from nv import ui
from nv.agent import Agent
from nv.config import Config


def _parse_subtasks(text: str, limit: int) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except ValueError:
        return []
    tasks = [it for it in items
             if isinstance(it, dict) and it.get("task")][:limit]
    return tasks


def run_team(cfg: Config, task: str, workers: int, agents_md: str) -> None:
    workers = max(1, min(workers, 6))

    # 1) plan
    ui.banner(f"planner: splitting the task into up to {workers} subtasks")
    planner = Agent(cfg, "planner", name="planner", agents_md=agents_md)
    plan_text = planner.run(
        f"N={workers}. Split this task into at most {workers} independent "
        f"subtasks (non-overlapping files):\n\n{task}")
    subtasks = _parse_subtasks(plan_text, workers)
    if not subtasks:
        ui.warn("planner did not return valid subtasks — running as a single agent")
        subtasks = [{"title": "task", "task": task}]

    ui.banner("plan")
    for i, st in enumerate(subtasks, 1):
        ui.out(f"  {i}. {st.get('title', 'subtask')}", ui.BOLD)
        ui.info(f"     {st['task'][:200]}")
    ok, feedback = ui.CONFIRM.ask(f"run {len(subtasks)} agent(s) on this plan?")
    if not ok:
        ui.warn("team run cancelled" + (f" — {feedback}" if feedback else ""))
        return

    # 2) execute in parallel (quiet workers; confirmations still serialized)
    results: dict[str, str] = {}
    stop = threading.Event()

    def work(idx: int, sub: dict) -> None:
        name = f"worker-{idx}"
        try:
            agent = Agent(cfg, "coder", name=name, agents_md=agents_md,
                          quiet=True)
            agent.stop_event = stop
            ui.out(f"\n[{name}] started: {sub.get('title', '')}", ui.CYAN)
            results[name] = agent.run(sub["task"])
            ui.out(f"\n[{name}] finished: {results[name][:300]}", ui.CYAN)
        except Exception as e:  # a dead worker must be visible, not silent
            results[name] = f"ERROR: worker crashed: {e}"
            ui.error(f"\n[{name}] crashed: {e}")

    threads = [threading.Thread(target=work, args=(i, st), daemon=True)
               for i, st in enumerate(subtasks, 1)]
    for t in threads:
        t.start()
    try:
        for t in threads:
            while t.is_alive():
                t.join(0.5)
    except KeyboardInterrupt:
        ui.warn("\nCtrl+C — asking workers to stop (they finish the "
                "current model step, no new edits)...")
        stop.set()
        for t in threads:
            t.join(15)
        ui.warn("team run stopped; /diff shows what was already changed, "
                "/undo reverts it")
        return

    failed = [n for n, r in results.items() if r.startswith("ERROR")]
    if failed:
        ui.warn(f"{len(failed)} of {len(threads)} worker(s) failed: "
                f"{', '.join(sorted(failed))} — their subtasks are NOT done")

    # 3) review by a separate agent
    ui.banner("reviewer: checking the combined git diff")
    reviewer = Agent(cfg, "reviewer", name="reviewer", agents_md=agents_md,
                     model=cfg.reviewer_model)
    report = reviewer.run(
        "Review the current git diff produced by other agents for this task:\n"
        f"{task}\n\nStart with the git_diff tool.")
    ui.banner("review report")
    ui.out(report)

    if "NEEDS_FIXES" in report:
        ok, _ = ui.CONFIRM.ask("reviewer requested fixes — run a fixer agent?")
        if ok:
            fixer = Agent(cfg, "coder", name="fixer", agents_md=agents_md)
            summary = fixer.run(
                "A reviewer found issues in the current git diff. Fix ONLY the "
                "issues listed below with minimal changes:\n\n" + report)
            ui.banner("fixer summary")
            ui.out(summary)
