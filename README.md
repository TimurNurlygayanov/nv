# nv — local LLM agent console

A console tool that turns your Ollama models (qwen3-coder etc.) into coding
agents: they read and modify code, run commands, answer questions and write
reports — with **your confirmation before every change**, fully sandboxed to
the folder where you start it.

No dependencies. Pure Python 3.10+ stdlib.

## Install

With [uv](https://docs.astral.sh/uv/) (recommended — one command, `nv`
becomes available everywhere):

```bash
uv tool install --editable .
```

Run it once without installing:

```bash
uvx --from . nv
```

No uv? `pip install -e .` or `python -m nv` from the repo folder also work.
No dependencies either way — pure Python 3.10+ stdlib.

## Setup

Just run `nv` — if the configured host is unreachable it offers to find
Ollama for you. Discovery checks only explicit candidates — the
`OLLAMA_HOST` environment variable, the previously saved host, and
`127.0.0.1`/localhost — verifies each with `/api/tags` and shows the models
the server has (nv never port-scans the network). Pick a model once and
everything is saved to `~/.nv.json`. Rerun any time with `/scan`
(or `nv --scan`).

> Remote desktop setup: on the desktop set `OLLAMA_HOST=0.0.0.0`, restart
> Ollama and allow it through the firewall; on this machine set
> `OLLAMA_HOST=http://<desktop-ip>:11434` (or run `/host ...` once) and nv
> will pick it up.

Manual setup still works:

```bash
nv --host http://192.168.1.50:11434 --model qwen3-coder:30b
```

## Use

Start `nv` in the project folder you want the agent to work on. That folder
becomes the sandbox — the agent cannot see anything outside it.

```
nv> fix the failing test in tests/test_api.py
nv> /diff                     # see what the agent changed, colored
nv> /team 3 add type hints to all modules   # 3 parallel agents + reviewer
nv> /review                   # separate reviewer agent checks the git diff
```

### Pipeline

For coding tasks nv runs a three-stage pipeline:

1. **plan** — an `architect` agent (read-only access) explores the project and
   proposes a short numbered plan (3-7 small steps with exact file paths).
   Questions and trivial changes skip this automatically.
2. **approve** — you approve the plan (`y`), cancel it (`n`), or type feedback
   and the architect revises the plan.
3. **execute** — the coder agent follows the approved plan step by step (still
   confirming every file change / command), without adding work beyond it.

Toggle with `/plan on` / `/plan off` (or `--no-plan` for one run); force it
for a single task with `/plan <task>`.

One-shot mode: `nv "add a --verbose flag to cli.py"` or `nv --team 3 "task"`.

## Commands

| command | action |
|---|---|
| `/diff` / `/diff stat` | show git diff right in the chat |
| `/plan on|off`, `/plan <task>` | control the planning step |
| `/team N <task>` | planner splits the task, N agents work in parallel, a separate reviewer agent reviews the diff and can trigger a fixer |
| `/review` | run the reviewer agent on the current diff |
| `/agent <name>` | switch chat agent: `coder`, `writer`, `reviewer` |
| `/agents` | list built-in agent configs |
| `/scan` | re-check `OLLAMA_HOST` / saved host / localhost for Ollama + models |
| `/model`, `/host`, `/config` | settings |
| `/new` | reset conversation history |

## Confirmations

Every `write_file`, `edit_file` and `run_command` shows a preview (colored
diff for file changes) and waits for you:
`y` yes · `n` no · `a` yes-to-all for this task · or type feedback that is
sent back to the agent as a correction.

## Safety / sandbox

- All file access is restricted to the start folder; paths outside are refused.
- Commands are checked against a denylist: no network (`curl`, `wget`,
  `Invoke-WebRequest`, URLs, `ssh`…), no package installs (`pip install`,
  `npm install`, `winget`…), no `git push/pull/remote`, no paths outside the
  folder, no system-level commands.
- The only network activity nv itself performs is talking to your Ollama
  host; `/scan` only probes explicitly known addresses (`OLLAMA_HOST`,
  saved host, localhost) — it never scans the network.

## Local-model optimizations

Built-in agent configs are tuned for small/local models:

- **ranged reads** — `read_file` returns max 200 numbered lines per call; the
  agent is instructed to `search` first and read only relevant regions,
  never whole files;
- **minimal diffs** — prompts require the smallest possible change via exact
  text replacement (`edit_file`), no drive-by refactoring;
- **anti-hallucination** — paths must be verified with `list_files`/`search`
  before use; edits must quote exact text from a real read; failed calls must
  not be blindly repeated;
- **context compaction** — old tool outputs are truncated and oldest turns
  dropped when history approaches the model's context budget;
- **AGENTS.md** — if the folder contains `AGENTS.md` (or `CLAUDE.md`), its
  rules are injected into every agent's system prompt;
- **tool-call fallback** — if the model prints a JSON tool call as text
  instead of a structured call, nv parses it anyway.

## Config

`~/.nv.json` (global) or `.nv.json` in the project (overrides). Keys:
`host`, `model`, `review_model`, `num_ctx`, `temperature`, `max_steps`, `plan_first`,
`max_read_lines`, `max_search_hits`, `max_tool_output`, `command_timeout`,
`history_char_budget`.
