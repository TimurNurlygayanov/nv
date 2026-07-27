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

Everything works in **plain natural language** — the agent has tools for
reading, editing, running commands, analyzing big docs/logs, and undoing
its own changes, and picks them itself:

```
nv> fix the failing test in tests/test_api.py
nv> read the docs folder and generate user stories for the refunds feature
nv> here is why I think payment fails — check service.py and prove me wrong
nv> undo everything you just did
nv> what changed so far?
```

Slash commands exist as optional instant shortcuts (no model round-trip):
`/diff` shows the git diff immediately, `/undo` reverts without asking the
model, `/team 3 <task>` runs parallel agents, `/ask` forces the read-only
agent, `/load`, `/paste`, `/review`, `/resume`… — see `/help`.

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
| `/ask <question>` | fast read-only Q&A: how the system works, env vars, why tests fail, kubectl/CI advice — no confirmations possible, cites file:line |
| `/paste [question]` | paste a multi-line stack trace or log fragment (finish with `END`) and discuss it |
| `/load <file-or-folder> [question]` | feed files to the agent: inventory → relevant-file selection (confirmed by you) → chunked map-reduce extraction, with an upfront time estimate |
| `/undo` | revert the working tree to the checkpoint taken before the last task |
| `/resume` | continue yesterday's conversation (history persists in `.nv/session.json`) |
| `/note <fact>` | append a fact to project notes (`.nv/notes.md`); agents also save facts themselves and get all notes injected at start |
| `/diff` / `/diff stat` | show git diff right in the chat |
| `/plan on|off`, `/plan <task>` | control the planning step |
| `/team N <task>` | planner splits the task, N agents work in parallel, a separate reviewer agent reviews the diff and can trigger a fixer |
| `/review` | run the reviewer agent on the current diff |
| `/agent <name>` | switch chat agent: `coder`, `writer`, `reviewer` |
| `/agents` | list built-in agent configs |
| `/scan` | re-check `OLLAMA_HOST` / saved host / localhost for Ollama + models |
| `/model`, `/host`, `/config` | settings |
| `/new` | reset conversation history |

## Console theme by prompt

Ask for the look you want, in the chat or via `/theme`:

```
nv> make all text green and background black, increase the font size
nv> /theme orange banners and a purple prompt
nv> /theme reset
```

The agent's `configure_ui` tool (or `/theme <words>`) turns the request into
theme changes: nv's own colors (prompt, banners, info, warnings, diff),
the terminal's default text/background via OSC escape codes, and the console
font size via the Windows console API. Applied instantly and saved to
`~/.nv.json`, so the look survives restarts. Honest limitation: Windows
Terminal ignores the font-size API — nv tells you and points to
Ctrl+Plus / its settings; colors work everywhere modern.

## Answer personality

Work stays rigorous; the prose gets character. Set it in the chat
("answer me in a playful flirty tone") or directly:

```
nv> /style playful and lightly flirty
nv> /style dry sarcasm
nv> /style off
```

Guardrails are built into the prompt: code, diffs, commands, commit
messages and generated documents stay strictly professional; errors and
risks are stated plainly before any charm is applied. Style affects the
chat agents (coder, ask, writer) — reviewer/planner/architect output stays
untouched. Saved in `~/.nv.json`, applies from the next message.

## Terminal passthrough

The chat console is also your terminal — no window switching:

```
nv> !git log --oneline -5          # runs right here
nv> ls                             # bare commands are auto-detected too
nv> env
nv> !!replace foo with bar in every file under tests/
  $ Get-ChildItem tests -Recurse -File | ForEach-Object { (Get-Content $_) -replace 'foo','bar' | Set-Content $_ }
run it? [y/n or type a correction]
```

- `!<command>` runs immediately: no model call, no confirmation (you typed
  it), output printed in place. Interactive programs (`vim`, `less`) attach
  to the console.
- The captured output is **remembered**: your next message to the model
  automatically includes the recent commands + output as context, so
  "почему такой вывод?" right after `env | grep VAR` just works.
- `!!<description>` — the model turns words into one command, shows it, and
  you approve (`y`), reject (`n`), or type a correction and it retries.
- If any command exits non-zero, nv offers to let the model fix it using
  the error output.
- Commands run in your configured `shell` (default: PowerShell on Windows);
  cwd is the project root. These are *your* commands — the agent sandbox
  denylist does not apply to them.

## Daily QA workflows

- **Why is CI red?** — `/load ci_run.log why did the e2e stage fail` — the log
  is chunked and distilled by the local model (any size, it just takes time),
  and the agent explains the failures from the digest.
- **User stories from 20k lines of docs** — `/load docs/ generate user
  stories for <feature>`: nv first builds a cheap inventory of all files
  (headings + previews, no LLM), one quick model call selects only the
  relevant files (you confirm the list, or correct it with feedback), and
  only those are chunk-parsed with a relevance-driven extraction prompt —
  nothing is dropped from the middle of documents.
- **Job estimates** — before any chunked job starts, nv projects its duration
  from measured model speeds (every Ollama reply updates a rolling
  tokens/sec average in `~/.nv-perf.json` — it self-calibrates). Quick jobs
  just run; anything over `confirm_over_seconds` (default 60) asks first:
  "this is a LONG job: estimated ~1.3 h — start it?". If the model was never
  measured, the first chunk is timed and then you decide. Progress lines
  show elapsed time and ETA; Ctrl+C aborts cleanly.
- **Debugging** — `/paste`, paste the stack trace, finish with `END`; the
  agent reads the involved files and explains the cause.
- **"How does this work?"** — `/ask which env variables does the payment
  service read` — read-only agent, instant, cites file:line, suggests exact
  commands (kubectl, env setup) for you to run manually.
- **Fearless changes** — a git checkpoint is taken before every task;
  `/undo` reverts everything the agents did, `/diff` shows it first.
- **Knowledge accumulates** — facts saved via `/note` (or by agents through
  their `save_note` tool) land in `.nv/notes.md` and are injected into every
  agent's system prompt, so the model "remembers" your system across sessions.
- **Long investigations** — chat history is saved automatically; next day
  `/resume` continues where you stopped. Ctrl+C aborts a running task
  without killing the console.

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

## Minimalism enforcement

"One line that does the job, not two million" is enforced mechanically,
not just requested politely:

1. **Prompt rule** — every agent carries a mandatory brevity rule: short
   final answers, no pasting back file contents, touch nothing beyond the
   ask.
2. **Hard token cap** — every reply is generated with Ollama's
   `num_predict` limit (`max_answer_tokens`, default 2048), so the model
   physically cannot produce a wall of text. Auxiliary calls are capped
   tighter (command synthesis 200, theme 300, file selection 500).
3. **Diff-size gate** — any single change touching more than
   `max_diff_lines` (default 60) prints a LARGE CHANGE warning and demands
   explicit approval — even if you pressed `a` (yes-to-all). Rejecting it
   tells the model to make the smallest change that solves the task or
   split it into steps.
4. **Minimizer pass** — if a finished task still changed more lines than
   the limit, nv offers to run a dedicated `minimizer` agent: it re-reads
   the diff and shrinks it (dead code, needless refactors, verbose
   constructs) with behavior kept identical, every cut confirmed by you.
   Run it any time with `/minimize`.

Both limits are configurable in `.nv.json`.

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
`history_char_budget`, `confirm_over_seconds`, `shell`, `theme`,
`personality`, `max_answer_tokens`, `max_diff_lines`.
