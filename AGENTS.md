# Rules for agents working on nv

- Pure Python 3.10+ **stdlib only** — never add dependencies, never touch
  pyproject dependencies. If a feature seems to need a library, it needs a
  simpler design instead.
- The only allowed network activity in the code is talking to the user's
  Ollama host (and probing OLLAMA_HOST/localhost). Never add code that
  scans networks or calls other services.
- Keep the sandbox promises: agent file access stays inside the start
  folder; the command denylist in sandbox.py must only grow, never shrink.
- Every mutating action (file write/edit, command) must go through user
  confirmation. Never add an unattended "auto-approve" default.
- Optimize for small local models: capped/ranged file reads, capped search
  results, short focused prompts, history compaction. Do not add features
  that assume a 100k-token context.
- Minimal-diff style: small functions, no classes where a function works,
  match existing module layout (one concern per module: ui, tools, agent,
  ingest, ...).
- Comments explain constraints, not restate code. Docstrings at module top
  describe the "why".
- After changes, verify with the smoke tests in the scratchpad (or ask the
  user to run them); a feature is not done until its failure paths return
  helpful messages instead of tracebacks.
- User-facing text (README, console output) is plain and concrete: say what
  will happen, show the command, avoid marketing language.
