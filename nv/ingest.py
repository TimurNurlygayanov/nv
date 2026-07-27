"""Feed big inputs (docs folders, CI logs, test output) to the chat.

Pipeline for large inputs:
  1. inventory  — cheap, no LLM: list text files with previews/headings
  2. select     — one LLM call picks only the files relevant to the question
  3. extract    — selected files are chunked; every chunk is distilled with a
                  mode-aware prompt (docs vs log). No chunks are dropped.
  4. merge      — partial extracts are merged (hierarchically if needed)

Chunk counts are known before starting, so jobs are estimated up front
(using measured speeds from nv.perf) and long jobs require confirmation.
"""
from __future__ import annotations

import fnmatch
import json
import re
import time
from pathlib import Path

from nv import perf, ui
from nv.config import Config
from nv.ollama import OllamaClient, strip_thinking
from nv.tools import SKIP_DIRS, TEXT_EXT_HINT

DIRECT_LIMIT = 15000    # chars: below this a single file goes in verbatim
CHUNK_CHARS = 9000
HARD_CHUNK_CAP = 500    # absolute runaway backstop (~5-8 hours of work)
AVG_OUT_TOKENS = 300    # typical extraction answer length
MAX_INVENTORY = 250
MAX_SELECTED = 30

_EXTRACT_SYS = ("You extract facts from project material for a QA engineer. "
                "Output only the extracted items, no preamble.")

_MODE_PROMPTS = {
    "log": ("Extract from this log fragment: error messages, failed test "
            "names, stack traces (keep the innermost frames), warnings and "
            "anomalies. Quote exact identifiers and values. "
            "If nothing notable, output exactly 'nothing relevant'."),
    "docs": ("From this documentation fragment extract ONLY information "
             "relevant to the goal above: affected services and components, "
             "API endpoints with parameters, data models, constraints and "
             "limits, existing behaviors, dependencies, edge cases. Quote "
             "exact names and values. "
             "If nothing is relevant to the goal, output exactly 'nothing relevant'."),
}

_LOG_LINE = re.compile(
    r"ERROR|WARN|FAIL|Traceback|Exception|\d{2}:\d{2}:\d{2}", re.IGNORECASE)


def _skip(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in SKIP_DIRS)


def detect_mode(pieces: list[tuple[str, str]]) -> str:
    """'log' or 'docs', by extension and content sampling."""
    log_votes = 0
    for name, text in pieces:
        if Path(name).suffix.lower() in (".log", ".out", ".err"):
            log_votes += 1
            continue
        sample = text.splitlines()[:200]
        if sample and sum(1 for ln in sample if _LOG_LINE.search(ln)) > len(sample) * 0.15:
            log_votes += 1
    return "log" if log_votes > len(pieces) / 2 else "docs"


# ---- stage 1: inventory (no LLM) ------------------------------------

def inventory(base: Path, root: Path) -> list[dict]:
    """Text files under base with a short preview (md headings preferred)."""
    entries: list[dict] = []
    for p in sorted(base.rglob("*")):
        rel = p.relative_to(root)
        if any(_skip(part) for part in rel.parts) or not p.is_file():
            continue
        if p.suffix.lower() not in TEXT_EXT_HINT and p.suffix.lower() != ".rst":
            continue
        try:
            if p.stat().st_size > 2_000_000:
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines()
        headings = [ln.strip() for ln in lines if ln.lstrip().startswith("#")][:6]
        preview = "; ".join(headings) if headings else \
            " / ".join(ln.strip()[:60] for ln in lines if ln.strip())[:120]
        entries.append({"path": str(rel).replace("\\", "/"),
                        "lines": len(lines), "preview": preview[:120]})
        if len(entries) >= MAX_INVENTORY:
            break
    return entries


# ---- stage 2: relevance selection (one LLM call) --------------------

def select_files(cfg: Config, entries: list[dict], question: str) -> list[str] | None:
    """Returns relevant paths, or None if the model's answer was unusable."""
    listing = "\n".join(
        f"{i}. {e['path']} | {e['lines']} lines | {e['preview']}"
        for i, e in enumerate(entries, 1))
    client = OllamaClient(cfg.host, cfg.model, num_ctx=cfg.num_ctx,
                          temperature=0.1)
    reply = client.chat([
        {"role": "system",
         "content": "You select which project files are relevant to a goal. "
                    "Output ONLY a JSON array of file paths, nothing else."},
        {"role": "user",
         "content": f"GOAL: {question}\n\nFiles (path | lines | preview):\n"
                    f"{listing}\n\nOutput a JSON array with the paths relevant "
                    f"to the goal, most relevant first, at most {MAX_SELECTED}. "
                    "When unsure about a file, include it."},
    ])
    text = strip_thinking(reply.get("content") or "")
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        items = json.loads(m.group(0))
    except ValueError:
        return None
    known = {e["path"] for e in entries}
    picked = [s for s in items if isinstance(s, str) and s in known]
    return picked[:MAX_SELECTED] or None


# ---- estimates -------------------------------------------------------

def count_chunks(pieces: list[tuple[str, str]]) -> int:
    return sum(len(split_chunks(t)) for _, t in pieces)


def estimate_seconds(cfg: Config, n_chunks: int) -> float | None:
    sp = perf.speeds(cfg.host, cfg.model)
    if not sp:
        return None
    prompt_tps, gen_tps = sp
    per_chunk = (CHUNK_CHARS / 3.5) / prompt_tps + AVG_OUT_TOKENS / gen_tps
    merges = (1 if n_chunks > 1 else 0) + n_chunks // 10
    return per_chunk * (n_chunks + merges)


# ---- stage 3+4: extract and merge -----------------------------------

def split_chunks(text: str, size: int = CHUNK_CHARS) -> list[str]:
    chunks: list[str] = []
    buf: list[str] = []
    n = 0
    for raw in text.splitlines():
        # a single enormous line (minified JSON etc.) is hard-split
        lines = [raw[i:i + size] for i in range(0, len(raw), size)] or [""]
        for line in lines:
            buf.append(line)
            n += len(line) + 1
            if n >= size:
                chunks.append("\n".join(buf))
                buf, n = [], 0
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def _merge(client: OllamaClient, partials: list[str], focus: str,
           note) -> str:
    """Merge partial extracts; hierarchically when they don't fit at once."""
    while len(partials) > 1:
        batches: list[list[str]] = [[]]
        size = 0
        for part in partials:
            if size > 30000 and batches[-1]:
                batches.append([])
                size = 0
            batches[-1].append(part)
            size += len(part)
        merged: list[str] = []
        for bi, batch in enumerate(batches, 1):
            if len(batches) > 1:
                note(f"merging batch {bi}/{len(batches)}...")
            elif len(batch) > 1:
                note("merging partial extracts...")
            if len(batch) == 1:
                merged.append(batch[0])
                continue
            reply = client.chat([
                {"role": "system", "content": _EXTRACT_SYS},
                {"role": "user", "content":
                    f"{focus}Merge these partial extracts into ONE structured "
                    "digest: deduplicated facts grouped by topic, most "
                    "important first. Keep exact names and values.\n\n"
                    + "\n\n".join(batch)},
            ])
            merged.append(strip_thinking(reply.get("content") or "").strip())
        if len(merged) == len(partials):  # safety: guarantee progress
            merged = ["\n\n".join(merged)]
        partials = merged
    return partials[0] if partials else ""


def interactive_pipeline(cfg: Config, target: Path,
                         question: str) -> tuple[str, bool, str] | None:
    """Full flow: (folder -> inventory -> select -> confirm) -> estimate ->
    confirm -> extract+merge. Talks to the user via ui.CONFIRM.
    Returns (text, was_summarized, source) or None if cancelled/not found."""
    p = target
    if not p.exists():
        ui.warn(f"not found: {p}")
        return None
    try:
        p.relative_to(cfg.root)
        rel_root = cfg.root
    except ValueError:  # explicit user path outside the project (e.g. a CI log)
        rel_root = p if p.is_dir() else p.parent

    if p.is_dir():
        if not question:
            ui.warn("a goal/question is needed to select relevant files "
                    "from a folder")
            return None
        entries = inventory(p, rel_root)
        if not entries:
            ui.warn(f"no text files found under {p}")
            return None
        ui.info(f"{len(entries)} text files under {p.name} — selecting the "
                "relevant ones (one quick model call)...")
        selected = select_files(cfg, entries, question)
        if selected is None:
            if len(entries) <= 10:
                selected = [e["path"] for e in entries]
                ui.warn("selection answer was unusable — taking all files")
            else:
                ui.warn("could not select relevant files; name a specific "
                        "file or a smaller folder")
                return None
        ui.banner(f"selected {len(selected)} file(s)")
        for s in selected:
            ui.out(f"  {s}")
        ok, feedback = ui.CONFIRM.ask("parse these files?")
        if not ok and feedback:
            retry = select_files(
                cfg, entries, f"{question}\nUser correction: {feedback}")
            if retry:
                selected = retry
                ui.banner(f"re-selected {len(selected)} file(s)")
                for s in selected:
                    ui.out(f"  {s}")
                ok, _ = ui.CONFIRM.ask("parse these files?")
        if not ok:
            ui.warn("analysis cancelled")
            return None
        pieces = []
        for s in selected:
            try:
                pieces.append((s, (rel_root / s).read_text(
                    encoding="utf-8", errors="replace")))
            except OSError:
                ui.warn(f"skipped unreadable file: {s}")
        if not pieces:
            return None
        source = f"{len(pieces)} files under {p.name}"
    else:
        pieces = [(p.name, p.read_text(encoding="utf-8", errors="replace"))]
        source = p.name

    total_chars = sum(len(t) for _, t in pieces)
    ui.info(f"input: {total_chars} chars, "
            f"{sum(t.count(chr(10)) + 1 for _, t in pieces)} lines")

    if len(pieces) == 1 and total_chars <= DIRECT_LIMIT:
        ui.info("small input — using it directly (quick job)")
        return pieces[0][1], False, source

    mode = detect_mode(pieces)
    n_chunks = count_chunks(pieces)
    est = estimate_seconds(cfg, n_chunks)
    ui.info(f"mode: {mode}, {n_chunks} chunks to process")
    confirm_cb = None
    if est is not None:
        if est > cfg.confirm_over_seconds:
            ok, _ = ui.CONFIRM.ask(
                f"this is a LONG job: estimated {perf.humanize(est)} "
                f"on {cfg.model}. start it?")
            if not ok:
                ui.warn("analysis cancelled")
                return None
        else:
            ui.info(f"estimated {perf.humanize(est)} — starting")
    else:
        ui.info("no speed data for this model yet — the first chunk will "
                "measure it, then you decide")

        def confirm_cb(projected: float, total: int) -> bool:
            ok, _ = ui.CONFIRM.ask(
                f"measured: full job ({total} chunks) will take about "
                f"{perf.humanize(projected)}. continue?")
            return ok

    started = time.time()
    digest = summarize_pieces(cfg, pieces, question, mode, ui.info,
                              confirm_cb, cfg.confirm_over_seconds)
    if digest is None:
        ui.warn("analysis cancelled")
        return None
    ui.info(f"digest ready in {perf.humanize(time.time() - started)}")
    ui.banner("digest")
    ui.out(digest)
    return digest, True, source


def summarize_pieces(cfg: Config, pieces: list[tuple[str, str]],
                     question: str, mode: str, on_status=None,
                     confirm_cb=None, confirm_over: float = 60.0) -> str | None:
    """Extract + merge. Returns the digest, or None if the user aborted via
    confirm_cb (called once after the first chunk when speed was unknown)."""
    def note(msg: str) -> None:
        if on_status:
            on_status(msg)

    labeled: list[tuple[str, str]] = []
    for name, text in pieces:
        for j, chunk in enumerate(split_chunks(text), 1):
            labeled.append((f"{name} #{j}", chunk))
    capped = 0
    if len(labeled) > HARD_CHUNK_CAP:
        capped = len(labeled) - HARD_CHUNK_CAP
        labeled = labeled[:HARD_CHUNK_CAP]
        note(f"warning: input truncated to {HARD_CHUNK_CAP} chunks "
             f"({capped} dropped)")

    client = OllamaClient(cfg.host, cfg.model, num_ctx=cfg.num_ctx,
                          temperature=0.1)
    focus = f"GOAL / user question: {question}\n" if question else ""
    extract_prompt = _MODE_PROMPTS[mode]

    partials: list[str] = []
    started = time.time()
    for i, (label, chunk) in enumerate(labeled, 1):
        reply = client.chat([
            {"role": "system", "content": _EXTRACT_SYS},
            {"role": "user",
             "content": f"{focus}{extract_prompt}\n\n{chunk}"},
        ])
        content = strip_thinking(reply.get("content") or "").strip()
        if content and "nothing relevant" not in content.lower()[:40] \
                and "nothing notable" not in content.lower()[:40]:
            partials.append(f"[{label}] {content}")

        elapsed = time.time() - started
        avg = elapsed / i
        if i < len(labeled):
            note(f"chunk {i}/{len(labeled)} done — elapsed "
                 f"{perf.humanize(elapsed)}, ETA {perf.humanize(avg * (len(labeled) - i))}")
        if i == 1 and confirm_cb and len(labeled) > 2:
            projected = avg * len(labeled) * 1.15  # + merge overhead
            if projected > confirm_over and not confirm_cb(projected, len(labeled)):
                return None

    if not partials:
        digest = "nothing relevant to the question was found in the input"
    else:
        digest = _merge(client, partials, focus, note)
    if capped:
        digest += f"\n\n[note: {capped} trailing chunks were not processed]"
    return digest
