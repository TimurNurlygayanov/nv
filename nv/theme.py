"""Runtime console theming, driven by natural language.

Three layers, applied together:
- semantic colors of nv's own output (prompt, banners, info, diff, ...)
- the terminal's default text/background colors via OSC 10/11 escape
  sequences (supported by Windows Terminal and most modern terminals)
- console font size via the classic conhost API (Windows Terminal ignores
  that API — nv says so and points to Ctrl+Plus)

The theme persists in the global config and is re-applied on startup.
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import sys

from nv import ui
from nv.config import Config
from nv.ollama import OllamaClient, OllamaError, strip_thinking

NAMED_COLORS = {
    "black": "#0c0c0c", "red": "#e74856", "green": "#16c60c",
    "yellow": "#f9f1a5", "blue": "#3b78ff", "magenta": "#b4009e",
    "cyan": "#61d6d6", "white": "#f2f2f2", "gray": "#808080",
    "grey": "#808080", "orange": "#ff8c00", "purple": "#8a2be2",
    "pink": "#ff69b4", "lime": "#a4c400", "brown": "#8b4513",
    "dark_gray": "#404040", "light_gray": "#c0c0c0",
}
COLOR_ROLES = ("fg", "bg", "prompt", "banner", "info", "warn", "error",
               "tool", "diff_add", "diff_del")
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
_BOLD_ROLES = {"prompt", "banner"}


def _to_hex(color: str) -> str | None:
    color = str(color).strip().lower().replace(" ", "_")
    if _HEX.match(color):
        return color
    return NAMED_COLORS.get(color)


def _sgr(hex_color: str, bold: bool = False) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"{ui.BOLD if bold else ''}\x1b[38;2;{r};{g};{b}m"


def validate(changes: dict) -> tuple[dict, list[str]]:
    """Normalize a raw patch. Returns (clean_patch, errors)."""
    patch: dict = {}
    errors: list[str] = []
    for key, value in (changes or {}).items():
        if value in (None, ""):
            continue
        if key in COLOR_ROLES:
            hex_color = _to_hex(value)
            if hex_color:
                patch[key] = hex_color
            else:
                errors.append(f"unknown color '{value}' for {key} "
                              f"(use a name like green/black or #rrggbb)")
        elif key == "font_size":
            spec = str(value).strip()
            if re.match(r"^[+-]?\d{1,2}$", spec):
                patch[key] = spec
            else:
                errors.append(f"bad font_size '{value}' (use 8..72 or +N/-N)")
        else:
            errors.append(f"unknown theme key '{key}'")
    return patch, errors


# ---- font size (conhost API) ----------------------------------------

_ORIG_FONT_SIZE: int | None = None


def _set_font_size(spec: str) -> str:
    global _ORIG_FONT_SIZE
    if os.name != "nt":
        return "font size: only supported on Windows consoles"
    try:
        class COORD(ctypes.Structure):
            _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]

        class FONTEX(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_ulong), ("nFont", ctypes.c_ulong),
                        ("dwFontSize", COORD), ("FontFamily", ctypes.c_uint),
                        ("FontWeight", ctypes.c_uint),
                        ("FaceName", ctypes.c_wchar * 32)]

        k32 = ctypes.windll.kernel32
        handle = k32.GetStdHandle(-11)
        info = FONTEX()
        info.cbSize = ctypes.sizeof(FONTEX)
        if not k32.GetCurrentConsoleFontEx(handle, False, ctypes.byref(info)):
            return "font size: console font API unavailable here"
        current = info.dwFontSize.Y or 16
        if _ORIG_FONT_SIZE is None:
            _ORIG_FONT_SIZE = current
        size = current + int(spec) if spec[0] in "+-" else int(spec)
        size = max(8, min(72, size))
        info.dwFontSize.X = 0
        info.dwFontSize.Y = size
        ok = k32.SetCurrentConsoleFontEx(handle, False, ctypes.byref(info))
    except (OSError, ValueError, AttributeError) as e:
        return f"font size: failed ({e})"
    if os.environ.get("WT_SESSION"):
        return (f"font size: requested {size}, but Windows Terminal ignores "
                "this API — use Ctrl+Plus/Ctrl+Minus or its settings")
    return f"font size: {size}" if ok else "font size: the console refused"


# ---- apply / reset ---------------------------------------------------

def apply(theme: dict, quiet: bool = False) -> list[str]:
    """Apply a validated theme. Returns human-readable descriptions."""
    applied: list[str] = []
    for role, value in theme.items():
        if role == "fg":
            if ui.COLOR:
                sys.stdout.write(f"\x1b]10;{value}\x07")
            applied.append(f"text color: {value}")
        elif role == "bg":
            if ui.COLOR:
                sys.stdout.write(f"\x1b]11;{value}\x07")
            applied.append(f"background: {value}")
        elif role == "font_size":
            applied.append(_set_font_size(value))
        elif role in COLOR_ROLES:
            ui.CODES[role] = _sgr(value, bold=role in _BOLD_ROLES)
            applied.append(f"{role} color: {value}")
    sys.stdout.flush()
    if not quiet:
        for line in applied:
            ui.info(line)
    return applied


def reset() -> None:
    if ui.COLOR:
        sys.stdout.write("\x1b]110\x07\x1b]111\x07")  # default fg/bg
        sys.stdout.flush()
    ui.CODES.clear()
    ui.CODES.update(ui.DEFAULT_CODES)
    if _ORIG_FONT_SIZE is not None:
        _set_font_size(str(_ORIG_FONT_SIZE))
    ui.info("theme reset to defaults")


def describe(theme: dict) -> str:
    if not theme:
        return "(default theme — change it with /theme <description> " \
               "or just ask in the chat)"
    return "\n".join(f"  {k}: {v}" for k, v in sorted(theme.items()))


# ---- natural language -> theme patch (for /theme) -------------------

def from_prompt(cfg: Config, request: str) -> dict | None:
    client = OllamaClient(cfg.host, cfg.model, num_ctx=4096, temperature=0.1,
                          num_predict=300)
    try:
        reply = client.chat([
            {"role": "system", "content":
                "You translate a console-appearance request into JSON. "
                "Allowed keys: fg (default text color), bg (background), "
                "prompt, banner, info, warn, error, tool, diff_add, diff_del "
                "(colors as names like green/black/cyan or #rrggbb), and "
                "font_size (8..72, or '+2'/'-2' relative). Include only the "
                "keys the user asked to change. Output ONLY the JSON object."},
            {"role": "user", "content": request},
        ])
    except OllamaError as e:
        ui.error(str(e))
        return None
    text = strip_thinking(reply.get("content") or "")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        ui.warn("the model returned no theme JSON")
        return None
    try:
        raw = json.loads(m.group(0))
    except ValueError:
        ui.warn("could not parse the model's theme JSON")
        return None
    patch, errors = validate(raw)
    for err in errors:
        ui.warn(err)
    return patch or None
