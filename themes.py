"""
Web UI colors for light/dark mode.

static/style.css expects many --variables (e.g. --bg_primary). Those names stay fixed so you
do not touch CSS when reskinning — only the small palettes below.

Flow: index.html is rendered with themes_json; app.js sets document.documentElement styles.
"""

from __future__ import annotations


def _light_tokens() -> dict[str, str]:
    """A handful of named colors; the rest are aliases for readability in CSS."""
    paper = "#fffefb"
    panel = "#f5f0e8"
    rail = "#e9eced"
    ink = "#366c9c"
    muted = "#8194b1"
    faint = "#b0c9df"
    accent = "#2563eb"
    danger = "#ef4444"
    danger_dark = "#dc2626"
    ok = "#10b981"

    return {
        "bg_primary": paper,
        "bg_secondary": panel,
        "bg_tertiary": rail,
        "bg_quaternary": panel,
        "text_primary": ink,
        "text_secondary": muted,
        "text_tertiary": faint,
        "accent_primary": muted,
        "accent_hover": ink,
        "accent_secondary": accent,
        "msg_user_bg": rail,
        "msg_user_text": ink,
        "msg_assistant_bg": paper,
        "msg_assistant_text": ink,
        "status_connected": ok,
        "status_disconnected": danger,
        "btn_primary": muted,
        "btn_primary_hover": ink,
        "btn_danger": danger,
        "btn_danger_hover": danger_dark,
        "btn_secondary": rail,
        "btn_secondary_hover": faint,
        "modal_overlay": "rgba(0,0,0,0.5)",
        "modal_bg": paper,
        "scrollbar_track": panel,
        "scrollbar_thumb": faint,
        "scrollbar_thumb_hover": muted,
    }


def _dark_tokens() -> dict[str, str]:
    surface = "#1a1a1a"
    raised = "#242424"
    border = "#333333"
    ink = "#e0e0e0"
    muted = "#a0a0a0"
    faint = "#666666"
    accent = "#4a9eff"
    cool = "#8194b1"
    cool_hi = "#b0c9df"
    danger = "#ef4444"
    danger_dark = "#dc2626"
    ok = "#10b981"

    return {
        "bg_primary": surface,
        "bg_secondary": raised,
        "bg_tertiary": border,
        "bg_quaternary": raised,
        "text_primary": ink,
        "text_secondary": muted,
        "text_tertiary": faint,
        "accent_primary": cool,
        "accent_hover": cool_hi,
        "accent_secondary": accent,
        "msg_user_bg": border,
        "msg_user_text": ink,
        "msg_assistant_bg": raised,
        "msg_assistant_text": ink,
        "status_connected": ok,
        "status_disconnected": danger,
        "btn_primary": cool,
        "btn_primary_hover": cool_hi,
        "btn_danger": danger,
        "btn_danger_hover": danger_dark,
        "btn_secondary": border,
        "btn_secondary_hover": "#444444",
        "modal_overlay": "rgba(0,0,0,0.8)",
        "modal_bg": raised,
        "scrollbar_track": raised,
        "scrollbar_thumb": border,
        "scrollbar_thumb_hover": "#444444",
    }


COLORS_LIGHT = _light_tokens()
COLORS_DARK = _dark_tokens()
