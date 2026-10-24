"""Inline SVG icons (Feather-style, stroke-based) used inside custom HTML cards.

Streamlit's `:material/name:` icon syntax only works inside native widgets
(buttons, st.Page, st.metric, ...). Custom HTML components need their own
markup, so icons used inside `st.markdown(..., unsafe_allow_html=True)`
blocks come from here instead -- plain inline SVG, no emoji, no external
image requests.
"""

_PATHS: dict[str, str] = {
    "home": '<path d="M3 11.5 12 4l9 7.5"/><path d="M5 10v10h14V10"/><path d="M9 20v-6h6v6"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.2" fill="currentColor"/>',
    "insights": '<path d="M4 19h16"/><path d="M7 19V10"/><path d="M12 19V5"/><path d="M17 19v-7"/>',
    "performance": '<path d="M3 3v18h18"/><path d="M7 15l4-5 3 3 5-7"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6"/><circle cx="12" cy="8" r="1" fill="currentColor"/>',
    "mail": '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="m3 7 9 6 9-6"/>',
    "users": '<circle cx="9" cy="8" r="3.2"/><path d="M3 20c0-3.3 2.7-6 6-6s6 2.7 6 6"/><circle cx="17" cy="9" r="2.6"/><path d="M15.5 14.2c2.4.5 4.5 2.6 4.5 5.8"/>',
    "award": '<circle cx="12" cy="8" r="5.5"/><path d="m8.5 12.8-1.7 6.7 5.2-2.6 5.2 2.6-1.7-6.7"/>',
    "alert": '<path d="M10.3 3.6 1.9 18a1.6 1.6 0 0 0 1.4 2.4h17.4a1.6 1.6 0 0 0 1.4-2.4L13.7 3.6a1.6 1.6 0 0 0-2.8 0Z"/><path d="M12 9v4.5"/><circle cx="12" cy="16.5" r="1" fill="currentColor"/>',
    "layers": '<path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 13 9 5 9-5"/>',
    "check-circle": '<circle cx="12" cy="12" r="9.5"/><path d="m8 12.5 2.8 2.8L16.5 9"/>',
    "x-circle": '<circle cx="12" cy="12" r="9.5"/><path d="m9 9 6 6M15 9l-6 6"/>',
    "book": '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2Z"/>',
    "github": '<path d="M9 19c-4.3 1.3-4.3-2.5-6-3m12 5v-3.3c0-.9.3-1.5.7-1.8-2.5-.3-5.2-1.3-5.2-5.6 0-1.2.4-2.2 1.1-3-.1-.3-.5-1.5.1-3.1 0 0 .9-.3 3 1.1a10.3 10.3 0 0 1 5.4 0c2.1-1.4 3-1.1 3-1.1.6 1.6.2 2.8.1 3.1.7.8 1.1 1.8 1.1 3 0 4.3-2.7 5.3-5.2 5.6.4.3.8 1 .8 2.1V21"/>',
    "calendar": '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M16 3v4M8 3v4M3 10h18"/>',
    "clock": '<circle cx="12" cy="12" r="9.5"/><path d="M12 7v5l3.5 2"/>',
    "trending-up": '<path d="m3 17 6-6 4 4 8-9"/><path d="M15 6h6v6"/>',
    "shield": '<path d="M12 2.5 20 6v6c0 5-3.4 8.4-8 9.5-4.6-1.1-8-4.5-8-9.5V6l8-3.5Z"/>',
    "database": '<ellipse cx="12" cy="5.5" rx="8" ry="3"/><path d="M4 5.5V12c0 1.7 3.6 3 8 3s8-1.3 8-3V5.5"/><path d="M4 12v6.5c0 1.7 3.6 3 8 3s8-1.3 8-3V12"/>',
    "cpu": '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/>',
    "scale": '<path d="M12 3v18M7 21h10"/><path d="m4 7 4-1 4 1-2 7a2.5 2.5 0 0 1-4.5 0L4 7Z"/><path d="m12 7 4-1 4 1-2 7a2.5 2.5 0 0 1-4.5 0L12 7Z"/>',
    "send": '<path d="M22 2 11 13"/><path d="m22 2-7 20-4-9-9-4 20-7Z"/>',
    "heart": '<path d="M12 20.5s-7.5-4.6-9.8-9A5.4 5.4 0 0 1 12 6a5.4 5.4 0 0 1 9.8 5.5c-2.3 4.4-9.8 9-9.8 9Z"/>',
    "arrow-right": '<path d="M5 12h14"/><path d="m13 6 6 6-6 6"/>',
    "zap": '<path d="M13 2 4 14h7l-1 8 9-12h-7l1-8Z"/>',
    "filter": '<path d="M4 4h16l-6 8v7l-4 2v-9L4 4Z"/>',
}


def svg(name: str, size: int = 22, stroke_width: float = 1.8) -> str:
    """Return an inline SVG snippet for a named icon.

    Args:
        name: Icon key from the internal icon set (see ``_PATHS``).
        size: Width/height in pixels.
        stroke_width: SVG stroke width for line icons.

    Returns:
        An ``<svg>...</svg>`` string using ``currentColor``, so it
        inherits color from its CSS context.
    """
    path = _PATHS.get(name, _PATHS["info"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">{path}</svg>'
    )
