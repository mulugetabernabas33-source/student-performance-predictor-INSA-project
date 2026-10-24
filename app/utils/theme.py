"""Loads the shared stylesheet once per session."""

from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
STYLE_PATH = APP_DIR / "assets" / "style.css"


def load_theme() -> None:
    """Inject the app's stylesheet into the current page.

    Safe to call on every page -- Streamlit re-renders the DOM per page
    load, so the CSS must be re-injected each time rather than cached
    across reruns.
    """
    css = STYLE_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
