"""Full-width gradient hero section used on the Home page."""

import streamlit as st

from utils.icons import svg


def render(badge: str, title_html: str, subtitle: str) -> None:
    """Render the hero markup (badge, animated headline, subtitle).

    The primary call-to-action button is rendered separately by the
    caller with ``st.page_link`` immediately after this, since Streamlit
    widgets cannot live inside a raw HTML string.

    Args:
        badge: Short pill text above the headline.
        title_html: Headline HTML (may include a ``<span class="gradient-text">``
            wrapped portion for emphasis).
        subtitle: One-line value proposition under the headline.
    """
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-content">
            <div class="hero-badge">{svg("zap", 14)} {badge}</div>
            <h1 class="hero-title">{title_html}</h1>
            <p class="hero-subtitle">{subtitle}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
