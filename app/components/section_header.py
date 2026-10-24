"""A consistent eyebrow + title + subtitle block used to open each section."""

import streamlit as st


def render(eyebrow: str, title: str, subtitle: str | None = None, center: bool = False) -> None:
    """Render a section header.

    Args:
        eyebrow: Short uppercase label above the title (e.g. "How it works").
        title: Section title.
        subtitle: Optional supporting sentence under the title.
        center: If True, center-align the text (used on the Home page).
    """
    align = "center" if center else "left"
    subtitle_html = f'<p class="section-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="section-header fade-in-up" style="text-align:{align};">
          <div class="eyebrow">{eyebrow}</div>
          <h2>{title}</h2>
          {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
