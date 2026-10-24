"""Shared footer rendered at the bottom of every page."""

from datetime import date

import streamlit as st


def render() -> None:
    """Render the shared app footer with project name, year, and data source."""
    year = date.today().year
    st.markdown(
        f"""
        <div class="app-footer">
          <div class="footer-brand">Student Performance Predictor</div>
          <div>&copy; {year} &middot; Built for early academic support, not student evaluation.</div>
          <div>Data source: UCI Machine Learning Repository &mdash; Student Performance Data Set (Cortez &amp; Silva, 2008).</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
