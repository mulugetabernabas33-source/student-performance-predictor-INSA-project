"""Contact / Team page: author info, links, and a local-only feedback form."""

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
for _p in (str(APP_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from components import footer, section_header  # noqa: E402
from utils.icons import svg  # noqa: E402
from utils.theme import load_theme  # noqa: E402

load_theme()

section_header.render("Contact", "Team & links", center=False)

info_col, links_col = st.columns([1.3, 1], gap="large")

with info_col:
    st.markdown(
        """
        <div class="card">
          <div class="form-group-title">Author</div>
          <p style="margin:0;"><strong>Student Performance Predictor</strong> &mdash; a course project
          applying supervised machine learning (regression and classification) to an early
          academic-risk screening problem.</p>
          <p style="margin-top:0.75rem; margin-bottom:0; color:var(--text-faint); font-size:0.85rem;">
            Course: Applied Machine Learning / Data Science Capstone
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with links_col:
    st.markdown(
        f"""
        <div class="card">
          <div class="form-group-title">{svg("github", 20)} Links</div>
          <p style="margin:0 0 0.5rem;">
            <a href="https://github.com/your-username/student-performance-predictor" target="_blank">
              GitHub repository
            </a>
          </p>
          <p style="margin:0;">
            <a href="https://archive.ics.uci.edu/dataset/320/student+performance" target="_blank">
              Dataset source (UCI)
            </a>
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

section_header.render("Feedback", "Tell us what you think", center=False)

if "feedback_sent" not in st.session_state:
    st.session_state["feedback_sent"] = False

if st.session_state["feedback_sent"]:
    st.markdown(
        f"""
        <div class="card" style="text-align:center;">
          <div class="icon-wrap" style="margin:0 auto 0.8rem; background:var(--pass-bg); color:var(--pass);">
            {svg("check-circle", 24)}
          </div>
          <h4 style="margin:0 0 0.4rem;">Thank you for your feedback</h4>
          <p style="margin:0;">This form is for demonstration purposes only &mdash; nothing was sent or stored.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Submit another response"):
        st.session_state["feedback_sent"] = False
        st.rerun()
else:
    with st.form("feedback_form", border=False):
        with st.container(border=True, key="form-group-feedback"):
            name = st.text_input("Name (optional)")
            email = st.text_input("Email (optional)")
            message = st.text_area("Message", placeholder="What would make this tool more useful to you?", height=120)
            submitted = st.form_submit_button("Send feedback", type="primary")

    if submitted:
        if not message.strip():
            st.warning("Please add a short message before sending.")
        else:
            st.session_state["feedback_sent"] = True
            st.rerun()

st.write("")
st.write("")
footer.render()
