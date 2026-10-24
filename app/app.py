"""Entry point: multipage routing for the Student Performance Predictor app.

Run with:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon=":material/school:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

pages = [
    st.Page("pages/home.py", title="Home", icon=":material/home:", default=True, url_path="home"),
    st.Page("pages/predictor.py", title="Predictor", icon=":material/psychology:", url_path="predictor"),
    st.Page("pages/insights.py", title="Insights", icon=":material/bar_chart:", url_path="insights"),
    st.Page(
        "pages/model_performance.py",
        title="Model Performance",
        icon=":material/monitoring:",
        url_path="model-performance",
    ),
    st.Page("pages/about.py", title="About", icon=":material/info:", url_path="about"),
]

navigation = st.navigation(pages, position="top")
navigation.run()
