"""Home page: hero, headline stats, and a 3-step "how it works" overview."""

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
for _p in (str(APP_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from components import footer, hero, metric_card, section_header  # noqa: E402
from utils.data_loader import get_scenario_classification, load_dataset  # noqa: E402
from utils.icons import svg  # noqa: E402
from utils.theme import load_theme  # noqa: E402
from src.preprocess import EARLY_WARNING_FEATURES  # noqa: E402

load_theme()

hero.render(
    badge="Early-warning ML for educators",
    title_html='Know which students need support &mdash; <span class="gradient-text">before the first grade exists</span>.',
    subtitle=(
        "A machine learning tool that screens for academic risk using only demographic, "
        "family, and behavioral data &mdash; built on the UCI Student Performance dataset."
    ),
)

with st.container(key="hero-cta"):
    st.page_link("pages/predictor.py", label="Try the predictor", icon=":material/arrow_forward:")

dataset = load_dataset()
scenario_a = get_scenario_classification("early_warning")

metric_card.animated_stat_row(
    [
        {"icon": "users", "value": len(dataset), "label": "Students analyzed", "decimals": 0},
        {"icon": "award", "value": scenario_a["metrics"]["accuracy"] * 100, "label": "Model accuracy", "decimals": 1, "suffix": "%"},
        {"icon": "alert", "value": scenario_a["metrics"]["recall_fail"] * 100, "label": "Recall on fail class", "decimals": 1, "suffix": "%"},
        {"icon": "layers", "value": len(EARLY_WARNING_FEATURES), "label": "Behavioral features used", "decimals": 0},
    ]
)

st.markdown(
    f"""
    <div class="callout callout-info" style="margin-top:1.5rem;">
      {svg("info", 20)}
      <div>
        These numbers describe the deployed <strong>early-warning</strong> model, trained without
        any prior grade history &mdash; see <strong>Model Performance</strong> for the full comparison
        against a model that includes mid-term grades.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

section_header.render(
    "How it works",
    "From a quick form to an actionable signal",
    subtitle="Three steps, no grade history required.",
    center=True,
)

step_cols = st.columns(3, gap="medium")
steps = [
    {
        "icon": "home",
        "title": "Describe the student",
        "body": "Fill in a short form covering school, family background, study habits, and lifestyle &mdash; about two minutes.",
    },
    {
        "icon": "cpu",
        "title": "The model predicts",
        "body": "A Random Forest classifier, cross-validated for accuracy, returns a calibrated probability of passing.",
    },
    {
        "icon": "shield",
        "title": "Act on the signal",
        "body": "Review the top risk factors in plain language and prioritize attention before grades fall behind.",
    },
]
for col, step, i in zip(step_cols, steps, range(1, 4)):
    with col:
        st.markdown(
            f"""
            <div class="card step-card fade-in-up delay-{i}">
              <div class="step-number">{i}</div>
              <div class="icon-wrap">{svg(step["icon"], 22)}</div>
              <h4 style="margin:0 0 0.4rem;">{step["title"]}</h4>
              <p style="margin:0; font-size:0.92rem;">{step["body"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")
st.write("")
footer.render()
