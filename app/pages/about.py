"""About page: project story, methodology timeline, tech stack, ethics."""

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

st.markdown(
    '<div class="card card-gradient-border" style="display:flex; justify-content:space-between; '
    'align-items:center; flex-wrap:wrap; gap:1rem; margin-bottom:1.5rem;">'
    '<div><strong style="font-size:1.1rem;">Bernabas Mulugeta Genanaw</strong>'
    '<div style="color:var(--text-muted); font-size:0.88rem; margin-top:0.2rem;">Student ID: CTC-5711-26</div></div>'
    '<span class="chip">Academic project</span>'
    "</div>",
    unsafe_allow_html=True,
)

section_header.render("About", "The story behind this project", center=False)

st.markdown(
    """
    <div class="card">
      <p style="margin:0;">
        Schools often only discover a struggling student once grades are already low &mdash;
        by which point a term is well underway. This project asks a narrower question: how much
        can be predicted using only what's known about a student <em>before</em> any grades exist
        &mdash; their background, habits, and school environment &mdash; and is that signal strong
        enough to be useful as an early screen?
      </p>
      <p style="margin-top:0.75rem; margin-bottom:0;">
        The honest answer, explored in depth on the <strong>Model Performance</strong> page, is:
        partially. The resulting tool is built to be transparent about that limit rather than
        oversell its accuracy.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

section_header.render("Dataset", "Source and citation", center=False)
st.markdown(
    """
    <div class="card">
      <p style="margin:0;">
        <a href="https://archive.ics.uci.edu/dataset/320/student+performance" target="_blank">
        UCI Machine Learning Repository &mdash; Student Performance Data Set</a> (Math course subset):
        395 students at two Portuguese secondary schools, 30 demographic / family / social / school
        features plus three grade columns.
      </p>
      <p style="margin-top:0.75rem; margin-bottom:0; font-size:0.85rem; color:var(--text-faint);">
        P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance.
        In A. Brito and J. Teixeira Eds., <em>Proceedings of 5th FUture BUsiness TEChnology Conference
        (FUBUTEC 2008)</em> pp. 5-12, Porto, Portugal, April 2008, EUROSIS, ISBN 978-9077381-39-7.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

section_header.render("Methodology", "How this was built", center=False)

timeline_steps = [
    ("01", "Data", "Downloaded the official UCI archive, verified no missing values or duplicates, documented every attribute."),
    ("02", "Exploration", "Distributions, correlations, and targeted plots (absences, study time) to find what's worth modeling."),
    ("03", "Pipeline", "A single scikit-learn ColumnTransformer + Pipeline so preprocessing is identical at train and inference time."),
    ("04", "Model comparison", "Cross-validated Linear/Ridge/Random Forest/Gradient Boosting (regression) and Logistic Regression/Random Forest/Gradient Boosting (classification) across two scenarios."),
    ("05", "Evaluation", "RMSE/MAE/R² for regression; accuracy, fail-class recall, and ROC-AUC for classification, checked against a naive baseline."),
    ("06", "Explainability", "SHAP values explain individual predictions in the Predictor tool, not just global feature importance."),
    ("07", "This app", "A Streamlit interface so the model is usable without touching code."),
]
timeline_items = []
for i, (num, title, body) in enumerate(timeline_steps):
    delay = f"{i * 0.08:.2f}s"
    timeline_items.append(
        f'<div class="timeline-item" style="animation-delay:{delay};">'
        f'<div class="timeline-meta">Step {num}</div>'
        f"<h4>{title}</h4>"
        f"<p>{body}</p>"
        f"</div>"
    )
# Built as a single line (no blank lines, no leading indentation) because
# Streamlit's markdown parser drops into literal code-block rendering for
# any content after a blank line breaks an in-progress raw HTML block.
timeline_html = '<div class="timeline">' + "".join(timeline_items) + "</div>"
st.markdown(timeline_html, unsafe_allow_html=True)

section_header.render("Tech stack", "Built with", center=False)
stack = [
    "Python", "pandas", "NumPy", "scikit-learn", "SHAP", "Plotly",
    "Streamlit", "Jupyter", "pytest", "matplotlib", "seaborn",
]
chips_html = "".join(f'<span class="chip">{item}</span>' for item in stack)
st.markdown(f'<div class="card">{chips_html}</div>', unsafe_allow_html=True)

section_header.render("Responsible use", "Limitations and ethics", center=False)
limit_col, ethics_col = st.columns(2, gap="large")
with limit_col:
    st.markdown(
        f"""
        <div class="card">
          <div class="form-group-title">{svg("alert", 20)} Limitations</div>
          <ul style="padding-left:1.1rem; font-size:0.9rem; color:var(--text-muted);">
            <li>Small dataset: 395 students from two schools in one region.</li>
            <li>Modest early-warning accuracy (recall on the fail class is ~46%).</li>
            <li>Correlation, not causation &mdash; feature importance should not be read as a causal lever.</li>
            <li>Several features (alcohol use, family relationship quality) are self-reported.</li>
            <li>Trained only on the Math course; not validated on the Portuguese course dataset.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
with ethics_col:
    st.markdown(
        f"""
        <div class="card">
          <div class="form-group-title">{svg("shield", 20)} Fairness and bias</div>
          <ul style="padding-left:1.1rem; font-size:0.9rem; color:var(--text-muted);">
            <li>The model uses sex, family structure, and parental background as inputs &mdash; features
                that can encode societal bias rather than a student's own effort or ability.</li>
            <li>A prediction should inform a supportive conversation, never a denial of opportunity
                or resources.</li>
            <li>This dataset reflects two schools in Portugal circa 2008; outputs may not transfer
                to other populations, curricula, or grading systems.</li>
            <li>No individual student data is collected, stored, or transmitted by this app.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
st.write("")
footer.render()
