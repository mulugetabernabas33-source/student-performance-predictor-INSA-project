"""Predictor page: the main early-warning tool."""

import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
for _p in (str(APP_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from components import footer, section_header  # noqa: E402
from utils.data_loader import load_classifier_pipeline, load_regressor_pipeline  # noqa: E402
from utils.form_options import (  # noqa: E402
    EDUCATION_LEVELS,
    FIVE_POINT_SCALE,
    GENERIC_ADVICE,
    GUARDIANS,
    JOB_LABELS,
    JOBS,
    REASONS,
    RISK_FACTOR_ADVICE,
    STUDY_TIMES,
    TRAVEL_TIMES,
    YES_NO,
)
from utils.icons import svg  # noqa: E402
from utils.theme import load_theme  # noqa: E402
from src.evaluate_classification import explain_prediction, predict_proba_single  # noqa: E402
from src.preprocess import EARLY_WARNING_FEATURES, PASS_THRESHOLD  # noqa: E402

load_theme()

section_header.render(
    "Early-warning tool",
    "Predict a student's outcome",
    "Fill in a student profile using only information available before any grades exist. "
    "Nothing here is stored or sent anywhere.",
)

st.markdown(
    f"""
    <div class="callout">
      {svg("alert", 20)}
      <div>
        <strong>For early support, not for judging students.</strong>
        This estimate is a screening signal to help prioritize attention, built on a small historical
        dataset. It should never be the sole basis for a decision about a specific student &mdash;
        always combine it with teacher judgment.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

classifier = load_classifier_pipeline()
regressor = load_regressor_pipeline()

if classifier is None:
    st.error(
        "No trained classifier found at `models/best_classifier.joblib`. "
        "Run `python -m src.classify` first, then reload this page."
    )
    st.stop()

with st.form("predictor_form", border=False):
    col_school, col_academic = st.columns(2)

    with col_school:
        with st.container(border=True, key="form-group-school"):
            st.markdown(
                f'<div class="form-group-title">{svg("home", 20)} School &amp; demographics</div>',
                unsafe_allow_html=True,
            )
            school = st.selectbox("School", ["GP", "MS"], help="GP = Gabriel Pereira, MS = Mousinho da Silveira")
            sex = st.selectbox("Sex", ["F", "M"])
            age = st.slider("Age", 15, 22, 17)
            address = st.selectbox("Home address", ["U", "R"], format_func=lambda x: "Urban" if x == "U" else "Rural")
            reason = st.selectbox("Reason for choosing school", list(REASONS), format_func=lambda x: REASONS[x])
            traveltime = st.select_slider("Home-to-school travel time", options=list(TRAVEL_TIMES), format_func=lambda x: TRAVEL_TIMES[x], value=1)
            absences = st.slider("School absences so far", 0, 93, 4)

    with col_academic:
        with st.container(border=True, key="form-group-academic"):
            st.markdown(
                f'<div class="form-group-title">{svg("book", 20)} Academic</div>',
                unsafe_allow_html=True,
            )
            studytime = st.select_slider("Weekly study time", options=list(STUDY_TIMES), format_func=lambda x: STUDY_TIMES[x], value=2)
            failures = st.slider("Past class failures", 0, 4, 0)
            schoolsup = st.selectbox("Extra educational support (school)", ["no", "yes"], format_func=lambda x: YES_NO[x])
            paid = st.selectbox("Extra paid classes", ["no", "yes"], format_func=lambda x: YES_NO[x])
            higher = st.selectbox("Wants higher education", ["no", "yes"], format_func=lambda x: YES_NO[x], index=1)
            activities = st.selectbox("Extra-curricular activities", ["no", "yes"], format_func=lambda x: YES_NO[x])

    col_family, col_lifestyle = st.columns(2)

    with col_family:
        with st.container(border=True, key="form-group-family"):
            st.markdown(
                f'<div class="form-group-title">{svg("users", 20)} Family</div>',
                unsafe_allow_html=True,
            )
            famsize = st.selectbox("Family size", ["LE3", "GT3"], format_func=lambda x: "3 or fewer" if x == "LE3" else "More than 3")
            pstatus = st.selectbox("Parents' cohabitation", ["T", "A"], format_func=lambda x: "Living together" if x == "T" else "Apart")
            medu = st.select_slider("Mother's education", options=list(EDUCATION_LEVELS), format_func=lambda x: EDUCATION_LEVELS[x], value=2)
            fedu = st.select_slider("Father's education", options=list(EDUCATION_LEVELS), format_func=lambda x: EDUCATION_LEVELS[x], value=2)
            mjob = st.selectbox("Mother's job", JOBS, format_func=lambda x: JOB_LABELS[x])
            fjob = st.selectbox("Father's job", JOBS, format_func=lambda x: JOB_LABELS[x])
            guardian = st.selectbox("Guardian", list(GUARDIANS), format_func=lambda x: GUARDIANS[x])
            famsup = st.selectbox("Family educational support", ["no", "yes"], format_func=lambda x: YES_NO[x], index=1)
            famrel = st.select_slider("Quality of family relationships", options=list(FIVE_POINT_SCALE), format_func=lambda x: FIVE_POINT_SCALE[x], value=4)

    with col_lifestyle:
        with st.container(border=True, key="form-group-lifestyle"):
            st.markdown(
                f'<div class="form-group-title">{svg("heart", 20)} Lifestyle</div>',
                unsafe_allow_html=True,
            )
            freetime = st.select_slider("Free time after school", options=list(FIVE_POINT_SCALE), format_func=lambda x: FIVE_POINT_SCALE[x], value=3)
            goout = st.select_slider("Going out with friends", options=list(FIVE_POINT_SCALE), format_func=lambda x: FIVE_POINT_SCALE[x], value=3)
            dalc = st.select_slider("Workday alcohol consumption", options=list(FIVE_POINT_SCALE), format_func=lambda x: FIVE_POINT_SCALE[x], value=1)
            walc = st.select_slider("Weekend alcohol consumption", options=list(FIVE_POINT_SCALE), format_func=lambda x: FIVE_POINT_SCALE[x], value=1)
            health = st.select_slider("Current health status", options=list(FIVE_POINT_SCALE), format_func=lambda x: FIVE_POINT_SCALE[x], value=3)
            romantic = st.selectbox("In a romantic relationship", ["no", "yes"], format_func=lambda x: YES_NO[x])
            internet = st.selectbox("Internet access at home", ["no", "yes"], format_func=lambda x: YES_NO[x], index=1)
            nursery = st.selectbox("Attended nursery school", ["no", "yes"], format_func=lambda x: YES_NO[x], index=1)

    st.write("")
    submitted = st.form_submit_button("Predict outcome", type="primary", use_container_width=True)

if submitted:
    inputs = {
        "school": school, "sex": sex, "age": age, "address": address, "famsize": famsize,
        "Pstatus": pstatus, "Medu": medu, "Fedu": fedu, "Mjob": mjob, "Fjob": fjob,
        "reason": reason, "guardian": guardian, "traveltime": traveltime, "studytime": studytime,
        "failures": failures, "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
        "activities": activities, "nursery": nursery, "higher": higher, "internet": internet,
        "romantic": romantic, "famrel": famrel, "freetime": freetime, "goout": goout,
        "Dalc": dalc, "Walc": walc, "health": health, "absences": absences,
    }
    st.session_state["predictor_inputs"] = inputs

if "predictor_inputs" in st.session_state:
    inputs = st.session_state["predictor_inputs"]

    with st.spinner("Analyzing profile..."):
        time.sleep(0.6)
        try:
            X_row = pd.DataFrame([inputs])[EARLY_WARNING_FEATURES]
            pass_probability = predict_proba_single(classifier, X_row)
            predicted_grade = float(regressor.predict(X_row)[0]) if regressor is not None else None
            shap_df = explain_prediction(classifier, X_row, top_n=8)
            prediction_error = None
        except Exception:
            prediction_error = (
                "Something went wrong while generating this prediction. Please check the "
                "inputs and try again -- if the problem persists, the model files may need "
                "to be retrained (see the README)."
            )
            pass_probability = None

    if prediction_error:
        st.error(prediction_error)
    else:
        will_pass = pass_probability >= 0.5
        badge_class = "badge-pass" if will_pass else "badge-risk"
        badge_text = "Likely to pass" if will_pass else "At risk of failing"
        badge_icon = "check-circle" if will_pass else "x-circle"

        section_header.render("Result", "Prediction", center=False)

        result_col, shap_col = st.columns([1, 1.2], gap="large")

        with result_col:
            gauge_color = "#22c55e" if pass_probability >= 0.65 else ("#f59e0b" if pass_probability >= 0.5 else "#ef4444")
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=round(pass_probability * 100, 1),
                    number={"suffix": "%", "font": {"size": 44, "color": "#e6eaf2"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#64748b", "tickfont": {"color": "#94a3b8"}},
                        "bar": {"color": gauge_color, "thickness": 0.28},
                        "bgcolor": "rgba(0,0,0,0)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 50], "color": "rgba(239,68,68,0.15)"},
                            {"range": [50, 65], "color": "rgba(245,158,11,0.15)"},
                            {"range": [65, 100], "color": "rgba(34,197,94,0.15)"},
                        ],
                        "threshold": {"line": {"color": "#e6eaf2", "width": 2}, "thickness": 0.8, "value": 50},
                    },
                )
            )
            fig.update_layout(
                height=260,
                margin=dict(l=20, r=20, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e6eaf2", "family": "Inter, sans-serif"},
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown(
                f"""
                <div style="text-align:center;">
                  <span class="badge {badge_class}">{svg(badge_icon, 18)} {badge_text}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if predicted_grade is not None:
                clamped = max(0.0, min(20.0, predicted_grade))
                st.markdown(
                    f"""
                    <div style="text-align:center; margin-top:1rem; color:var(--text-muted); font-size:0.9rem;">
                      Estimated final grade: <strong style="color:var(--text);">{clamped:.1f} / 20</strong>
                      &nbsp;(pass threshold: {PASS_THRESHOLD})
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with shap_col:
            st.markdown("##### Why this prediction")
            shap_sorted = shap_df.sort_values("shap_value")
            colors = ["#ef4444" if v < 0 else "#22c55e" for v in shap_sorted["shap_value"]]
            bar_fig = go.Figure(
                go.Bar(
                    x=shap_sorted["shap_value"],
                    y=shap_sorted["feature"],
                    orientation="h",
                    marker_color=colors,
                )
            )
            bar_fig.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e6eaf2", "family": "Inter, sans-serif"},
                xaxis={"title": "Contribution (toward pass / toward fail)", "gridcolor": "rgba(148,163,184,0.12)", "zerolinecolor": "rgba(148,163,184,0.3)"},
                yaxis={"gridcolor": "rgba(148,163,184,0.08)"},
            )
            st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("##### Suggested focus areas")
        risk_factors = shap_df[shap_df["shap_value"] < 0].sort_values("shap_value").head(3)
        if risk_factors.empty:
            st.markdown(
                f"""
                <div class="callout callout-info">{svg("check-circle", 20)}
                  <div>No strong risk factors were identified for this profile.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            for _, row in risk_factors.iterrows():
                advice = RISK_FACTOR_ADVICE.get(row["feature"], GENERIC_ADVICE)
                st.markdown(
                    f"""
                    <div class="card" style="margin-bottom:0.75rem;">
                      <strong>{row['feature']}</strong>
                      <p style="margin-top:0.35rem;">{advice}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

st.write("")
footer.render()
