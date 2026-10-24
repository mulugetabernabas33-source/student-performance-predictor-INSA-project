"""Streamlit app: early-warning predictor for a student's final grade (G3).

Run with:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import EARLY_WARNING_FEATURES, PASS_THRESHOLD  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"

EDUCATION_LEVELS = {
    0: "None",
    1: "Primary education (4th grade)",
    2: "5th to 9th grade",
    3: "Secondary education",
    4: "Higher education",
}
TRAVEL_TIMES = {
    1: "< 15 min",
    2: "15 to 30 min",
    3: "30 min to 1 hour",
    4: "> 1 hour",
}
STUDY_TIMES = {
    1: "< 2 hours",
    2: "2 to 5 hours",
    3: "5 to 10 hours",
    4: "> 10 hours",
}
FIVE_POINT_SCALE = {1: "1 - Very low", 2: "2 - Low", 3: "3 - Medium", 4: "4 - High", 5: "5 - Very high"}
YES_NO = {"yes": "Yes", "no": "No"}


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def render_sidebar_info() -> None:
    st.sidebar.title("About")
    st.sidebar.markdown(
        """
        This app predicts a student's **final grade** (G3, 0-20) for a
        math course using only information available **before any grades
        exist** — demographics, family background, study habits, and
        behavior.

        It's trained on the UCI
        [Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)
        dataset (395 students) using a Random Forest regressor.

        Because no grade history is used, accuracy is modest
        (R² ≈ 0.30, ~67% pass/fail accuracy) — this is a **screening
        signal**, not a certainty, intended to help flag students who may
        benefit from early support.
        """
    )


def render_form() -> dict:
    st.subheader("Student profile")

    col1, col2, col3 = st.columns(3)
    with col1:
        school = st.selectbox("School", ["GP", "MS"], help="GP = Gabriel Pereira, MS = Mousinho da Silveira")
        sex = st.selectbox("Sex", ["F", "M"])
        age = st.slider("Age", 15, 22, 17)
        address = st.selectbox("Home address type", ["U", "R"], format_func=lambda x: "Urban" if x == "U" else "Rural")
        famsize = st.selectbox("Family size", ["LE3", "GT3"], format_func=lambda x: "3 or fewer" if x == "LE3" else "More than 3")
        pstatus = st.selectbox("Parents' cohabitation status", ["T", "A"], format_func=lambda x: "Living together" if x == "T" else "Apart")

    with col2:
        medu = st.selectbox("Mother's education", list(EDUCATION_LEVELS), format_func=lambda x: EDUCATION_LEVELS[x], index=2)
        fedu = st.selectbox("Father's education", list(EDUCATION_LEVELS), format_func=lambda x: EDUCATION_LEVELS[x], index=2)
        mjob = st.selectbox("Mother's job", ["at_home", "health", "other", "services", "teacher"])
        fjob = st.selectbox("Father's job", ["at_home", "health", "other", "services", "teacher"])
        reason = st.selectbox("Reason for choosing this school", ["course", "home", "reputation", "other"])
        guardian = st.selectbox("Guardian", ["mother", "father", "other"])

    with col3:
        traveltime = st.selectbox("Home-to-school travel time", list(TRAVEL_TIMES), format_func=lambda x: TRAVEL_TIMES[x], index=0)
        studytime = st.selectbox("Weekly study time", list(STUDY_TIMES), format_func=lambda x: STUDY_TIMES[x], index=1)
        failures = st.slider("Past class failures", 0, 4, 0)
        absences = st.slider("School absences (so far)", 0, 93, 4)
        health = st.selectbox("Current health status", list(FIVE_POINT_SCALE), format_func=lambda x: FIVE_POINT_SCALE[x], index=2)

    st.subheader("Support and lifestyle")
    col4, col5, col6 = st.columns(3)
    with col4:
        schoolsup = st.selectbox("Extra educational support", ["no", "yes"], format_func=lambda x: YES_NO[x])
        famsup = st.selectbox("Family educational support", ["no", "yes"], format_func=lambda x: YES_NO[x], index=1)
        paid = st.selectbox("Extra paid classes", ["no", "yes"], format_func=lambda x: YES_NO[x])
        activities = st.selectbox("Extra-curricular activities", ["no", "yes"], format_func=lambda x: YES_NO[x])
        nursery = st.selectbox("Attended nursery school", ["no", "yes"], format_func=lambda x: YES_NO[x], index=1)
    with col5:
        higher = st.selectbox("Wants higher education", ["no", "yes"], format_func=lambda x: YES_NO[x], index=1)
        internet = st.selectbox("Internet access at home", ["no", "yes"], format_func=lambda x: YES_NO[x], index=1)
        romantic = st.selectbox("In a romantic relationship", ["no", "yes"], format_func=lambda x: YES_NO[x])
        famrel = st.select_slider("Quality of family relationships", options=list(FIVE_POINT_SCALE), format_func=lambda x: FIVE_POINT_SCALE[x], value=4)
    with col6:
        freetime = st.select_slider("Free time after school", options=list(FIVE_POINT_SCALE), format_func=lambda x: FIVE_POINT_SCALE[x], value=3)
        goout = st.select_slider("Going out with friends", options=list(FIVE_POINT_SCALE), format_func=lambda x: FIVE_POINT_SCALE[x], value=3)
        dalc = st.select_slider("Workday alcohol consumption", options=list(FIVE_POINT_SCALE), format_func=lambda x: FIVE_POINT_SCALE[x], value=1)
        walc = st.select_slider("Weekend alcohol consumption", options=list(FIVE_POINT_SCALE), format_func=lambda x: FIVE_POINT_SCALE[x], value=1)

    return {
        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": pstatus,
        "Medu": medu,
        "Fedu": fedu,
        "Mjob": mjob,
        "Fjob": fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": dalc,
        "Walc": walc,
        "health": health,
        "absences": absences,
    }


def render_prediction(model, inputs: dict) -> None:
    X = pd.DataFrame([inputs])[EARLY_WARNING_FEATURES]
    predicted_grade = float(model.predict(X)[0])
    predicted_grade_clamped = max(0.0, min(20.0, predicted_grade))
    will_pass = predicted_grade >= PASS_THRESHOLD

    st.subheader("Prediction")
    col1, col2 = st.columns(2)
    col1.metric("Predicted final grade (G3)", f"{predicted_grade_clamped:.1f} / 20")
    col2.metric("Predicted outcome", "Pass" if will_pass else "At risk of failing")
    st.progress(predicted_grade_clamped / 20)

    if not will_pass:
        st.warning(
            "This profile suggests a risk of failing. Consider flagging for "
            "early tutoring, attendance support, or a check-in with a "
            "school counselor."
        )
    else:
        st.success("This profile suggests the student is likely to pass.")

    st.caption(
        "Prediction from an early-warning model trained without any grade "
        "history — treat it as a screening signal, not a certainty."
    )


def main() -> None:
    st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")
    st.title("🎓 Student Performance Predictor")
    st.markdown(
        "Estimate a student's final math grade **before any grades exist**, "
        "using demographic, family, and behavioral data."
    )

    render_sidebar_info()
    model = load_model()

    if model is None:
        st.error(
            f"No trained model found at `{MODEL_PATH.relative_to(PROJECT_ROOT)}`. "
            "Run `python -m src.train` first."
        )
        return

    inputs = render_form()

    if st.button("Predict final grade", type="primary"):
        render_prediction(model, inputs)


if __name__ == "__main__":
    main()
