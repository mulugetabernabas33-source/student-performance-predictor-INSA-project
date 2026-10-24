"""Smoke tests for the Streamlit app: every page renders without exceptions,
and the Predictor page's end-to-end submit flow produces a result.

Uses Streamlit's built-in AppTest harness, which runs a page script in a
simulated session without needing a browser.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parent.parent / "app"

# Home uses st.page_link, which needs the real multipage registry from
# app.py to resolve -- it's covered separately by test_home_page_via_full_app.
PAGE_FILES = [
    "pages/predictor.py",
    "pages/insights.py",
    "pages/model_performance.py",
    "pages/about.py",
]


@pytest.mark.parametrize("page_file", PAGE_FILES)
def test_page_renders_without_exceptions(page_file):
    """Each page runs top-to-bottom with no uncaught exception."""
    at = AppTest.from_file(str(APP_DIR / page_file), default_timeout=60)
    at.run()
    assert not at.exception, f"{page_file} raised: {[e.value for e in at.exception]}"


def test_home_page_via_full_app():
    """Home (the default page) renders cleanly through the real navigation entry point."""
    at = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, f"Home raised: {[e.value for e in at.exception]}"


def test_predictor_form_submits_and_shows_result():
    """Submitting the Predictor form with default values renders a prediction."""
    at = AppTest.from_file(str(APP_DIR / "pages/predictor.py"), default_timeout=60)
    at.run()

    submit_buttons = [b for b in at.get("button") if "Predict" in (b.label or "")]
    assert submit_buttons, "Predict button not found"
    submit_buttons[0].click().run()

    assert not at.exception, f"Prediction raised: {[e.value for e in at.exception]}"
    assert len(at.get("plotly_chart")) >= 2, "Expected a gauge and a SHAP bar chart"

    badge_markdown = [m.value for m in at.markdown if "badge-pass" in (m.value or "") or "badge-risk" in (m.value or "")]
    assert badge_markdown, "Expected a PASS/AT-RISK badge to render"


def test_insights_filters_do_not_raise():
    """Changing an Insights filter reruns cleanly."""
    at = AppTest.from_file(str(APP_DIR / "pages/insights.py"), default_timeout=60)
    at.run()

    sex_filter = [s for s in at.get("selectbox") if s.label == "Sex"][0]
    sex_filter.set_value("F").run()
    assert not at.exception
