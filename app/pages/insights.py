"""Insights page: interactive EDA dashboard with cross-filtering."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
for _p in (str(APP_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from components import footer, section_header  # noqa: E402
from utils.data_loader import load_dataset  # noqa: E402
from utils.form_options import STUDY_TIMES  # noqa: E402
from utils.theme import load_theme  # noqa: E402

load_theme()

PLOTLY_FONT = {"color": "#e6eaf2", "family": "Inter, sans-serif"}
GRID_COLOR = "rgba(148,163,184,0.12)"
TRANSITION = {"duration": 450, "easing": "cubic-in-out"}


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    """Apply the shared dark-theme layout to a Plotly figure."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=PLOTLY_FONT,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        transition=TRANSITION,
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    return fig


section_header.render(
    "Insights",
    "Explore the data behind the model",
    "Filter the dataset to see how grade outcomes shift across groups.",
)

df = load_dataset()

FILTER_KEYS = ["insights_sex_filter", "insights_school_filter", "insights_study_filter"]

filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1, 1, 1.4, 1])
with filter_col1:
    sex_filter = st.selectbox("Sex", ["All", "F", "M"], key="insights_sex_filter")
with filter_col2:
    school_filter = st.selectbox("School", ["All", "GP", "MS"], key="insights_school_filter")
with filter_col3:
    study_filter = st.selectbox(
        "Weekly study time", ["All"] + list(STUDY_TIMES.values()), key="insights_study_filter",
    )
with filter_col4:
    st.markdown("<div style='height:1.8rem;'></div>", unsafe_allow_html=True)
    reset = st.button("Reset filters", use_container_width=True)

if reset:
    for key in FILTER_KEYS:
        st.session_state.pop(key, None)
    st.rerun()

filtered = df.copy()
if sex_filter != "All":
    filtered = filtered[filtered["sex"] == sex_filter]
if school_filter != "All":
    filtered = filtered[filtered["school"] == school_filter]
if study_filter != "All":
    study_code = next(k for k, v in STUDY_TIMES.items() if v == study_filter)
    filtered = filtered[filtered["studytime"] == study_code]

st.markdown(
    f"<p style='font-size:0.85rem; color:var(--text-muted);'>Showing <strong style='color:var(--text);'>{len(filtered)}</strong> "
    f"of {len(df)} students.</p>",
    unsafe_allow_html=True,
)

if filtered.empty:
    st.warning("No students match this filter combination. Try widening your filters.")
    st.stop()

chart_col1, chart_col2 = st.columns(2, gap="large")

with chart_col1:
    st.markdown("##### Final grade distribution")
    hist_fig = px.histogram(
        filtered, x="G3", nbins=21, color_discrete_sequence=["#6366f1"],
        labels={"G3": "Final grade (0-20)"},
    )
    hist_fig.add_vline(x=10, line_dash="dash", line_color="#f59e0b", annotation_text="Pass threshold")
    hist_fig.update_layout(yaxis_title="Number of students")
    st.plotly_chart(style_fig(hist_fig), use_container_width=True, config={"displayModeBar": False})

with chart_col2:
    st.markdown("##### Final grade by weekly study time")
    box_fig = px.box(
        filtered, x="studytime", y="G3", color="studytime",
        color_discrete_sequence=px.colors.sequential.Tealgrn,
        labels={"studytime": "Study time code", "G3": "Final grade"},
    )
    box_fig.update_xaxes(tickmode="array", tickvals=list(STUDY_TIMES.keys()), ticktext=list(STUDY_TIMES.values()))
    box_fig.update_layout(showlegend=False)
    st.plotly_chart(style_fig(box_fig), use_container_width=True, config={"displayModeBar": False})

chart_col3, chart_col4 = st.columns(2, gap="large")

with chart_col3:
    st.markdown("##### Pass rate by absence band")
    bins = [-1, 0, 2, 5, 10, 100]
    labels = ["0", "1-2", "3-5", "6-10", "10+"]
    banded = filtered.copy()
    banded["absence_band"] = pd.cut(banded["absences"], bins=bins, labels=labels)
    pass_rate = banded.groupby("absence_band", observed=True)["pass"].mean().reindex(labels)
    bar_fig = px.bar(
        x=pass_rate.index, y=pass_rate.values,
        labels={"x": "Number of absences", "y": "Pass rate"},
        color=pass_rate.values, color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
    )
    bar_fig.update_layout(coloraxis_showscale=False, yaxis_tickformat=".0%")
    st.plotly_chart(style_fig(bar_fig), use_container_width=True, config={"displayModeBar": False})

with chart_col4:
    st.markdown("##### Correlation heatmap (numeric features)")
    numeric_cols = [c for c in filtered.select_dtypes(include="number").columns if c != "pass"]
    corr = filtered[numeric_cols].corr()
    # A diverging scale with a white midpoint (e.g. Plotly's default "RdBu")
    # reads as a large pale/white patch against this app's dark theme, so
    # the near-zero midpoint is the dark surface color instead of white.
    heat_fig = go.Figure(
        data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale=[[0, "#ef4444"], [0.5, "#121a2e"], [1, "#22d3ee"]],
            zmid=0, zmin=-1, zmax=1,
            hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
        )
    )
    heat_fig.update_layout(height=380)
    st.plotly_chart(style_fig(heat_fig, height=380), use_container_width=True, config={"displayModeBar": False})

st.write("")
st.write("")
footer.render()
