"""Model Performance page: leaderboards, ROC curves, confusion matrices,
and the Scenario A vs. Scenario B comparison that explains the leakage
decision behind which model gets deployed.
"""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
for _p in (str(APP_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from components import footer, section_header  # noqa: E402
from utils.data_loader import (  # noqa: E402
    get_scenario_classification,
    get_scenario_regression,
    naive_baseline_accuracy,
)
from utils.icons import svg  # noqa: E402
from utils.theme import load_theme  # noqa: E402

load_theme()

PLOTLY_FONT = {"color": "#e6eaf2", "family": "Inter, sans-serif"}
GRID_COLOR = "rgba(148,163,184,0.12)"


def leaderboard_table(df, score_col: str, score_label: str) -> str:
    """Render a cross-validation leaderboard as a styled HTML table."""
    best_idx = df[score_col].idxmax()
    rows = []
    for i, row in df.iterrows():
        is_best = i == best_idx
        row_style = "background:rgba(34,197,94,0.08);" if is_best else ""
        badge = (
            f'<span class="chip" style="border-color:var(--pass);color:var(--pass);">{svg("award", 13)} Best</span>'
            if is_best
            else ""
        )
        model_name = str(row["model"]).replace("_", " ").title()
        rows.append(
            f'<tr style="{row_style}"><td style="padding:0.6rem 0.75rem;">{model_name} {badge}</td>'
            f'<td style="padding:0.6rem 0.75rem; text-align:right;">{row[score_col]:.3f}</td></tr>'
        )
    return f"""
    <table style="width:100%; border-collapse:collapse; font-size:0.92rem;">
      <thead>
        <tr style="border-bottom:1px solid var(--border-strong); color:var(--text-muted); text-align:left;">
          <th style="padding:0.5rem 0.75rem;">Model</th>
          <th style="padding:0.5rem 0.75rem; text-align:right;">{score_label}</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


section_header.render(
    "Model performance",
    "How the models compare",
    "Cross-validated leaderboards, ROC curves, and a confusion matrix for both scenarios "
    "-- plus why only one of them is deployed.",
)

st.markdown(
    f"""
    <div class="callout">
      {svg("alert", 20)}
      <div>
        <strong>The leakage decision.</strong> Scenario B includes the first and second period
        grades (G1, G2) as features. Because grades within a term are highly autocorrelated,
        this inflates accuracy dramatically &mdash; but it isn't an early-warning signal, since by
        the time G1 exists, a third of the term has already happened. <strong>Only Scenario A is deployed
        in the Predictor.</strong> Scenario B is shown here for comparison only.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

scenario_a = get_scenario_classification("early_warning")
scenario_b = get_scenario_classification("full")
reg_a = get_scenario_regression("early_warning")
reg_b = get_scenario_regression("full")
baseline_acc = naive_baseline_accuracy("early_warning")

st.write("")
col_a, col_b = st.columns(2, gap="large")
scenario_cards = [
    (col_a, "Scenario A", "Early-warning (no prior grades)", scenario_a, True),
    (col_b, "Scenario B", "Full history (includes G1, G2)", scenario_b, False),
]
for col, name, desc, scenario, is_deployed in scenario_cards:
    with col:
        chip = (
            '<span class="chip" style="border-color:var(--accent-cyan); color:var(--accent-cyan);">Deployed</span>'
            if is_deployed
            else '<span class="chip">Comparison only</span>'
        )
        m = scenario["metrics"]
        best_model = scenario["leaderboard"].iloc[0]["model"].replace("_", " ").title()
        st.markdown(
            f"""
            <div class="card">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                <div><strong>{name}</strong><div style="font-size:0.82rem; color:var(--text-muted);">{desc}</div></div>
                {chip}
              </div>
              <div style="font-size:0.85rem; color:var(--text-muted); margin-bottom:0.5rem;">Best model: {best_model}</div>
              <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:0.5rem; text-align:center;">
                <div><div class="stat-value" style="font-size:1.4rem;">{m['accuracy']*100:.1f}%</div><div class="stat-label">Accuracy</div></div>
                <div><div class="stat-value" style="font-size:1.4rem;">{m['recall_fail']*100:.1f}%</div><div class="stat-label">Recall (fail)</div></div>
                <div><div class="stat-value" style="font-size:1.4rem;">{m['roc_auc']:.2f}</div><div class="stat-label">ROC-AUC</div></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    f"""
    <p style="margin-top:0.75rem; font-size:0.85rem;">
      For reference, a naive "always predict pass" baseline reaches <strong>{baseline_acc*100:.1f}%</strong>
      accuracy on the same Scenario A test split.
    </p>
    """,
    unsafe_allow_html=True,
)

section_header.render("Cross-validation", "Classifier leaderboards", center=False)
lb_col_a, lb_col_b = st.columns(2, gap="large")
with lb_col_a:
    st.markdown(
        f'<div class="card"><div class="form-group-title">{svg("target", 18)} Scenario A</div>'
        + leaderboard_table(scenario_a["leaderboard"], "cv_roc_auc_mean", "CV ROC-AUC")
        + "</div>",
        unsafe_allow_html=True,
    )
with lb_col_b:
    st.markdown(
        f'<div class="card"><div class="form-group-title">{svg("target", 18)} Scenario B</div>'
        + leaderboard_table(scenario_b["leaderboard"], "cv_roc_auc_mean", "CV ROC-AUC")
        + "</div>",
        unsafe_allow_html=True,
    )

section_header.render("Discrimination", "ROC curves", center=False)
roc_fig = go.Figure()
roc_fig.add_trace(
    go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#64748b"), name="Random chance", showlegend=True)
)
roc_fig.add_trace(
    go.Scatter(
        x=scenario_a["roc"].fpr, y=scenario_a["roc"].tpr, mode="lines",
        line=dict(color="#22d3ee", width=3),
        name=f"Scenario A (AUC = {scenario_a['roc'].auc:.2f})",
    )
)
roc_fig.add_trace(
    go.Scatter(
        x=scenario_b["roc"].fpr, y=scenario_b["roc"].tpr, mode="lines",
        line=dict(color="#6366f1", width=3),
        name=f"Scenario B (AUC = {scenario_b['roc'].auc:.2f})",
    )
)
roc_fig.update_layout(
    height=420,
    margin=dict(l=10, r=10, t=30, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=PLOTLY_FONT,
    xaxis=dict(title="False positive rate", gridcolor=GRID_COLOR, range=[0, 1]),
    yaxis=dict(title="True positive rate", gridcolor=GRID_COLOR, range=[0, 1]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
st.plotly_chart(roc_fig, use_container_width=True, config={"displayModeBar": False})

section_header.render("Error analysis", "Confusion matrices (test set)", center=False)
cm_col_a, cm_col_b = st.columns(2, gap="large")


def confusion_heatmap(cm, title: str):
    labels = ["Fail", "Pass"]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale=[[0, "#121a2e"], [1, "#6366f1"]],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 18, "color": "#e6eaf2"},
            showscale=False,
        )
    )
    fig.update_layout(
        title=dict(text=title, font={"size": 14, "color": "#94a3b8"}),
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=PLOTLY_FONT,
        xaxis=dict(title="Predicted", side="bottom"),
        yaxis=dict(title="Actual", autorange="reversed"),
    )
    return fig


with cm_col_a:
    st.plotly_chart(confusion_heatmap(scenario_a["confusion_matrix"], "Scenario A"), use_container_width=True, config={"displayModeBar": False})
with cm_col_b:
    st.plotly_chart(confusion_heatmap(scenario_b["confusion_matrix"], "Scenario B"), use_container_width=True, config={"displayModeBar": False})

section_header.render("Alternate framing", "Regression: predicting the numeric grade", center=False)
st.markdown(
    """
    <p style="font-size:0.9rem;">
      The same two scenarios were also modeled as regression &mdash; predicting the continuous
      final grade (G3, 0-20) instead of a binary label. Random Forest was again the strongest
      model in both scenarios.
    </p>
    """,
    unsafe_allow_html=True,
)
reg_table = f"""
<table style="width:100%; border-collapse:collapse; font-size:0.92rem;">
  <thead>
    <tr style="border-bottom:1px solid var(--border-strong); color:var(--text-muted); text-align:left;">
      <th style="padding:0.5rem 0.75rem;">Scenario</th>
      <th style="padding:0.5rem 0.75rem; text-align:right;">R&sup2;</th>
      <th style="padding:0.5rem 0.75rem; text-align:right;">RMSE</th>
      <th style="padding:0.5rem 0.75rem; text-align:right;">MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="padding:0.6rem 0.75rem;">A &mdash; Early-warning</td>
        <td style="padding:0.6rem 0.75rem; text-align:right;">{reg_a['metrics']['r2']:.2f}</td>
        <td style="padding:0.6rem 0.75rem; text-align:right;">{reg_a['metrics']['rmse']:.2f}</td>
        <td style="padding:0.6rem 0.75rem; text-align:right;">{reg_a['metrics']['mae']:.2f}</td></tr>
    <tr style="background:rgba(99,102,241,0.08);"><td style="padding:0.6rem 0.75rem;">B &mdash; Full (+ G1, G2)</td>
        <td style="padding:0.6rem 0.75rem; text-align:right;">{reg_b['metrics']['r2']:.2f}</td>
        <td style="padding:0.6rem 0.75rem; text-align:right;">{reg_b['metrics']['rmse']:.2f}</td>
        <td style="padding:0.6rem 0.75rem; text-align:right;">{reg_b['metrics']['mae']:.2f}</td></tr>
  </tbody>
</table>
"""
st.markdown(f'<div class="card">{reg_table}</div>', unsafe_allow_html=True)

st.write("")
st.write("")
footer.render()
