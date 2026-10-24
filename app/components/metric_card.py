"""Stat cards, including a JS count-up variant for the Home page hero stats."""

import json
import uuid

import streamlit as st

from utils.icons import svg


def static_metric_card(icon_name: str, value: str, label: str) -> str:
    """Build the HTML for a non-animated stat card.

    Args:
        icon_name: Key into :mod:`app.utils.icons`.
        value: Pre-formatted value string (e.g. ``"395"``, ``"88.6%"``).
        label: Caption under the value.

    Returns:
        An HTML string for the card (caller embeds it in a markdown block).
    """
    return f"""
    <div class="card stat-card">
      <div class="stat-icon">{svg(icon_name, 20)}</div>
      <div class="stat-value">{value}</div>
      <div class="stat-label">{label}</div>
    </div>
    """


def animated_stat_row(stats: list[dict]) -> None:
    """Render a row of count-up animated stat cards.

    Each stat dict supports: ``icon`` (icon key), ``value`` (numeric
    target), ``decimals`` (int, default 0), ``prefix``/``suffix`` (str),
    and ``label``. Uses ``st.html`` with JavaScript enabled so the counters
    animate on load; falls back to showing the final value immediately if
    the viewer prefers reduced motion.

    Args:
        stats: List of stat dicts, rendered left-to-right in one row.
    """
    uid = uuid.uuid4().hex[:8]
    cards_html = []
    for i, s in enumerate(stats):
        counter_id = f"stat-{uid}-{i}"
        cards_html.append(
            f"""
            <div class="card stat-card fade-in-up delay-{min(i + 1, 4)}">
              <div class="stat-icon">{svg(s["icon"], 20)}</div>
              <div class="stat-value">
                <span id="{counter_id}">{s.get("prefix", "")}0{s.get("suffix", "")}</span>
              </div>
              <div class="stat-label">{s["label"]}</div>
            </div>
            """
        )

    targets = json.dumps(
        [
            {
                "id": f"stat-{uid}-{i}",
                "target": s["value"],
                "decimals": s.get("decimals", 0),
                "prefix": s.get("prefix", ""),
                "suffix": s.get("suffix", ""),
            }
            for i, s in enumerate(stats)
        ]
    )

    html = f"""
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1rem;">
      {''.join(cards_html)}
    </div>
    <script>
      (function() {{
        const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        const items = {targets};
        const duration = 1300;
        items.forEach(function(item) {{
          const el = document.getElementById(item.id);
          if (!el) return;
          if (reduceMotion) {{
            el.textContent = item.prefix + item.target.toFixed(item.decimals) + item.suffix;
            return;
          }}
          const start = performance.now();
          function tick(now) {{
            const p = Math.min((now - start) / duration, 1);
            const eased = 1 - Math.pow(1 - p, 3);
            const current = item.target * eased;
            el.textContent = item.prefix + current.toFixed(item.decimals) + item.suffix;
            if (p < 1) requestAnimationFrame(tick);
          }}
          requestAnimationFrame(tick);
        }});
      }})();
    </script>
    """
    st.html(html, unsafe_allow_javascript=True)
