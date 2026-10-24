"""Stat cards, including a JS count-up variant for the Home page hero stats."""

import json
import uuid

import streamlit as st

from utils.icons import svg


def animated_stat_row(stats: list[dict]) -> None:
    """Render a row of count-up animated stat cards.

    Each stat dict supports: ``icon`` (icon key), ``value`` (numeric
    target), ``decimals`` (int, default 0), ``prefix``/``suffix`` (str),
    and ``label``. The cards (including their SVG icons) are rendered via
    ``st.markdown`` -- ``st.html``'s sanitizer strips raw ``<svg>`` tags,
    which would leave the icons blank. A separate script-only ``st.html``
    call (no HTML to sanitize) drives the count-up animation by id,
    falling back to the final value immediately for reduced-motion users.

    Args:
        stats: List of stat dicts, rendered left-to-right in one row.
    """
    uid = uuid.uuid4().hex[:8]
    card_chunks = []
    for i, s in enumerate(stats):
        counter_id = f"stat-{uid}-{i}"
        card_chunks.append(
            f'<div class="card card-glass stat-card fade-in-up delay-{min(i + 1, 4)}">'
            f'<div class="stat-icon">{svg(s["icon"], 20)}</div>'
            f'<div class="stat-value"><span id="{counter_id}">'
            f'{s.get("prefix", "")}0{s.get("suffix", "")}</span></div>'
            f'<div class="stat-label">{s["label"]}</div>'
            f"</div>"
        )
    # Single-line, blank-line-free HTML: a blank line inside an in-progress
    # raw HTML block makes Streamlit's markdown parser drop out of HTML
    # mode, which would otherwise leave every card after the first
    # rendering as literal, unstyled text.
    grid_html = (
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1rem;">'
        + "".join(card_chunks)
        + "</div>"
    )
    st.markdown(grid_html, unsafe_allow_html=True)

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
    script = f"""<script>
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
    </script>"""
    st.html(script, unsafe_allow_javascript=True)
