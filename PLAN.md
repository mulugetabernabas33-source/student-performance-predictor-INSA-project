# Project Plan

**Author:** Bernabas Mulugeta Genanaw (Student ID: CTC-5711-26)

## Objective

Build an end-to-end, portfolio-quality ML project that predicts student academic performance, with a clear and honest narrative rather than a single headline accuracy number.

## Scope decisions

- **Dataset**: UCI Student Performance (Math course, 395 rows) — small enough to commit directly to the repo, well-documented, and widely used enough that results are easy to sanity-check against prior published work.
- **Target**: final grade `G3` (regression, 0–20), with a derived pass/fail label (`G3 >= 10`) for an interpretable secondary metric.
- **Two feature sets instead of one**: early in EDA it was clear that `G1`/`G2` (period grades) would dominate any model, making a single headline R² misleading about what's actually useful for early intervention. Comparing an "early-warning" set (no grade history) against a "full" set (with `G1`/`G2`) makes that trade-off explicit instead of hiding it.
- **Deployed model**: the early-warning model, since it's the only one of the two usable before a term's first grades exist — the actual use case this project targets.

## Build phases

1. **Data** — download the official UCI archive, verify there are no missing values/duplicates, document attribute meanings (`data/raw/student.txt`).
2. **EDA** — distributions, correlations, and a few targeted plots (absences vs. pass rate, study time vs. grade) to find the features worth engineering around.
3. **Pipeline** — a single scikit-learn `ColumnTransformer` + `Pipeline` so preprocessing is identical at train and inference time; avoids train/serve skew in the app.
4. **Model comparison** — cross-validate a small, well-understood set of regressors (Linear, Ridge, Random Forest, Gradient Boosting) rather than overfitting the search to this small dataset.
5. **Evaluation** — regression metrics (RMSE, MAE, R²) plus the derived pass/fail accuracy, explicitly checked against a naive baseline to avoid overstating the classification result.
6. **App** — a Streamlit form mirroring the early-warning feature set, so a non-technical user (teacher, counselor) can get a prediction without touching code.
7. **Docs** — README (overview + results for a browsing visitor), REPORT (methodology and honest discussion of limitations for a technical reviewer), this PLAN (the reasoning behind the above).

## Engineering conventions

- No absolute paths — everything resolves relative to `PROJECT_ROOT` via `pathlib`.
- `src/` modules are the single source of truth for feature lists, preprocessing, and training; the notebook and app both import from `src` rather than redefining logic.
- Notebook outputs are committed (not stripped) so plots render directly on GitHub without requiring a local run.
- Tests cover the three things most likely to silently break: data loads, the pipeline fits, and predictions are finite and in a plausible range — not exhaustive, but enough to catch a broken refactor.
