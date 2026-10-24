# Report: Predicting Student Performance

## 1. Problem statement

Schools want to identify students at risk of failing early enough to act on it. Most grade-prediction work (including the original study behind this dataset) predicts final grades using **prior period grades** as features — which is accurate, but not actionable at the start of a term, when no grades exist yet.

This project frames the problem two ways to make that trade-off explicit, referred to throughout as two scenarios:

1. **Scenario A — Early-warning** — predict the outcome using only demographic, family, and behavioral features available before the term's first graded assessment.
2. **Scenario B — Full history** — the same task, adding the first and second period grades (`G1`, `G2`) as features, to measure how much predictive signal is lost by excluding them (and how much of Scenario B's accuracy is really just leakage from grades that don't exist yet at decision time).

Both scenarios are modeled two ways: as **regression** (predicting the continuous final grade `G3`) and as **direct classification** (predicting the binary `pass`/`fail` label). The app deploys the Scenario A classifier, since it is the only combination that is both actionable before any grades exist and gives a calibrated probability + confusion-matrix-style evaluation rather than a grade rounded into a verdict.

## 2. Data

[UCI Student Performance Data Set](https://archive.ics.uci.edu/dataset/320/student+performance) (Cortez & Silva, 2008), Math course subset: 395 students at two Portuguese secondary schools, 30 demographic/family/social/school features plus three grade columns (`G1`, `G2`, `G3`, each 0–20). No missing values or duplicate rows.

- Final grade (`G3`) mean: 10.4, std: 4.6, range 0–20.
- Pass rate (`G3 >= 10`): 67.1%.
- `G1` correlates with `G3` at r = 0.80; `G2` at r = 0.90 — expected, since grades within a term are highly autocorrelated.

### Strongest non-grade correlates of G3

| Feature | Correlation with G3 |
|---|---|
| `failures` (past class failures) | −0.36 |
| `Medu` (mother's education) | +0.22 |
| `age` | −0.16 |
| `Fedu` (father's education) | +0.15 |
| `goout` (going out with friends) | −0.13 |
| `traveltime` | −0.12 |
| `studytime` | +0.10 |

No single demographic/behavioral feature correlates strongly with `G3` in isolation — consistent with the early-warning model's modest R² (see below). The signal is real but diffuse across many weak features.

## 3. Methodology

### 3.1 Preprocessing

- Numeric features: standard-scaled.
- Categorical features: one-hot encoded (`handle_unknown="ignore"` so the deployed pipeline never breaks on an unseen category).
- Implemented as a single `ColumnTransformer` inside a scikit-learn `Pipeline`, so the exact same preprocessing is applied at train and inference time (see `src/preprocess.py`).

### 3.2 Model selection

**Regression** (`src/train.py`): four regressors were 5-fold cross-validated on each scenario, scored by R²: Linear Regression, Ridge, Random Forest, and Gradient Boosting. Random Forest ranked first on both scenarios.

**Classification** (`src/classify.py`): three classifiers were 5-fold cross-validated on each scenario, scored by ROC-AUC: Logistic Regression, Random Forest (`class_weight="balanced"` to offset the 67/33 pass/fail split), and Gradient Boosting.

### 3.3 Evaluation

Each selected model was refit on the full training split (80%) and evaluated once on a held-out test split (20%, `random_state=42`, stratified by label for classification). Regression metrics: RMSE, MAE, R². Classification metrics: accuracy, precision/recall/F1 for the pass class, **recall for the fail class**, and ROC-AUC. Recall on the fail class is reported separately because it is the metric that matters most operationally — missing a student who will fail (a false negative) is the costly error for an early-warning tool, more so than flagging a student who ultimately passes.

## 4. Results: regression (predicting the final grade)

| Scenario | Best model | CV R² (mean ± std) | Test R² | Test RMSE | Test MAE | Test pass/fail accuracy* |
|---|---|---|---|---|---|---|
| A — Early-warning | Random Forest | 0.21 ± 0.15 | 0.30 | 3.78 | 3.01 | 67.1% |
| B — Full (+ G1, G2) | Random Forest | 0.89 ± 0.03 | 0.80 | 2.01 | 1.19 | 89.9% |

\* Pass/fail accuracy here is *derived* from the regression output (predicted grade rounded and thresholded at 10), not a classifier's own decision — see Section 5 for the direct classification results the app actually uses.

For reference, a naive baseline that always predicts "pass" gets 65.8% accuracy on this test split — so Scenario A's derived 67.1% pass/fail accuracy is only a marginal improvement on that specific metric, even though its R² of 0.30 shows it explains real variance in the continuous grade. **This mismatch is worth stating plainly**: a model can meaningfully reduce error on the regression target while barely beating a trivial baseline on a derived binary metric, especially when the classes are imbalanced. Anyone deploying this kind of model should look at the regression error and the full predicted-vs-actual spread (`reports/figures/06_predicted_vs_actual.png`), not just a single accuracy number.

### Feature importance (Scenario A regressor)

The Random Forest's top features are `absences` and `failures`, each roughly 2–4x more important than any other single feature (`reports/figures/07_feature_importance.png`). Both are also the two features a school could most directly act on — attendance monitoring and support for students with prior failures — which makes them a reasonable starting point for intervention even given the model's modest accuracy.

## 5. Results: direct classification (pass/fail) — deployed in the app

Rather than deriving pass/fail from a rounded grade, this variant trains directly on the binary label, which is what the Streamlit app's probability gauge, confusion matrix, and ROC curve are built from (`src/classify.py`, `src/evaluate_classification.py`).

| Scenario | Best model | CV ROC-AUC (mean ± std) | Test accuracy | Test recall (pass) | **Test recall (fail)** | Test ROC-AUC |
|---|---|---|---|---|---|---|
| A — Early-warning | Random Forest | 0.73 ± 0.08 | 69.6% | 81.1% | **46.2%** | 0.65 |
| B — Full (+ G1, G2) | Gradient Boosting | 0.98 ± 0.02 | 88.6% | 86.8% | **92.3%** | 0.94 |

Direct classification confirms the regression-based finding from a different angle: Scenario A catches only about 46% of students who will actually fail (recall on the fail class), even with class-balanced training — it is a weak screen on its own, better used to prioritize review than to make a final call. Scenario B's 92.3% fail-recall is, again, driven by `G1`/`G2` already containing most of the answer, not by the demographic/behavioral features becoming more informative.

## 6. Discussion

- **Why Scenario A underperforms**: grades are influenced by many factors this dataset doesn't capture (teacher effects, specific test content, short-term events), and the available behavioral/demographic features are individually weak predictors. This is expected and matches the original Cortez & Silva study, which also found G1/G2 to be by far the strongest predictors of G3.
- **Why we still deploy Scenario A**: Scenario B's accuracy is an artifact of predicting the outcome from something that already encodes most of the answer (G1 and G2 are themselves grades on a 0–20 scale, highly correlated with G3). It isn't an early-warning tool — by the time G1 exists, a third of the term has already happened. This is the "leakage decision" referenced on the app's Model Performance page: Scenario B is shown for comparison only, never for deployment.
- **What Scenario A is actually good for**: a coarse risk score to prioritize attention, not a replacement for teacher judgment. Given its 46% recall on the fail class, a student the model marks "pass" can still fail — it should never be the sole basis for withholding support from a student who requests it.

## 7. Limitations

See the [README's Limitations section](README.md#limitations) — small single-region dataset, no causal interpretation of feature importance, self-reported survey features, and single-subject (Math only) training.

## 8. Possible future work

- Evaluate on the Portuguese course dataset (`data/raw/student-por.csv`) to check whether the same features generalize across subjects.
- Tune the classification decision threshold below 0.5 to trade some precision for higher fail-class recall, since missing an at-risk student is the costlier error here.
- Collect features that update over the term (e.g., running attendance, homework completion) to turn this into a continuously updating risk score rather than a one-time prediction.
