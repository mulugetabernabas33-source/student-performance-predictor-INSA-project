# Report: Predicting Student Performance

## 1. Problem statement

Schools want to identify students at risk of failing early enough to act on it. Most grade-prediction work (including the original study behind this dataset) predicts final grades using **prior period grades** as features — which is accurate, but not actionable at the start of a term, when no grades exist yet.

This project frames the problem two ways to make that trade-off explicit:

1. **Early-warning model** — predict the final grade (`G3`, 0–20) using only demographic, family, and behavioral features available before the term's first graded assessment.
2. **Full model** — the same task, adding the first and second period grades (`G1`, `G2`) as features, to measure how much predictive signal is lost by excluding them.

The deployed app uses the early-warning model, since it is the only one of the two that is useful for early intervention.

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

Four regressors were 5-fold cross-validated on each feature set, scored by R²: Linear Regression, Ridge, Random Forest, and Gradient Boosting (see `src/train.py`). Random Forest ranked first on both feature sets and was selected for final training and deployment.

### 3.3 Evaluation

The selected model was refit on the full training split (80%) and evaluated once on a held-out test split (20%, `random_state=42`). Metrics: RMSE, MAE, R² for the regression task, plus a derived pass/fail accuracy (predicted grade rounded and thresholded at 10) to connect the regression output to the practical pass/fail decision a teacher cares about.

## 4. Results

| Feature set | Best model | CV R² (mean ± std) | Test R² | Test RMSE | Test MAE | Test pass/fail accuracy |
|---|---|---|---|---|---|---|
| Early-warning | Random Forest | 0.21 ± 0.15 | 0.30 | 3.78 | 3.01 | 67.1% |
| Full (+ G1, G2) | Random Forest | 0.89 ± 0.03 | 0.80 | 2.01 | 1.19 | 89.9% |

For reference, a naive baseline that always predicts "pass" gets 65.8% accuracy on this test split — so the early-warning model's 67.1% pass/fail accuracy is only a marginal improvement on that specific metric, even though its R² of 0.30 shows it explains real variance in the continuous grade. **This mismatch is worth stating plainly**: a model can meaningfully reduce error on the regression target while barely beating a trivial baseline on a derived binary metric, especially when the classes are imbalanced (67/33 pass/fail split). Anyone deploying this kind of model should look at the regression error and the full predicted-vs-actual spread (`reports/figures/06_predicted_vs_actual.png`), not just a single accuracy number.

### Feature importance (early-warning model)

The Random Forest's top features are `absences` and `failures`, each roughly 2–4x more important than any other single feature (`reports/figures/07_feature_importance.png`). Both are also the two features a school could most directly act on — attendance monitoring and support for students with prior failures — which makes them a reasonable starting point for intervention even given the model's modest accuracy.

## 5. Discussion

- **Why the early-warning model underperforms**: grades are influenced by many factors this dataset doesn't capture (teacher effects, specific test content, short-term events), and the available behavioral/demographic features are individually weak predictors. This is expected and matches the original Cortez & Silva study, which also found G1/G2 to be by far the strongest predictors of G3.
- **Why we still deploy it**: the full model's accuracy is an artifact of predicting G3 from something that already encodes most of the answer (G1 and G2 are themselves grades on a 0–20 scale, highly correlated with G3). It isn't an early-warning tool — by the time G1 exists, a third of the term has already happened.
- **What the early-warning model is actually good for**: a coarse, continuous risk score to prioritize attention, not a replacement for teacher judgment. The pass/fail label should be treated as a convenience summary of the predicted grade, not a verdict.

## 6. Limitations

See the [README's Limitations section](README.md#limitations) — small single-region dataset, no causal interpretation of feature importance, self-reported survey features, and single-subject (Math only) training.

## 7. Possible future work

- Evaluate on the Portuguese course dataset (`data/raw/student-por.csv`) to check whether the same features generalize across subjects.
- Try calibrated classification directly on the pass/fail label (rather than deriving it from a regression) to get class probabilities and a proper precision/recall trade-off for an imbalanced target.
- Collect features that update over the term (e.g., running attendance, homework completion) to turn this into a continuously updating risk score rather than a one-time prediction.
