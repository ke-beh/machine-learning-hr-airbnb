# Technical review and changes

Review date: 6 October 2026. Cell references below are **zero-based indexes in the
original supplied notebooks**, before the archive notice was added.

## Review conclusion

The project demonstrates substantial exploratory analysis, feature engineering,
model comparison and tuning. The original headline scores are not defensible
estimates of generalization. The current implementation rebuilds evaluation from
raw inputs and supersedes those scores. It is a retrospective educational
benchmark, not a deployed HR or pricing system.

## What was inspected

- All source cells in both notebooks: 219 in Assignment 1 and 140 in Assignment 2.
- Saved text outputs, reported metrics, search settings and recorded warnings.
- All six CSV schemas, row counts, missing values and relevant distributions.
- The eight-page Assignment 2 brief, including required and optional deliverables.
- The Word report's extracted text (821 paragraphs). Its 66 embedded images were
  not individually rendered and verified; the report is excluded from publication.
- Static pickle metadata and generated CatBoost training-log metadata. The pickle
  was disassembled, not loaded or executed.

There is no original README, Assignment 1 brief, slide deck, Streamlit app, video,
dataset license file or dependency manifest in the supplied folder. Absence from
this folder does not establish whether these were submitted elsewhere.

## Findings and disposition

| Priority | Evidence in original source | Problem | Current disposition |
|---|---|---|---|
| Critical | Assignment 2 cell 13: `SMOTE().fit_resample(X, y)` | Entire dataset is resampled into training after the split. Test records directly appear in training. | New HR workflow fits only on the training partition; no SMOTE. Class weighting is compared inside CV. Original 96.88% KNN test accuracy is withdrawn. |
| Critical | Assignment 2 cell 136: `stacked_model.fit(base_model_predictions, y_test)` | Stack is fitted and evaluated on test labels. Earlier stacking also uses in-sample base predictions as inputs. | New stack uses group-disjoint out-of-fold predictions and fits its meta-model on training labels only. Original R² 0.8804 is withdrawn. |
| High | Assignment 1 cells 201, 218 | `price_per_minimum_stay` includes the target; the neighbourhood price aggregate also uses target information. Both remain in the source export. | Neither feature exists in the corrected predictor. Do not rerun archived preparation to supply current training. |
| High | Assignment 1 cells 179, 205, 216, 218 versus supplied `listings_new.csv` | Source would produce numeric room dummy names and additional target-derived features. Supplied CSV has string room dummy names and omits the two target-derived features. Scaling changes `listing_new`, while export writes a separate frame. | Bypass both preprocessed CSVs and derive all model inputs from raw data. Saved CSV is not evidence of reproducibility. |
| High | Assignment 1 cell 108 and Airbnb clustering, counts and caps | Learned preprocessing is fitted before train/test separation and outside CV. | Imputation, encoding, scaling, caps, clustering, counts and availability median are fitted inside every training fold, including stacking inner folds. |
| High | Assignment 2 search cells following cell 13 | Searches operate on already resampled data; synthetic observations can contaminate CV folds. | New searches use raw training frames and pipelines. Any future sampler belongs inside an imbalanced-learn pipeline. |
| High | Assignment 2 cells 78, 139 | Models are declared best from contaminated test scores. Baseline and tuned CV runs use a different data regime from the resampled holdout runs. | Candidate selection uses training CV only. Holdout is evaluated once for the locked winner and a dummy baseline. |
| High | Assignment 2 cell 88 | Random listing split may place the same host on both sides. | New outer split, model-selection CV and stacking inner CV are disjoint by `host_id`. IDs are excluded from model features. |
| Medium | Assignment 2 cell 120 and saved warnings | `subsample` 1.2 and 1.4 are invalid; 1,152 of 1,728 fits failed. | New, bounded grids contain valid parameters and use `error_score='raise'`. |
| Medium | Assignment 2 cell 132 | CatBoost search includes depths 24 and 36, beyond supported CPU depth. | Legacy grid is not reused. Current benchmark uses scikit-learn models. |
| Medium | Assignment 2 cells 120, 124, 132 | Search defaults to R² but prints its negative as a generic best score; LightGBM instead uses negative MSE. | New regression selection metric is consistently negative RMSE on `log1p(price)`, labelled explicitly. |
| Medium | Assignment 2 cell 128 | LightGBM `subsample` varies without enabling a positive bagging frequency. | Not carried into the new benchmark. Legacy settings should not be described as proven subsampling experiments. |
| Medium | Assignment 2 cell 72 | Early-stopping parameter is present, but no explicit evaluation set is passed. | Do not claim early stopping actually occurred from the parameter alone. New boosting uses a fixed iteration budget. |
| Medium | Assignment 1 cell 176 | Price-filtered data are assigned to `filtered_listings`, but downstream processing continues with `listing_new`. The prose calls percentile filtering IQR filtering. | Current scope includes all positive-priced Central Region listings, without target percentile trimming. One zero-price listing is excluded by a fixed validity rule. |
| Medium | Assignment 2 cells 84, 90, 91 | Metrics are on `log1p(price)`, not currency; prose claims MAPE although no MAPE is computed. | New results report log-space metrics and original-price-unit MAE/RMSE/R² separately. Currency is not asserted without source metadata. |
| Medium | Assignment 2 cells 7, 78 | Narrative says classes are balanced despite 50,140 negatives and 4,668 positives. Accuracy alone obscures weak promotion recall. | New model selection uses average precision. README includes prevalence, majority baseline, precision, recall and F1 at a stated threshold. |
| Medium | Assignment 1 cells 66, 179 and original SMOTE | Nominal categories are arbitrary integers. Distance-based models and standard SMOTE treat these as ordered numeric distances. | New pipelines one-hot encode nominal categories, including location clusters. Class weighting avoids synthetic fractional categories. |
| Medium | Assignment 2 cell 12 and many estimators | Unseeded split, sampler and estimators make exact reproduction unreliable. | Explicit seed, bounded search, package versions and input SHA-256 hashes are recorded. |
| Documentation | Assignment 1 cells 35, 39, 41, 43, 49, 82–85, 101, 115, 199, 211, 215 | Rating pie labels can mismatch frequency order; fairness and equal-rate conclusions are unsupported; chart title mentions awards without filtering awards; SMOTE code is inside string literals; `experience_ratio` is discussed but not created; one row count is 7,987 instead of 7,907; feature descriptions/dropped-column claims do not consistently match code. | Current documentation describes verified code and limits. Original prose remains visible only in the clearly labelled historical archive. |
| Documentation | Word report preprocessing and results | Report mentions `popularity_score`, `room_type_premium`, imputations and metrics not present in the supplied implementation; some numbers differ from saved outputs. | README is grounded in source and new results, not copied from report claims. |

An important distinction: the original *baseline* HR cross-validation calls on
`X, y` do not themselves use the SMOTE-resampled frame. They still inherit
whole-dataset preprocessing. Tuned CV results additionally inherit parameters
chosen using the contaminated search. They should not be combined into a clean
leaderboard or substituted for a fresh evaluation.

The dangerous target-derived Airbnb columns are **absent from the supplied
`listings_new.csv`**. Their presence in the preparation source is a reproducibility
and future-leakage problem; it is not proof that those columns were used in the
saved Assignment 2 run.

## Corrected evaluation protocol

HR retains the original eight-feature concept: three nominal categories, KPI,
awards, training score, the composite performance score and training effectiveness.
Missing prior ratings are treated as no recorded rating when forming the composite.
This assumption does not mean zero is an observed poor rating. Education, gender,
age and employee ID are not predictors. Exclusion alone does not prove fairness.

HR uses a seeded, stratified 75/25 split and three-fold stratified training CV.
Candidate families are logistic regression, random forest and histogram gradient
boosting. Small grids test regularization, class weighting and tree complexity.
Average precision selects the winner. The threshold remains 0.5; it has not been
optimized on the holdout.

Airbnb retains the historical Central Region study scope. It excludes one
nonpositive price and does not remove expensive listings based on the target.
The 80/20 split is by hosts, so row proportions need not be exactly 80/20.
Three-fold grouped CV compares ridge regression, random forest, histogram
gradient boosting and group-aware stacking. The stack combines fixed RF and
histogram-boosting pipelines with a ridge meta-model. This is a new 2026
implementation, different from the original RF/XGBoost/linear-regression stack.
Every inner stacking fit includes its own feature engineering and preprocessing.
Final base pipelines are refitted on the full training partition after generating
out-of-fold meta-training inputs. Test labels are never passed to `fit`.

Airbnb predictors include raw latitude/longitude, room type, neighbourhood, stay
rules, review count, host listing count, availability, MRT distance and train-fitted
geographic/popularity features. Host and listing names are excluded. Raw coordinates
are retained in addition to clusters. The new distance implementation uses a
spherical great-circle approximation, whereas the historical notebook used geopy's
ellipsoidal geodesic. Neither measures walking accessibility.

Regression selection minimizes log-space RMSE. Back-transforming log predictions
does not provide a bias-corrected expected monetary price; original-scale errors
are reported honestly, with no claim of revenue uplift. The close CV difference
between stacking and histogram boosting does not establish statistically
significant superiority.

## What changed versus the coursework

The original eight classifier and five regressor families remain documented in the
archives. Their enormous search grids were not rerun. The current smaller benchmark
prioritizes verifiable evaluation over algorithm count. It adds simple baselines,
consistent selection metrics, grouped stacking, reproducibility records, tests and
fresh plots. No deployment or Streamlit implementation has been invented.

Only the archive packaging changed in the historical notebooks: filenames,
environment metadata, execution counters and saved outputs were cleaned, and a
notice was prepended. Every original source cell is otherwise unchanged. The
`source_manifest.json` records source hashes. Updated code is under `src/`.

## Remaining limits

- Source URLs, redistribution terms, snapshot dates and price currency remain
  unverified. Raw CSVs and school documents are not bundled.
- The holdouts are retrospective splits of old coursework data, not a newly
  collected external dataset or proof of current market performance.
- HR labels may encode historical decisions and bias. There is no fairness audit,
  causal analysis, calibration study or deployment validation.
- Airbnb validation represents unseen hosts within the historical Central Region
  sample. It is not a temporal forecast or evidence of Singapore-wide accuracy.
- Availability is not observed occupancy, and listing counts are not observed
  demand. Correlations do not demonstrate causal effects of transit proximity.
- The MRT snapshot may include stations not operating at the listing date.
- Full historical notebooks were source-reviewed, not executed end to end.

## Technical references

- [imbalanced-learn: data leakage and resampling](https://imbalanced-learn.org/stable/common_pitfalls.html)
- [scikit-learn: stacking and out-of-fold predictions](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html)
- [scikit-learn: common preprocessing pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)

These explain the evaluation principles; the specific findings above come from
the supplied files and recorded outputs.
