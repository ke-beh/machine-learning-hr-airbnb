"""Run from the repository root: python -m src.evaluate --data-dir PATH."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    mean_absolute_error, root_mean_squared_error, r2_score, PrecisionRecallDisplay)
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits
from .features import make_pipeline, nearest_mrt_km
from .stacking import GroupStackingRegressor

SEED = 42


def select_model(task, candidates, X, y, cv):
    records, best, best_score = [], None, -np.inf
    scoring = 'average_precision' if task == 'hr' else 'neg_root_mean_squared_error'
    for name, estimator, grid in candidates:
        print(f'{task}: validating {name}', flush=True)
        search = GridSearchCV(estimator, grid, cv=cv, scoring=scoring,
                              n_jobs=1, error_score='raise', return_train_score=False)
        search.fit(X, y)
        records.append({'model': name, 'cv_score': float(search.best_score_),
                        'cv_std': float(search.cv_results_['std_test_score'][search.best_index_]),
                        'best_params': search.best_params_})
        if search.best_score_ > best_score:
            best_score, best = search.best_score_, (name, search.best_estimator_)
    return best, records


def classification_metrics(y, pred, prob):
    return {k: float(v) for k, v in {
        'accuracy': accuracy_score(y, pred), 'balanced_accuracy': balanced_accuracy_score(y, pred),
        'precision': precision_score(y, pred, zero_division=0), 'recall': recall_score(y, pred),
        'f1': f1_score(y, pred), 'average_precision': average_precision_score(y, prob),
        'roc_auc': roc_auc_score(y, prob)}.items()}


def regression_metrics(y_log, pred_log):
    pred_log = np.maximum(pred_log, 0)
    return {'rmse_log1p': float(root_mean_squared_error(y_log, pred_log)),
            'r2_log1p': float(r2_score(y_log, pred_log)),
            'mae_price_units': float(mean_absolute_error(np.expm1(y_log), np.expm1(pred_log))),
            'rmse_price_units': float(root_mean_squared_error(np.expm1(y_log), np.expm1(pred_log))),
            'r2_price_units': float(r2_score(np.expm1(y_log), np.expm1(pred_log)))}


def run(data_dir, output_dir):
    start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = {name: pd.read_csv(data_dir / name) for name in ['hr_data.csv', 'listings.csv', 'Cleaned_MRT_Stations.csv']}
    hr = raw['hr_data.csv']
    if hr.employee_id.isna().any() or hr.employee_id.duplicated().any():
        raise ValueError('HR employee IDs must be complete and unique for this split protocol.')
    X, y = hr.drop(columns='is_promoted'), hr.is_promoted
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=SEED)
    assert not set(Xtr.employee_id) & set(Xte.employee_id)
    hr_candidates = [
        ('Logistic regression', make_pipeline('hr', LogisticRegression(max_iter=1000, random_state=SEED)),
         {'model__C': [.1, 1.0], 'model__class_weight': [None, 'balanced']}),
        ('Random forest', make_pipeline('hr', RandomForestClassifier(n_estimators=150, n_jobs=2, random_state=SEED)),
         {'model__min_samples_leaf': [2, 8], 'model__class_weight': [None, 'balanced']}),
        ('Histogram gradient boosting', make_pipeline('hr', HistGradientBoostingClassifier(max_iter=150, early_stopping=False, random_state=SEED)),
         {'model__max_leaf_nodes': [15, 31]})]
    (hr_name, hr_model), hr_cv = select_model('hr', hr_candidates, Xtr, ytr,
        StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED))
    # The winner is locked before this first and only holdout evaluation.
    hr_pred, hr_prob = hr_model.predict(Xte), hr_model.predict_proba(Xte)[:, 1]
    dummy = DummyClassifier(strategy='prior').fit(np.zeros((len(ytr), 1)), ytr)
    dummy_pred = dummy.predict(np.zeros((len(yte), 1)))
    dummy_prob = dummy.predict_proba(np.zeros((len(yte), 1)))[:, 1]
    hr_result = {'selected_model': hr_name, 'selection_metric': 'average_precision', 'cv': hr_cv,
        'train_rows': len(Xtr), 'test_rows': len(Xte), 'promotion_prevalence': float(y.mean()),
        'threshold': .5, 'test': classification_metrics(yte, hr_pred, hr_prob),
        'baseline_test': classification_metrics(yte, dummy_pred, dummy_prob),
        'confusion_matrix': confusion_matrix(yte, hr_pred).tolist(), 'employee_overlap': 0}
    fig, ax = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(yte, hr_prob, ax=ax, name=hr_name)
    ax.axhline(yte.mean(), color='gray', linestyle='--', label='No-skill prevalence')
    ax.legend(); ax.set_title('HR promotion: untouched holdout'); fig.tight_layout()
    fig.savefig(output_dir / 'hr_precision_recall.png', dpi=160); plt.close(fig)

    listings = raw['listings.csv']
    air = listings.loc[(listings.neighbourhood_group == 'Central Region') & (listings.price > 0)].copy()
    if air.id.duplicated().any() or air.host_id.isna().any():
        raise ValueError('Listing IDs must be unique and host IDs complete.')
    air['distance_from_mrt'] = nearest_mrt_km(air, raw['Cleaned_MRT_Stations.csv'])
    X, y = air.drop(columns='price'), np.log1p(air.price)
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(X, y, X.host_id))
    Xtr, Xte, ytr, yte = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
    assert not set(Xtr.host_id) & set(Xte.host_id)
    cv = list(GroupKFold(n_splits=3).split(Xtr, ytr, Xtr.host_id))
    for tr, va in cv:
        assert not set(Xtr.iloc[tr].host_id) & set(Xtr.iloc[va].host_id)
    rf = make_pipeline('airbnb', RandomForestRegressor(n_estimators=150, min_samples_leaf=3, n_jobs=2, random_state=SEED))
    gb = make_pipeline('airbnb', HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15, early_stopping=False, random_state=SEED))
    air_candidates = [
        ('Ridge regression', make_pipeline('airbnb', Ridge()), {'model__alpha': [1., 10., 100.]}),
        ('Random forest', rf, {'model__min_samples_leaf': [3, 8]}),
        ('Histogram gradient boosting', gb, {'model__max_leaf_nodes': [15, 31]}),
        ('Group-aware stacking', GroupStackingRegressor([('rf', rf), ('gb', gb)]), {})]
    (air_name, air_model), air_cv = select_model('airbnb', air_candidates, Xtr, ytr, cv)
    air_pred = air_model.predict(Xte)
    baseline = DummyRegressor(strategy='mean').fit(np.zeros((len(ytr), 1)), ytr)
    baseline_pred = baseline.predict(np.zeros((len(yte), 1)))
    air_result = {'selected_model': air_name, 'selection_metric': 'negative_rmse_log1p', 'cv': air_cv,
        'rows': len(air), 'train_rows': len(Xtr), 'test_rows': len(Xte), 'host_overlap': 0,
        'train_hosts': int(Xtr.host_id.nunique()), 'test_hosts': int(Xte.host_id.nunique()),
        'excluded_nonpositive_central_prices': int(((listings.neighbourhood_group == 'Central Region') & (listings.price <= 0)).sum()),
        'test': regression_metrics(yte, air_pred), 'baseline_test': regression_metrics(yte, baseline_pred)}
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yte, air_pred, s=12, alpha=.4)
    lim = [min(yte.min(), air_pred.min()), max(yte.max(), air_pred.max())]
    ax.plot(lim, lim, '--', color='gray'); ax.set(xlabel='Actual log(1 + price)', ylabel='Predicted log(1 + price)',
        title='Airbnb: holdout hosts absent from training')
    fig.tight_layout(); fig.savefig(output_dir / 'airbnb_predictions.png', dpi=160); plt.close(fig)
    result = {'review_date': '2026-10-06', 'seed': SEED, 'python': platform.python_version(),
        'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'threadpoolctl']},
        'input_sha256': {name: hashlib.sha256((data_dir/name).read_bytes()).hexdigest() for name in raw},
        'hr': hr_result, 'airbnb': air_result, 'elapsed_seconds': round(time.time()-start, 1)}
    (output_dir / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/raw'))
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.data_dir, args.output_dir)
