"""Group-aware cross-fitted stacking on raw frames, including preprocessing."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.utils.validation import check_is_fitted


class GroupStackingRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, estimators, n_splits=3):
        self.estimators = estimators
        self.n_splits = n_splits

    def fit(self, X, y):
        y = np.asarray(y)
        groups = X['host_id'].to_numpy()
        cv = list(GroupKFold(n_splits=self.n_splits).split(X, y, groups))
        oof = np.zeros((len(X), len(self.estimators)))
        self.fold_host_overlap_ = []
        self.oof_coverage_ = np.zeros(len(X), dtype=int)
        for train, valid in cv:
            overlap = len(set(groups[train]) & set(groups[valid]))
            assert overlap == 0
            self.fold_host_overlap_.append(overlap)
            self.oof_coverage_[valid] += 1
            for j, (_, estimator) in enumerate(self.estimators):
                fitted = clone(estimator).fit(X.iloc[train], y[train])
                oof[valid, j] = fitted.predict(X.iloc[valid])
        assert np.all(self.oof_coverage_ == 1)
        self.meta_model_ = Ridge(alpha=1.0).fit(oof, y)
        self.estimators_ = [clone(estimator).fit(X, y) for _, estimator in self.estimators]
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, 'meta_model_')
        return self.meta_model_.predict(np.column_stack([m.predict(X) for m in self.estimators_]))
