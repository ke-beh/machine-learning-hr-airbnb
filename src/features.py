"""Feature construction. Learned statistics are fitted inside each CV fold."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class HRFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X[['department', 'region', 'recruitment_channel', 'KPIs_met >80%',
                 'awards_won?', 'avg_training_score']].copy()
        out['sum_metric'] = (X['awards_won?'] + X['previous_year_rating'].fillna(0)
                             + X['KPIs_met >80%'])
        out['training_effectiveness'] = X['avg_training_score'] / X['no_of_trainings'].replace(0, np.nan)
        return out


class AirbnbFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.caps_ = X[['minimum_nights', 'calculated_host_listings_count']].quantile(.95)
        self.counts_ = X['neighbourhood'].value_counts()
        self.availability_median_ = X['availability_365'].median()
        self.clusters_ = KMeans(n_clusters=10, n_init=10, random_state=42).fit(X[['latitude', 'longitude']])
        return self

    def transform(self, X):
        cols = ['neighbourhood', 'room_type', 'latitude', 'longitude', 'minimum_nights',
                'number_of_reviews', 'calculated_host_listings_count', 'availability_365',
                'distance_from_mrt']
        out = X[cols].copy()
        out['minimum_nights'] = np.log1p(out['minimum_nights'].clip(upper=self.caps_['minimum_nights']))
        out['calculated_host_listings_count'] = out['calculated_host_listings_count'].clip(
            upper=self.caps_['calculated_host_listings_count'])
        out['location_cluster'] = self.clusters_.predict(X[['latitude', 'longitude']]).astype(str)
        out['neighbourhood_popularity'] = X['neighbourhood'].map(self.counts_).fillna(0)
        out['high_availability'] = (X['availability_365'] > self.availability_median_).astype(int)
        return out


def nearest_mrt_km(listings, stations):
    """Vectorized spherical great-circle distance, km; fixed external reference.

    This deterministic operation does not fit on listing prices or distributions.
    It replaces the original slower ellipsoidal geodesic calculation; small
    numeric differences are expected. Station snapshot dates are unverified.
    """
    coords = np.radians(listings[['latitude', 'longitude']].to_numpy(dtype=float))
    stops = np.radians(stations[['Latitude', 'Longitude']].to_numpy(dtype=float))
    if len(stops) == 0 or not np.isfinite(stops).all() or not np.isfinite(coords).all():
        raise ValueError('MRT and listing coordinates must be nonempty and finite.')
    delta = coords[:, None, :] - stops[None, :, :]
    a = np.sin(delta[:, :, 0] / 2)**2 + np.cos(coords[:, None, 0]) * np.cos(stops[None, :, 0]) * np.sin(delta[:, :, 1] / 2)**2
    return (6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))).min(axis=1)


def make_pipeline(task, model):
    if task == 'hr':
        engineer = HRFeatures()
        cats = ['department', 'region', 'recruitment_channel']
        nums = ['KPIs_met >80%', 'awards_won?', 'avg_training_score', 'sum_metric', 'training_effectiveness']
    else:
        engineer = AirbnbFeatures()
        cats = ['neighbourhood', 'room_type', 'location_cluster']
        nums = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews',
                'calculated_host_listings_count', 'availability_365', 'distance_from_mrt',
                'neighbourhood_popularity', 'high_availability']
    numeric = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return Pipeline([('features', engineer), ('preprocess', ColumnTransformer([
        ('numeric', numeric, nums), ('categorical', categorical, cats)])), ('model', model)])
