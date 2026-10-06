"""Tests for leakage boundaries and train-only feature transformations."""
import unittest
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from src.features import AirbnbFeatures, HRFeatures, nearest_mrt_km, make_pipeline
from src.stacking import GroupStackingRegressor


def numeric_signal(X):
    return X[['signal']]


class IntegrityTests(unittest.TestCase):
    def air_frame(self):
        n = 30
        return pd.DataFrame({'neighbourhood': ['A']*15+['B']*15,
            'room_type': ['Private room']*n, 'latitude': np.linspace(1.28, 1.35, n),
            'longitude': np.linspace(103.8, 103.9, n), 'minimum_nights': np.arange(1, n+1),
            'number_of_reviews': np.arange(n), 'calculated_host_listings_count': np.arange(1, n+1),
            'availability_365': np.arange(n), 'distance_from_mrt': np.ones(n),
            'host_id': np.repeat(np.arange(10), 3)})

    def test_airbnb_does_not_learn_from_validation(self):
        train = self.air_frame()
        features = AirbnbFeatures().fit(train)
        caps = features.caps_.copy()
        validation = train.iloc[:2].copy()
        validation['neighbourhood'] = 'UNSEEN'
        validation['minimum_nights'] = 100000
        validation['price'] = 999999  # Must never become a feature.
        transformed = features.transform(validation)
        self.assertTrue(transformed.neighbourhood_popularity.eq(0).all())
        self.assertTrue(np.allclose(transformed.minimum_nights, np.log1p(caps.minimum_nights)))
        self.assertNotIn('price', transformed.columns)
        self.assertNotIn('host_id', transformed.columns)
        pd.testing.assert_series_equal(caps, features.caps_)

    def test_unseen_category_pipeline(self):
        frame = self.air_frame()
        model = make_pipeline('airbnb', DummyRegressor()).fit(frame, np.ones(len(frame)))
        valid = frame.iloc[:1].copy()
        valid['neighbourhood'] = 'NEW'
        self.assertTrue(np.isfinite(model.predict(valid)).all())

    def test_hr_features_are_target_free_and_safe_at_zero(self):
        X = pd.DataFrame({'department':['HR'], 'region':['r1'], 'recruitment_channel':['sourcing'],
            'KPIs_met >80%':[1], 'awards_won?':[0], 'avg_training_score':[80],
            'previous_year_rating':[np.nan], 'no_of_trainings':[0], 'is_promoted':[1]})
        result = HRFeatures().fit_transform(X)
        self.assertEqual(result.sum_metric.iloc[0], 1)
        self.assertTrue(np.isnan(result.training_effectiveness.iloc[0]))
        self.assertNotIn('is_promoted', result)

    def test_mrt_distance_identity_and_units(self):
        locations = pd.DataFrame({'latitude':[0., 1.], 'longitude':[0., 0.]})
        stations = pd.DataFrame({'Latitude':[0.], 'Longitude':[0.]})
        distances = nearest_mrt_km(locations, stations)
        self.assertEqual(distances[0], 0)
        self.assertAlmostEqual(distances[1], 111.195, places=2)

    def test_stacking_oof_hosts_do_not_overlap(self):
        X = pd.DataFrame({'host_id':np.repeat(np.arange(12), 2), 'signal':np.arange(24)})
        y = np.sin(np.arange(24))
        base = Pipeline([('select', FunctionTransformer(numeric_signal)), ('model', DummyRegressor())])
        model = GroupStackingRegressor([('a', base), ('b', base)]).fit(X, y)
        self.assertEqual(model.fold_host_overlap_, [0, 0, 0])
        self.assertTrue(np.all(model.oof_coverage_ == 1))
        self.assertTrue(np.isfinite(model.predict(X.iloc[:2])).all())


if __name__ == '__main__':
    unittest.main()
