import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ADULT_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
]

NUMERIC_RAW = [
    'age', 'fnlwgt', 'education-num',
    'capital-gain', 'capital-loss', 'hours-per-week'
]


class AdultFeatureEngineer(BaseEstimator, TransformerMixin):
    """Clean raw Adult columns and create leakage-safe derived features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X.columns = ADULT_COLUMNS

        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].astype(str).str.strip().replace({'?': np.nan})

        for col in NUMERIC_RAW:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        X = X.drop(columns=['education'])

        X['capital_net'] = X['capital-gain'] - X['capital-loss']
        X['hours_age_ratio'] = X['hours-per-week'] / (X['age'] + 1.0)
        X['capital_gain_flag'] = (X['capital-gain'] > 0).astype(int)

        X['age_band'] = pd.cut(
            X['age'],
            bins=[0, 25, 35, 50, np.inf],
            labels=['<=25', '26-35', '36-50', '51+'],
            include_lowest=True
        )

        return X

    def get_feature_names_out(self, input_features=None):
        return np.array([
            'age', 'workclass', 'fnlwgt', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week',
            'native-country', 'capital_net', 'hours_age_ratio',
            'capital_gain_flag', 'age_band'
        ], dtype=object)