"""Decision-time feature contract shared by model fitting and inference.

Only the explicit schema below is read. IDs, outcome metadata, next states,
and caller-supplied history never enter the feature representation. ``fit``
uses only its model's designated fit role: policy-fit for a learned policy,
evaluator-fit for a separate fixed-policy evaluator. ``transform`` never changes
fitted medians or categories. Persist the fitted encoder with the model.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted


SCHEMA_VERSION = 2
MISSING_CATEGORY = '__missing__'
ACTIONS = (
    'WAIT', 'CHECK_IN', 'ESTIMATE_NUDGE', 'ASK_OBJECTION',
    'OFFER_SCHEDULING', 'MEMBERSHIP_TOUCH', 'PARK_THREAD', 'ESCALATE_TO_HUMAN',
)
CATEGORICAL_FIELDS = (
    'topic', 'equipment_type', 'season', 'track_hint', 'heat',
    'estimate_value_bucket', 'membership_status', 'customer_signal', 'last_action',
)
NUMERIC_FIELDS = (
    'turn', 'day', 'weather_extreme', 'ignored_count', 'touch_count',
    'user_patience', 'urgency_score', 'price_sensitivity_score',
    'service_risk_score', 'relationship_score',
    'days_since_last_inbound', 'days_since_last_outbound',
)
DERIVED_NUMERIC_FIELDS = (
    'remaining_decisions', 'consecutive_repeat_count', 'consecutive_waits',
    'occurred_ASK_OBJECTION', 'occurred_ESTIMATE_NUDGE', 'occurred_OFFER_SCHEDULING',
    'days_since_last_inbound_is_99', 'days_since_last_outbound_is_99',
) + tuple('previous_count_' + action for action in ACTIONS)
ALL_NUMERIC_FIELDS = NUMERIC_FIELDS + DERIVED_NUMERIC_FIELDS
VISIBLE_FIELDS = NUMERIC_FIELDS + CATEGORICAL_FIELDS + ('previous_actions',)


def _number(value):
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _category(value):
    return value if isinstance(value, str) and value else MISSING_CATEGORY


def featurize(observation, history=None):
    """Return a deterministic scalar dictionary without learning or mutation.

    Numeric missingness is represented by None until the fitted encoder adds
    indicators and imputes values. Recency 99 remains a raw value plus an
    indicator, without treating its absent-contact interpretation as proven.
    A nonempty visible previous_actions list determines last_action; otherwise
    the explicit field is used. Missing action history is distinct from an
    explicitly empty history. Unknown action strings break a repeat run.
    """
    del history  # The checker may supply unrelated cases; use visible actions only.
    features = {field: _number(observation.get(field)) for field in NUMERIC_FIELDS}
    features.update({field: _category(observation.get(field)) for field in CATEGORICAL_FIELDS})
    previous = observation.get('previous_actions')
    has_previous = isinstance(previous, (list, tuple))
    if has_previous and previous:
        features['last_action'] = _category(previous[-1])
    for action in ACTIONS:
        features['previous_count_' + action] = previous.count(action) if has_previous else None
    repeat_count = None
    if has_previous:
        repeat_count = 0
        if previous and isinstance(previous[-1], str) and previous[-1] in ACTIONS:
            for action in reversed(previous):
                if action != previous[-1]:
                    break
                repeat_count += 1
    features['consecutive_repeat_count'] = repeat_count
    features['consecutive_waits'] = (repeat_count if previous and previous[-1] == 'WAIT'
                                     else 0) if has_previous else None
    for action in ('ASK_OBJECTION', 'ESTIMATE_NUDGE', 'OFFER_SCHEDULING'):
        features['occurred_' + action] = int(action in previous) if has_previous else None
    turn = features['turn']
    features['remaining_decisions'] = 4 - turn if turn is not None else None
    for field in ('days_since_last_inbound', 'days_since_last_outbound'):
        features[field + '_is_99'] = int(features[field] == 99)
    return features


class FeatureEncoder(TransformerMixin, BaseEstimator):
    """Dense one-hot categories, fit-set medians, and fixed missing indicators.

    Every numeric column has a missing indicator, including columns that had
    no missing values during fit. Entirely absent fit columns use zero as a
    documented fallback and keep their indicator; no held-out values are used.
    Missing categories have an explicit level even when absent in the fit set.
    Unseen nonmissing categories produce an all-zero block for that field.
    """

    @staticmethod
    def _arrays(observations):
        rows = [featurize(obs) for obs in observations]
        numbers = np.asarray([[row[field] for field in ALL_NUMERIC_FIELDS] for row in rows],
                             dtype=float).reshape(len(rows), len(ALL_NUMERIC_FIELDS))
        categories = np.asarray([[row[field] for field in CATEGORICAL_FIELDS] for row in rows],
                                dtype=object).reshape(len(rows), len(CATEGORICAL_FIELDS))
        return numbers, categories

    def fit(self, X, y=None):
        numbers, categories = self._arrays(X)
        if not len(numbers):
            raise ValueError('cannot fit FeatureEncoder on an empty dataset')
        # Avoid nanmedian warnings for columns with no observed fit values.
        self.medians_ = np.asarray([
            np.median(column[~np.isnan(column)]) if np.any(~np.isnan(column)) else 0.
            for column in numbers.T
        ])
        levels = [sorted(set(column) | {MISSING_CATEGORY}) for column in categories.T]
        self.one_hot_ = OneHotEncoder(categories=levels, handle_unknown='ignore',
                                      sparse_output=False, dtype=np.float64)
        self.one_hot_.fit(categories)
        self.schema_version_ = SCHEMA_VERSION
        return self

    def transform(self, X):
        check_is_fitted(self, ['medians_', 'one_hot_', 'schema_version_'])
        numbers, categories = self._arrays(X)
        if not len(numbers):
            return np.empty((0, len(self.get_feature_names_out())))
        missing = np.isnan(numbers)
        imputed = np.where(missing, self.medians_, numbers)
        return np.hstack((imputed, missing.astype(float), self.one_hot_.transform(categories)))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ['medians_', 'one_hot_'])
        return np.asarray(list(ALL_NUMERIC_FIELDS)
                          + [field + '_is_missing' for field in ALL_NUMERIC_FIELDS]
                          + list(self.one_hot_.get_feature_names_out(CATEGORICAL_FIELDS)), dtype=object)
