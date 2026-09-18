import copy

import joblib
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from submission.features import FeatureEncoder, featurize


def test_features_exclude_identifiers_and_all_outcome_metadata():
    obs = {'turn': 0, 'topic': 'open_estimate', 'case_id': 'a'}
    poisoned = dict(obs, case_id='b', episode_id='secret', step=99,
                    reward=100000, action_prob=0.99, done=True,
                    info={'outcome': 'sold'}, next_observation={'turn': 3},
                    action='PARK_THREAD', hidden_quality=999)
    assert featurize(obs) == featurize(poisoned)
    assert not {'case_id', 'episode_id', 'step', 'reward', 'action_prob',
                'done', 'info', 'next_observation', 'action', 'hidden_quality'} & featurize(obs).keys()


def test_features_derive_history_from_visible_actions_without_mutation():
    obs = {'turn': 3, 'previous_actions': ['WAIT', 'ASK_OBJECTION', 'ASK_OBJECTION'],
           'last_action': 'WAIT', 'days_since_last_outbound': 99,
           'days_since_last_inbound': 0, 'user_patience': 0}
    before = copy.deepcopy(obs)
    f = featurize(obs)
    assert f['last_action'] == 'ASK_OBJECTION'
    assert f['previous_count_WAIT'] == 1
    assert f['previous_count_ASK_OBJECTION'] == 2
    assert f['previous_count_CHECK_IN'] == 0
    assert f['consecutive_repeat_count'] == 2
    assert f['remaining_decisions'] == 1
    assert f['days_since_last_outbound'] == 99
    assert f['days_since_last_outbound_is_99'] == 1
    assert f['days_since_last_inbound_is_99'] == 0
    assert f['user_patience'] == 0
    assert obs == before


def test_missing_and_invalid_fields_are_explicit_and_last_action_has_fallback():
    f = featurize({'last_action': 'CHECK_IN', 'service_risk_score': float('inf'),
                  'relationship_score': 'not a number', 'weather_extreme': True})
    assert f['last_action'] == 'CHECK_IN'
    assert f['topic'] == '__missing__'
    assert f['service_risk_score'] is None
    assert f['relationship_score'] is None
    assert f['turn'] is None
    assert f['remaining_decisions'] is None
    assert f['weather_extreme'] == 1
    assert f['consecutive_repeat_count'] is None
    assert f['previous_count_WAIT'] is None
    empty = featurize({'turn': 0, 'previous_actions': []})
    assert empty['previous_count_WAIT'] == 0
    assert empty['consecutive_repeat_count'] == 0
    assert empty['remaining_decisions'] == 4


def test_numeric_imputation_uses_fit_only_medians_with_stable_missing_flags():
    encoder = FeatureEncoder().fit([{'day': 2}, {'day': 8}])
    names = list(encoder.get_feature_names_out())
    x = encoder.transform([{'day': None}, {'day': 1000}, {}])
    assert isinstance(x, np.ndarray)
    assert np.isfinite(x).all()
    np.testing.assert_array_equal(x[:, names.index('day')], [5, 1000, 5])
    np.testing.assert_array_equal(x[:, names.index('day_is_missing')], [1, 0, 1])
    # A wholly missing fit column has deterministic zero imputation and stays present.
    np.testing.assert_array_equal(x[:, names.index('urgency_score')], [0, 0, 0])
    np.testing.assert_array_equal(x[:, names.index('urgency_score_is_missing')], [1, 1, 1])
    assert encoder.transform([{}])[0, names.index('day')] == 5


def test_unknown_one_hot_category_is_ignored_and_missing_has_own_category():
    encoder = FeatureEncoder().fit([{'topic': 'open_estimate'}, {'topic': None}])
    names = list(encoder.get_feature_names_out())
    x = encoder.transform([{'topic': 'unseen_topic'}, {}])
    topic_columns = [i for i, name in enumerate(names) if name.startswith('topic_')]
    assert x[0, topic_columns].sum() == 0
    assert x[1, names.index('topic___missing__')] == 1


def test_encoding_is_identical_after_serialization_and_input_reordering(tmp_path):
    observations = [{'day': 1, 'topic': 'open_estimate'}, {'day': 9, 'topic': 'service_issue'}]
    encoder = FeatureEncoder()
    fitted = encoder.fit_transform(observations)
    np.testing.assert_array_equal(fitted, encoder.transform(observations))
    path = tmp_path / 'encoder.joblib'
    joblib.dump(encoder, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(restored.transform(observations), fitted)
    np.testing.assert_array_equal(encoder.transform([dict(reversed(list(o.items()))) for o in observations]), fitted)
    assert len(encoder.get_feature_names_out()) == fitted.shape[1]


def test_encoder_does_not_fit_on_transform_and_rejects_empty_fit():
    with pytest.raises(NotFittedError):
        FeatureEncoder().transform([{}])
    with pytest.raises(ValueError, match='empty'):
        FeatureEncoder().fit([])


def test_encoder_ignores_forbidden_fields_at_fit_and_inference():
    clean = [{'turn': 0, 'day': 1}, {'turn': 1, 'day': 3}]
    poisoned = [dict(obs, case_id=str(i), episode_id='training-only',
                     reward=1000 * i, action_prob=0.1,
                     info={'outcome': 'sold'}, next_observation={'day': 10000})
                for i, obs in enumerate(clean)]
    clean_encoder = FeatureEncoder().fit(clean)
    poisoned_encoder = FeatureEncoder().fit(poisoned)
    np.testing.assert_array_equal(clean_encoder.get_feature_names_out(),
                                  poisoned_encoder.get_feature_names_out())
    np.testing.assert_array_equal(clean_encoder.transform(clean),
                                  poisoned_encoder.transform(poisoned))
    np.testing.assert_array_equal(clean_encoder.transform(clean),
                                  clean_encoder.transform(poisoned))


def test_optional_history_is_ignored_even_for_missing_visible_history():
    for obs in ({}, {'previous_actions': ['WAIT', 'WAIT']}):
        assert featurize(obs, [{'action': 'ASK_OBJECTION', 'reward': 999}]) == featurize(obs)
        assert featurize(obs, ['ESCALATE_TO_HUMAN']) == featurize(obs)


@pytest.mark.parametrize('previous, waits', [([], 0), (['WAIT', 'WAIT'], 2),
                                           (['WAIT', 'CHECK_IN', 'WAIT'], 1),
                                           (['WAIT', 'unknown'], 0)])
def test_explicit_wait_runs_and_occurred_indicators(previous, waits):
    f = featurize({'previous_actions': previous})
    assert f['consecutive_waits'] == waits
    for action in ('ASK_OBJECTION', 'ESTIMATE_NUDGE', 'OFFER_SCHEDULING'):
        assert f['occurred_' + action] == int(action in previous)
    missing = featurize({})
    assert missing['consecutive_waits'] is None
    assert missing['occurred_ASK_OBJECTION'] is None
