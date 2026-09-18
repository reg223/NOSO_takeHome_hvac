"""Exact finite-horizon checks: expectations are hand-computed, not simulated."""
from copy import deepcopy
from itertools import product

import numpy as np
import pytest

from submission.evaluate import fit_fixed_policy_evaluator, score_episodes, summarize


A, B = 'CHECK_IN', 'WAIT'


def episode(name, actions=(A, A), rewards=(1., 3.), probs=(.5, .5)):
    observations = [dict(turn=t, topic='open_estimate', customer_signal='price_objection',
                         previous_actions=list(actions[:t]), heat='cold', user_patience=2,
                         ignored_count=0, membership_status='lapsed') for t in range(len(actions))]
    return [dict(episode_id=name, step=t, observation=obs, action=actions[t],
                 action_prob=probs[t], reward=rewards[t], done=t == len(actions)-1,
                 next_observation=observations[t+1] if t+1 < len(actions) else None,
                 info={'cost_cents': 2., 'outcome': 'max_steps' if t == len(actions)-1 else 'continue'})
            for t, obs in enumerate(observations)]


def policy(obs, history):
    return A


class KnownQ:
    metrics = ('reward', 'cost_cents', 'touches', 'outcome:max_steps')

    def __init__(self, wrong=False):
        self.wrong = wrong

    def predict_batch(self, observations, histories, actions):
        return np.array([[((10. if obs['turn'] == 0 else 7.) if self.wrong
                           else (4. if obs['turn'] == 0 else 3.)), 0., 0., 0.]
                         for obs in observations])

    def support_count(self, observation, action):
        return 0 if observation['turn'] == 1 else 5


def test_enumerated_two_action_two_step_pdis_and_dr():
    episodes = [episode(str(i), actions) for i, actions in enumerate(product((A, B), repeat=2))]
    scores = score_episodes(policy, episodes, KnownQ(wrong=True))
    assert [s['estimates']['pdis']['reward'] for s in scores] == [14., 2., 0., 0.]
    # Wrong Q: 10 + 2*(1+7-10) + 4*(3-7) = -10 on AA.
    assert [s['estimates']['dr']['reward'] for s in scores] == [-10., 6., 10., 10.]
    assert np.mean([s['estimates']['dr']['reward'] for s in scores]) == 4.
    assert scores[2]['steps'][1]['weight'] == 0  # BA never restarts.
    assert scores[0]['steps'][1]['zero_support'] is True
    assert scores[0]['estimates']['logged']['reward'] == 4.  # No double cost deduction.
    assert scores[0]['estimates']['logged']['cost_cents'] == 4.
    assert scores[0]['estimates']['logged']['touches'] == 2.
    assert scores[0]['estimates']['logged']['outcome:max_steps'] == 1.
    exact = score_episodes(policy, episodes, KnownQ())
    assert [s['estimates']['dr']['reward'] for s in exact] == [4.] * 4


def test_caps_apply_to_raw_cumulative_weights_independently():
    scored = score_episodes(policy, [episode('x', probs=(.05, .05))], KnownQ(wrong=True))[0]
    assert [s['weight'] for s in scored['steps']] == [20., 400.]
    assert scored['estimates']['pdis_cap_10']['reward'] == 40.
    assert scored['estimates']['pdis_cap_50']['reward'] == 170.
    assert scored['estimates']['pdis_cap_100']['reward'] == 320.
    assert scored['estimates']['dr_cap_10']['reward'] == -50.


def test_terminal_next_observation_never_continues_even_max_steps():
    ep = episode('x', actions=(A,), rewards=(-1.,), probs=(1.,))
    ep[0]['next_observation'] = {'turn': 1, 'reward': 9999}
    result = score_episodes(policy, [ep], KnownQ(wrong=True))[0]
    assert result['estimates']['dr']['reward'] == -1.


@pytest.mark.parametrize('bad', [0., -1., float('nan'), 1.1])
def test_rejects_invalid_propensities(bad):
    ep = episode('x'); ep[0]['action_prob'] = bad
    with pytest.raises(ValueError):
        score_episodes(policy, [ep], KnownQ())


def test_rejects_incomplete_and_duplicate_episodes():
    with pytest.raises(ValueError):
        score_episodes(policy, [episode('x')[:1]], KnownQ())
    with pytest.raises(ValueError):
        score_episodes(policy, [episode('x'), episode('x')], KnownQ())


def test_policy_sees_only_current_visible_fields_and_safe_prior_history():
    ep = episode('secret')
    for row in ep:
        row['observation'].update(case_id='secret', reward=900, info={'secret': 1})
    ep[0]['next_observation'] = deepcopy(ep[1]['observation'])
    def strict(obs, history):
        assert not {'case_id', 'reward', 'info', 'episode_id'} & obs.keys()
        for prior in history:
            assert set(prior) == {'observation', 'action'}
            assert not {'case_id', 'reward', 'info', 'done'} & prior['observation'].keys()
        # Mutation must not contaminate subsequent calls or source data.
        obs['topic'] = 'mutated'
        return A
    before = deepcopy(ep)
    score_episodes(strict, [ep], KnownQ())
    assert ep == before


def test_fixed_policy_backward_targets_and_fit_only_encoder():
    # Two terminal alternatives at turn1: target A gives 6, B gives 100.
    train = [episode('a', (A, A), (-1., 6.)), episode('b', (A, B), (-1., 100.)),
             episode('terminal', (A,), (-1.,), (1.,))]
    fitted = fit_fixed_policy_evaluator(train, policy, {'n_estimators': 10, 'min_samples_leaf': 1})
    targets = fitted.training_targets_
    assert targets['a'][0][0] == 5.
    assert targets['b'][0][0] == 5.
    assert targets['terminal'][0][0] == -1.
    assert fitted.support_count(train[0][1]['observation'], A) == 1
    assert fitted.support_count({'turn': 3}, A) == 0
    assert fitted.action_counts_by_turn[1][B] == 1
    before = fitted.encoder.medians_.copy()
    unknown = episode('heldout'); unknown[0]['observation']['day'] = 999999
    score_episodes(policy, [unknown], fitted)
    np.testing.assert_array_equal(before, fitted.encoder.medians_)
    assert fitted.metadata['action_order'] == ['WAIT', 'CHECK_IN', 'ESTIMATE_NUDGE', 'ASK_OBJECTION',
                                             'OFFER_SCHEDULING', 'MEMBERSHIP_TOUCH', 'PARK_THREAD', 'ESCALATE_TO_HUMAN']


def test_paired_bootstrap_aligns_ids_and_detects_missing_or_duplicate_ids():
    scores = score_episodes(policy, [episode('a'), episode('b', (B, A))], KnownQ(wrong=True))
    other = deepcopy(scores)
    for s in other:
        for values in s['estimates'].values():
            values['reward'] += 3.
    result = summarize({'one': scores, 'two': list(reversed(other))}, bootstrap_reps=100)
    paired = result['paired_differences']['two-minus-one']['dr']['reward']
    assert paired == {'mean': 3., 'ci95': [3., 3.]}
    assert result == summarize({'one': scores, 'two': other}, bootstrap_reps=100)
    with pytest.raises(ValueError):
        summarize({'one': scores, 'two': other[:1]}, bootstrap_reps=10)
    with pytest.raises(ValueError):
        summarize([scores[0], scores[0]], bootstrap_reps=10)


def test_summary_ess_slices_and_replay_are_labeled():
    eps = [episode('a'), episode('b', (B, A))]
    eps[0][1]['observation']['user_patience'] = 0
    eps[0][0]['next_observation'] = deepcopy(eps[0][1]['observation'])
    result = summarize(score_episodes(policy, eps, KnownQ()), bootstrap_reps=30)
    assert result['weights_by_turn']['1']['ess'] == 1.
    assert result['weights_by_turn']['1']['max_normalized_weight'] == 1.
    assert result['support']['zero_support_decisions'] == 2
    assert result['slices']['cold_or_ignored']['episodes'] == 2
    assert result['later_replay']['exhausted_decisions'] == 1
    assert 'conditional' in result['interval_note']


def test_score_only_outcome_cannot_silently_disappear():
    fitted = fit_fixed_policy_evaluator([episode('train')], policy, {'n_estimators': 2})
    unseen = episode('test')
    unseen[-1]['info']['outcome'] = 'unseen_terminal'
    with pytest.raises(ValueError, match='outcome'):
        score_episodes(policy, [unseen], fitted)


def test_scoring_rejects_a_different_fixed_policy():
    fitted = fit_fixed_policy_evaluator([episode('train')], policy, {'n_estimators': 2})
    with pytest.raises(ValueError, match='policy'):
        score_episodes(lambda obs, hist: B, [episode('test')], fitted)


def test_zero_match_ess_is_zero_without_nan():
    scores = score_episodes(policy, [episode('a', (B, B)), episode('b', (B, A))], KnownQ())
    result = summarize(scores, bootstrap_reps=10)
    assert result['weights_by_turn']['0']['ess'] == 0.
    assert result['weights_by_turn']['1']['nonzero_count'] == 0
    assert result['weights_by_turn']['1']['max_normalized_weight'] == 0.
    assert result['estimates']['pdis']['reward']['mean'] == 0.


def test_paired_slice_differences_use_shared_episode_samples():
    scores = score_episodes(policy, [episode('a'), episode('b', (B, A))], KnownQ())
    other = deepcopy(scores)
    for s in other:
        s['estimates']['dr']['reward'] += 2.
    result = summarize({'first': scores, 'second': other}, bootstrap_reps=20)
    sliced = result['paired_slices']['cold_or_ignored']
    assert sliced['episodes'] == 2
    assert sliced['paired_differences']['second-minus-first']['dr']['reward'] == {'mean': 2., 'ci95': [2., 2.]}
    assert result['paired_slices']['service']['episodes'] == 0
    assert result['paired_slices']['service']['paired_differences'] is None


def test_fit_metadata_records_support_cells_serializably():
    import json
    fitted = fit_fixed_policy_evaluator([episode('train')], policy, {'n_estimators': 2})
    metadata = json.loads(json.dumps(fitted.metadata))
    assert metadata['support_counts'][0]['count'] == 1
    assert metadata['action_counts_by_turn']['0'][A] == 1
