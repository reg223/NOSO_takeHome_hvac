"""Train-only, fixed-policy finite-horizon evaluation (undiscounted gamma=1).

Importance weights multiply along the entire logged prefix. Sensitivities cap
raw cumulative weights, not propensities or recursively clipped weights.
Models and support counts are fitted exclusively on the passed evaluator-fit
role. All intervals are conditional on those models, not causal guarantees.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from itertools import combinations
import math

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from submission.features import ACTIONS, FeatureEncoder, VISIBLE_FIELDS, featurize
from submission.parse import validate_episode
from submission.rules import OBJECTION_SIGNALS

DEFAULT_CONFIG = dict(n_estimators=100, max_depth=8, min_samples_leaf=30,
                      random_state=20260621, n_jobs=1)
WEIGHT_CAPS = (10, 50, 100)
TOUCH_ACTIONS = frozenset({'CHECK_IN', 'ESTIMATE_NUDGE', 'ASK_OBJECTION',
                           'OFFER_SCHEDULING', 'MEMBERSHIP_TOUCH'})
INTERVAL_NOTE = ('95% percentile whole-episode bootstrap intervals conditional on fitted models; '
                 'they do not remove hidden confounding, misspecified propensities, or extrapolation bias.')


def _validate_episodes(episodes):
    if not episodes:
        raise ValueError('at least one complete episode is required')
    ids = set()
    for episode in episodes:
        validate_episode(episode)
        episode_id = episode[0]['episode_id']
        if episode_id in ids:
            raise ValueError('duplicate episode_id')
        ids.add(episode_id)
        for row in episode:
            cost = row['info'].get('cost_cents')
            if isinstance(cost, bool) or not isinstance(cost, (float, int)) or not math.isfinite(cost):
                raise ValueError('info.cost_cents must be a finite number')
            if row['done'] and not isinstance(row['info'].get('outcome'), str):
                raise ValueError('terminal info.outcome must be a string')


def _visible(observation):
    return deepcopy({key: observation[key] for key in VISIBLE_FIELDS if key in observation})


def _choose(policy, observation, history):
    fn = policy.act if hasattr(policy, 'act') else policy
    action = fn(_visible(observation), deepcopy(history))
    if not isinstance(action, str) or action not in ACTIONS:
        raise ValueError('policy must return one canonical action')
    return action


def _contexts(episodes):
    records = []
    for episode in episodes:
        history = []
        for row in episode:
            records.append((row, deepcopy(history)))
            history.append({'observation': _visible(row['observation']), 'action': row['action']})
    return records


def _metric_vector(row, metrics):
    values = {'reward': float(row['reward']), 'cost_cents': float(row['info']['cost_cents']),
              'touches': float(row['action'] in TOUCH_ACTIONS)}
    return np.asarray([values[name] if name in values else
                       float(row['done'] and row['info']['outcome'] == name[len('outcome:'):])
                       for name in metrics])


def initial_slices(observation):
    """Frozen populations defined only by the initial decision-time state."""
    f = featurize(observation)
    return {
        'service': f['topic'] == 'service_issue' or f['customer_signal'] == 'unhappy'
                   or (f['service_risk_score'] is not None and f['service_risk_score'] > .78),
        'open_estimate_objection': f['topic'] == 'open_estimate' and f['customer_signal'] in OBJECTION_SIGNALS,
        'cold_or_ignored': f['heat'] == 'cold' or (f['ignored_count'] is not None and f['ignored_count'] >= 1),
        'membership': f['topic'] == 'membership_renewal' or f['membership_status'] in {'expiring', 'lapsed'},
    }


def _support_key(observation, action):
    f = featurize(observation)
    return (f['turn'], f['topic'], f['customer_signal'], action)


class FixedPolicyEvaluator:
    """Backward fitted policy evaluation with a fixed canonical action encoding.

    The feature encoder is role-local: fitting it here on evaluator-fit inputs
    deliberately differs from the policy-training encoder. Missing turn models
    return zeros but are separately flagged as unsupported during scoring.
    """

    def __init__(self, encoder, metrics, config):
        self.encoder = encoder
        self.metrics = tuple(metrics)
        self.config = dict(config)
        self.models = {}
        self.support_counts = Counter()
        self.action_counts_by_turn = defaultdict(Counter)
        self.training_targets_ = {}
        self.metadata = {'action_order': list(ACTIONS), 'metrics': list(metrics),
                         'config': dict(config), 'gamma': 1.,
                         'support_key': ['turn', 'topic', 'customer_signal', 'action'],
                         'touches_definition': sorted(TOUCH_ACTIONS),
                         'missing_turn_model': 'zero prediction, explicitly unsupported',
                         'weight_sensitivity': 'importance weight clipping at raw cumulative 10/50/100'}

    def _inputs(self, observations, actions):
        action_indices = [ACTIONS.index(action) for action in actions]
        return np.hstack((self.encoder.transform(observations), np.eye(len(ACTIONS))[action_indices]))

    def predict_batch(self, observations, histories, actions):
        del histories  # Shared encoder uses visible previous_actions, not external history.
        if len(observations) != len(actions):
            raise ValueError('observations and actions must have equal lengths')
        result = np.zeros((len(observations), len(self.metrics)))
        groups = defaultdict(list)
        for index, observation in enumerate(observations):
            groups[observation['turn']].append(index)
        for turn, indices in groups.items():
            if turn in self.models:
                x = self._inputs([observations[i] for i in indices], [actions[i] for i in indices])
                result[indices] = np.asarray(self.models[turn].predict(x)).reshape(len(indices), len(self.metrics))
        return result

    def support_count(self, observation, action):
        return self.support_counts[_support_key(observation, action)]


def fit_fixed_policy_evaluator(episodes, policy, config=None):
    """Fit gamma=1 reward + Q(next, frozen policy(next)), never max_a Q.

    Pass only the evaluator-fit role. Terminal next_observation, even if present,
    is ignored for all outcomes, including max_steps.
    """
    _validate_episodes(episodes)
    settings = dict(DEFAULT_CONFIG)
    if config:
        unknown = set(config) - set(settings)
        if unknown:
            raise ValueError('unknown evaluator config keys: ' + ', '.join(sorted(unknown)))
        settings.update(config)
    records = _contexts(episodes)
    metrics = ('reward', 'cost_cents', 'touches') + tuple(
        'outcome:' + name for name in sorted({ep[-1]['info']['outcome'] for ep in episodes}))
    evaluator = FixedPolicyEvaluator(FeatureEncoder().fit([r['observation'] for r, _ in records]),
                                     metrics, settings)
    evaluator.policy = policy
    evaluator.metadata['fit_episode_ids'] = sorted(ep[0]['episode_id'] for ep in episodes)
    evaluator.training_targets_ = {ep[0]['episode_id']: [None] * len(ep) for ep in episodes}
    by_turn = defaultdict(list)
    for row, history in records:
        by_turn[row['step']].append((row, history))
        evaluator.support_counts[_support_key(row['observation'], row['action'])] += 1
        evaluator.action_counts_by_turn[row['step']][row['action']] += 1
    for turn in sorted(by_turn, reverse=True):
        current = by_turn[turn]
        targets = np.asarray([_metric_vector(row, metrics) for row, _ in current])
        continuation_indices = [i for i, (row, _) in enumerate(current) if not row['done']]
        next_obs, next_hist, next_actions = [], [], []
        for i in continuation_indices:
            row, history = current[i]
            history = history + [{'observation': _visible(row['observation']), 'action': row['action']}]
            next_obs.append(row['next_observation'])
            next_hist.append(history)
            next_actions.append(_choose(policy, row['next_observation'], history))
        if continuation_indices:
            targets[continuation_indices] += evaluator.predict_batch(next_obs, next_hist, next_actions)
        for (row, _), target in zip(current, targets):
            evaluator.training_targets_[row['episode_id']][row['step']] = target.tolist()
        model = ExtraTreesRegressor(**settings)
        x = evaluator._inputs([r['observation'] for r, _ in current], [r['action'] for r, _ in current])
        model.fit(x, targets)
        evaluator.models[turn] = model
    evaluator.metadata['support_counts'] = [
        dict(zip(('turn', 'topic', 'customer_signal', 'action', 'count'), (*key, count)))
        for key, count in sorted(evaluator.support_counts.items())]
    evaluator.metadata['action_counts_by_turn'] = {
        str(turn): {action: counts[action] for action in ACTIONS}
        for turn, counts in sorted(evaluator.action_counts_by_turn.items())}
    return evaluator


def _predict(evaluator, observations, histories, actions):
    result = np.asarray(evaluator.predict_batch(observations, histories, actions), dtype=float)
    if result.shape != (len(observations), len(evaluator.metrics)) or not np.all(np.isfinite(result)):
        raise ValueError('evaluator predictions must be finite and match metric schema')
    return result


def score_episodes(policy, episodes, evaluator):
    """Return additive episode contributions; batch all Q predictions by turn.

    Direct/DR values remain visible in zero-support cells as explicitly labeled
    extrapolations. This evaluator is diagnostic and cannot authorize selection.
    """
    _validate_episodes(episodes)
    if hasattr(evaluator, 'policy') and evaluator.policy is not policy:
        raise ValueError('scoring policy must be the same frozen policy used to fit the evaluator')
    for episode in episodes:
        if 'outcome:' + episode[-1]['info']['outcome'] not in evaluator.metrics:
            raise ValueError('score-only terminal outcome missing from evaluator-fit metric schema')
    records = _contexts(episodes)
    observations = [row['observation'] for row, _ in records]
    histories = [history for _, history in records]
    candidate_actions = [_choose(policy, obs, hist) for obs, hist in zip(observations, histories)]
    values = _predict(evaluator, observations, histories, candidate_actions)
    q_logged = _predict(evaluator, observations, histories, [r['action'] for r, _ in records])
    results, offset = [], 0
    for episode in episodes:
        n = len(episode)
        v, q = values[offset:offset+n], q_logged[offset:offset+n]
        actions = candidate_actions[offset:offset+n]
        immediate = np.asarray([_metric_vector(row, evaluator.metrics) for row in episode])
        estimates = {'logged': immediate.sum(axis=0), 'direct': v[0].copy(),
                     'pdis': np.zeros(len(evaluator.metrics)), 'dr': v[0].copy()}
        for cap in WEIGHT_CAPS:
            estimates['pdis_cap_' + str(cap)] = np.zeros(len(evaluator.metrics))
            estimates['dr_cap_' + str(cap)] = v[0].copy()
        weight, steps, exhausted = 1., [], 0
        for t, row in enumerate(episode):
            weight *= float(actions[t] == row['action']) / row['action_prob']
            if not math.isfinite(weight):
                raise ValueError('cumulative importance weight overflow')
            next_value = np.zeros(len(evaluator.metrics)) if row['done'] else v[t+1]
            residual = immediate[t] + next_value - q[t]
            estimates['pdis'] += weight * immediate[t]
            estimates['dr'] += weight * residual
            for cap in WEIGHT_CAPS:
                clipped = min(weight, cap)
                estimates['pdis_cap_' + str(cap)] += clipped * immediate[t]
                estimates['dr_cap_' + str(cap)] += clipped * residual
            count = int(evaluator.support_count(row['observation'], actions[t]))
            obs = featurize(row['observation'])
            exhausted += int(t > 0 and ((obs['ignored_count'] is not None and obs['ignored_count'] >= 2)
                                        or (obs['user_patience'] is not None and obs['user_patience'] <= 0)))
            steps.append({'turn': t, 'weight': weight, 'propensity': row['action_prob'],
                          'candidate_action': actions[t], 'logged_action': row['action'],
                          'support_count': count, 'zero_support': count == 0,
                          'model_extrapolation': count == 0})
        results.append({'episode_id': episode[0]['episode_id'],
                        'estimates': {method: dict(zip(evaluator.metrics, map(float, vals)))
                                      for method, vals in estimates.items()},
                        'steps': steps, 'initial_slices': initial_slices(episode[0]['observation']),
                        'later_replay': {'exhausted_decisions': exhausted},
                        'model_extrapolation': any(step['zero_support'] for step in steps)})
        offset += n
    return results


def _aligned(scores):
    if not scores:
        raise ValueError('at least one score required')
    mapping = {score['episode_id']: score for score in scores}
    if len(mapping) != len(scores):
        raise ValueError('duplicate score episode_id')
    return [mapping[key] for key in sorted(mapping)]


def _columns(scores):
    methods = list(scores[0]['estimates'])
    metrics = list(scores[0]['estimates'][methods[0]])
    for score in scores:
        if set(score['estimates']) != set(methods) or any(
            set(score['estimates'][method]) != set(metrics) for method in methods
        ):
            raise ValueError('inconsistent score estimate schema')
    values = np.asarray([[[score['estimates'][method][metric] for metric in metrics]
                           for method in methods] for score in scores], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError('scores must be finite')
    return methods, metrics, values


def _estimates(values, methods, metrics, samples):
    means = values.mean(axis=0)
    # Avoid a reps x episodes x methods x metrics temporary on full dev data.
    bootstrap = np.asarray([values[indices].mean(axis=0) for indices in samples])
    low, high = np.percentile(bootstrap, [2.5, 97.5], axis=0)
    return {method: {metric: {'mean': float(means[i, j]),
                             'ci95': [float(low[i, j]), float(high[i, j])]}
                     for j, metric in enumerate(metrics)} for i, method in enumerate(methods)}


def _weight_summary(scores):
    grouped = defaultdict(list)
    for score in scores:
        for step in score['steps']:
            grouped[step['turn']].append(step['weight'])
    result = {}
    for turn, weights in sorted(grouped.items()):
        raw = np.asarray(weights)
        item = {'observed_decisions': len(raw)}
        for cap in (None,) + WEIGHT_CAPS:
            w = raw if cap is None else np.minimum(raw, cap)
            total, squares = float(w.sum()), float(w @ w)
            stats = {'ess': total * total / squares if squares else 0.,
                     'max_weight': float(w.max()), 'max_normalized_weight': float(w.max() / total) if total else 0.,
                     'nonzero_count': int(np.count_nonzero(w)), 'sum_weights': total}
            if cap is None:
                item.update(stats)
            else:
                item['cap_' + str(cap)] = stats
        result[str(turn)] = item
    return result


def _single_summary(scores, samples, seed, bootstrap_reps, include_slices=True):
    methods, metrics, values = _columns(scores)
    steps = [step for score in scores for step in score['steps']]
    zeros = sum(step['zero_support'] for step in steps)
    output = {'episodes': len(scores), 'estimates': _estimates(values, methods, metrics, samples),
              'interval_note': INTERVAL_NOTE, 'bootstrap_seed': seed, 'bootstrap_reps': bootstrap_reps,
              'weights_by_turn': _weight_summary(scores),
              'support': {'zero_support_decisions': zeros, 'decisions': len(steps),
                          'zero_support_fraction': zeros / len(steps),
                          'episodes_with_extrapolation': sum(s['model_extrapolation'] for s in scores),
                          'note': 'Broad cell counts are diagnostics, not proof of overlap; zero-support direct/DR values extrapolate.'},
              'propensity': {'min': min(s['propensity'] for s in steps),
                             'max': max(s['propensity'] for s in steps)},
              'later_replay': {'exhausted_decisions': sum(s['later_replay']['exhausted_decisions'] for s in scores),
                               'note': 'Later logged-state replay counts only, not causal slice estimates.'},
              'weight_sensitivity_note': 'importance weight clipping: cap raw cumulative weight independently at 10/50/100',
              'slices': {}}
    if include_slices:
        for name in scores[0]['initial_slices']:
            selected = [score for score in scores if score['initial_slices'][name]]
            if selected:
                rng = np.random.default_rng(seed)
                sub_samples = rng.integers(len(selected), size=(bootstrap_reps, len(selected)))
                output['slices'][name] = _single_summary(selected, sub_samples, seed, bootstrap_reps, False)
            else:
                output['slices'][name] = {'episodes': 0, 'estimates': {}}
    return output


def summarize(scores, seed=20260621, bootstrap_reps=1000):
    """Summarize one policy or a mapping with ID-aligned paired comparisons.

    Policies share a single ordered-ID bootstrap sample matrix. Initial-state
    slice summaries also reuse deterministic ID-aligned samples within slices.
    """
    if isinstance(bootstrap_reps, bool) or not isinstance(bootstrap_reps, int) or bootstrap_reps < 1:
        raise ValueError('bootstrap_reps must be a positive integer')
    is_mapping = isinstance(scores, dict)
    policies = {name: _aligned(value) for name, value in scores.items()} if is_mapping else {'policy': _aligned(scores)}
    if not policies:
        raise ValueError('at least one policy required')
    first = next(iter(policies.values()))
    ids = [score['episode_id'] for score in first]
    methods, metrics, _ = _columns(first)
    for value in policies.values():
        if [score['episode_id'] for score in value] != ids:
            raise ValueError('paired policies must contain the same episode IDs')
        other_methods, other_metrics, _ = _columns(value)
        if other_methods != methods or other_metrics != metrics:
            raise ValueError('paired policies must use identical metric/method order')
        if any(a['initial_slices'] != b['initial_slices'] for a, b in zip(first, value)):
            raise ValueError('paired policies must use identical initial populations')
    samples = np.random.default_rng(seed).integers(len(ids), size=(bootstrap_reps, len(ids)))
    summaries = {name: _single_summary(value, samples, seed, bootstrap_reps) for name, value in policies.items()}
    if not is_mapping:
        return summaries['policy']
    paired = {}
    for left, right in combinations(policies, 2):
        difference = _columns(policies[right])[2] - _columns(policies[left])[2]
        paired[right + '-minus-' + left] = _estimates(difference, methods, metrics, samples)
    paired_slices = {}
    for name in first[0]['initial_slices']:
        indices = [i for i, score in enumerate(first) if score['initial_slices'][name]]
        if not indices:
            paired_slices[name] = {'episodes': 0, 'paired_differences': None}
            continue
        sub_samples = np.random.default_rng(seed).integers(len(indices), size=(bootstrap_reps, len(indices)))
        differences = {}
        for left, right in combinations(policies, 2):
            difference = (_columns(policies[right])[2] - _columns(policies[left])[2])[indices]
            differences[right + '-minus-' + left] = _estimates(difference, methods, metrics, sub_samples)
        paired_slices[name] = {'episodes': len(indices), 'paired_differences': differences}
    return {'policies': summaries, 'paired_differences': paired, 'paired_episode_ids': ids,
            'paired_slices': paired_slices,
            'interval_note': INTERVAL_NOTE, 'bootstrap_seed': seed, 'bootstrap_reps': bootstrap_reps}
