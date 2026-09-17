"""Reproducible integrity audit and training-only development roles.

Run from the project root with ``python -m submission.audit --help``.
Incomplete episodes are reported and excluded; every other integrity error aborts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np

SEED = 20260621
ROLES = ('policy_fit', 'evaluator_fit', 'development_score')


def quantiles(values):
    if not values:
        return {}
    levels = [0., .01, .05, .25, .5, .75, .95, .99, 1.]
    return {str(q): float(v) for q, v in zip(levels, np.quantile(values, levels))}


def case_id(episode):
    ids = {row['observation']['case_id'] for row in episode
           if row['observation'].get('case_id') is not None}
    if len(ids) > 1:
        raise ValueError('case_id must be consistent within an episode')
    return next(iter(ids), None)


def check_disjoint(train, valid):
    ids = lambda eps: {ep[0]['episode_id'] for ep in eps}
    cases = lambda eps: {case_id(ep) for ep in eps if case_id(ep) is not None}
    if ids(train) & ids(valid) or cases(train) & cases(valid):
        raise ValueError('training/validation episode or case ID overlap')


def make_splits(episodes):
    """Shuffle sorted IDs with Generator(PCG64); assign whole case groups.

    For repeated cases, visit groups in first-occurrence order in the shuffled
    episode list, filling roles to their cumulative target boundaries. Whole
    groups may overshoot a boundary; actual sizes are explicitly recorded.
    """
    by_id = {ep[0]['episode_id']: ep for ep in episodes}
    if len(by_id) != len(episodes):
        raise ValueError('duplicate episode ID in split input')
    ordered = sorted(by_id)
    np.random.default_rng(SEED).shuffle(ordered)
    groups = defaultdict(list)
    for eid in ordered:
        case = case_id(by_id[eid])
        groups[('case', case) if case is not None else ('episode', eid)].append(eid)
    n = len(ordered)
    targets = [int(n * .70), int(n * .15)]
    targets.append(n - sum(targets))
    result = {role: [] for role in ROLES}
    assigned = 0
    boundaries = [targets[0], sum(targets[:2])]
    for group in groups.values():
        role = ROLES[0 if assigned < boundaries[0] else 1 if assigned < boundaries[1] else 2]
        result[role].extend(group)
        assigned += len(group)
    result.update(seed=SEED, rng='numpy.random.default_rng (PCG64)',
                  requested_counts=dict(zip(ROLES, targets)),
                  actual_counts={role: len(result[role]) for role in ROLES},
                  repeated_case_groups=sum(len(g) > 1 for g in groups.values()),
                  grouping='Whole case groups in first shuffled occurrence order; cumulative boundaries may be exceeded.')
    return result


def audit(episodes):
    """Summarize validated complete episodes; no reward or cost reweighting."""
    counts = {name: Counter() for name in ('turn', 'action', 'topic', 'signal', 'outcome')}
    joint = Counter()
    rewards, props, costs, returns = [], [], [], []
    terminal_next, terminal_outcomes, terminal_actions = Counter(), Counter(), Counter()
    terminal_action_checks = {action: {'terminal': 0, 'nonterminal': 0}
                              for action in ('PARK_THREAD', 'ESCALATE_TO_HUMAN')}
    recency = {field: {'values': [], 'missing_count': 0, 'is_99_count': 0,
                      'is_99_by_turn': Counter(), 'is_99_by_signal': Counter(),
                      'is_99_by_touch_count': Counter(), 'is_99_by_last_action': Counter(),
                      'transitions_from_99': Counter(), 'transitions_from_99_by_action': Counter()}
               for field in ('days_since_last_inbound', 'days_since_last_outbound')}
    for ep in episodes:
        if not ep or not ep[-1]['done'] or any(row['done'] for row in ep[:-1]):
            raise ValueError('audit requires complete validated episodes')
        returns.append(math.fsum(row['reward'] for row in ep))
        for row in ep:
            obs = row['observation']
            if row['action'] in terminal_action_checks:
                terminal_action_checks[row['action']]['terminal' if row['done'] else 'nonterminal'] += 1
                if not row['done']:
                    raise ValueError(f"{row['action']} must terminate the episode")
            labels = (str(obs['turn']), row['action'], str(obs.get('topic', '__missing__')),
                      str(obs.get('customer_signal', '__missing__')))
            for key, value in zip(('turn', 'action', 'topic', 'signal'), labels):
                counts[key][value] += 1
            joint[labels] += 1
            outcome = str(row['info'].get('outcome', '__missing__'))
            counts['outcome'][outcome] += 1
            rewards.append(row['reward'])
            props.append(row['action_prob'])
            cost = row['info'].get('cost_cents')
            if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost):
                raise ValueError(f"invalid/missing cost_cents in episode {row['episode_id']} step {row['step']}")
            costs.append(cost)
            if row['done']:
                terminal_next['null' if row['next_observation'] is None else 'object'] += 1
                terminal_outcomes[outcome] += 1
                terminal_actions[row['action']] += 1
            for field, summary in recency.items():
                value = obs.get(field)
                if value is None:
                    summary['missing_count'] += 1
                    continue
                summary['values'].append(value)
                if value == 99:
                    summary['is_99_count'] += 1
                    for suffix, label in [('turn', obs['turn']), ('signal', obs.get('customer_signal')),
                                          ('touch_count', obs.get('touch_count')), ('last_action', obs.get('last_action'))]:
                        summary['is_99_by_' + suffix][str(label)] += 1
                    nxt = row['next_observation']
                    if isinstance(nxt, dict):
                        summary['transitions_from_99'][str(nxt.get(field))] += 1
                        summary['transitions_from_99_by_action'][row['action'] + ' -> ' + str(nxt.get(field))] += 1
    for summary in recency.values():
        summary['quantiles'] = quantiles(summary.pop('values'))
    cases = Counter(case_id(ep) for ep in episodes if case_id(ep) is not None)
    return {'episodes': len(episodes), 'rows': len(rewards), 'terminal_rows': sum(terminal_next.values()),
            **{key + '_counts': dict(sorted(value.items())) for key, value in counts.items()},
            'turn_action_topic_signal_counts': [dict(zip(('turn', 'action', 'topic', 'signal'), key), count=value)
                                               for key, value in sorted(joint.items())],
            'propensity_quantiles': quantiles(props), 'reward_quantiles': quantiles(rewards),
            'episode_return_quantiles': quantiles(returns), 'reward_total': math.fsum(rewards),
            'cost_cents_total': math.fsum(costs), 'terminal_outcome_counts': dict(terminal_outcomes),
            'terminal_action_counts': dict(terminal_actions), 'terminal_action_checks': terminal_action_checks, 'terminal_next_observation': dict(terminal_next),
            'unique_case_ids': len(cases), 'repeated_case_ids': sum(n > 1 for n in cases.values()),
            'recency': recency,
            'recency_interpretation': '99 is retained as an explicit indicator. Cross-tabs and observed next-state transitions describe association only; absent-contact semantics are not proven.',
            'integrity': {'status': 'passed for included episodes', 'terminal_actions_checked': ['PARK_THREAD', 'ESCALATE_TO_HUMAN'],
                          'continuation_authority': 'done', 'unexpected_failures': 0}}


def dataset_metadata(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': Path(path).stat().st_size}


def main():
    from submission.parse import load_episodes_with_quarantine

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True, type=Path)
    parser.add_argument('--valid', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--splits', required=True, type=Path)
    args = parser.parse_args()
    train, train_quarantine = load_episodes_with_quarantine(args.train)
    valid, valid_quarantine = load_episodes_with_quarantine(args.valid)
    check_disjoint(train, valid)
    datasets = {'train': dataset_metadata(args.train), 'valid': dataset_metadata(args.valid)}
    versions = {'python': platform.python_version(), 'numpy': np.__version__}
    report = {'schema_version': 1, 'datasets': datasets, 'runtime': versions,
              'train': audit(train), 'valid': audit(valid),
              'quarantine': {'train': train_quarantine, 'valid': valid_quarantine},
              'usable_episode_ids': {'train': [ep[0]['episode_id'] for ep in train],
                                     'valid': [ep[0]['episode_id'] for ep in valid]},
              'cross_dataset_overlap': {'episode_ids': 0, 'case_ids': 0},
              'validation_exposure': 'Structural checks and preliminary aggregate counts were inspected during planning. This audit additionally inspects aggregate rewards, outcomes, costs, coverage and recency. No validation outcomes are used to construct development splits or tune models.',
              'integrity_policy': 'Incomplete episodes are quarantined from return estimation and splits. All other integrity failures abort artifact generation; training must stop until treatment is documented.'}
    splits = {'schema_version': 1, 'datasets': datasets, 'runtime': versions, **make_splits(train),
              'quarantined_episode_ids': [item['episode_id'] for item in train_quarantine]}
    for path, payload in ((args.output, report), (args.splits, splits)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'train_episodes': len(train), 'valid_episodes': len(valid),
                      'quarantined': len(train_quarantine) + len(valid_quarantine),
                      'split_counts': splits['actual_counts']}, indent=2))


if __name__ == '__main__':
    main()
