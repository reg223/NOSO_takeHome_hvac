"""Training-only checkpoint smoke checks; these are not policy-value estimates.

Run from the project root: .venv/bin/python scripts/check_checkpoint_one.py
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from starter.baselines import simple_rule
from submission.features import ACTIONS, FeatureEncoder, SCHEMA_VERSION, featurize
from submission.parse import load_episodes, validate_episode
from submission.rules import (DEFAULT_SERVICE_RISK_THRESHOLD, always_check_in,
                              always_estimate_nudge, eligible_actions,
                              fallback_action, service_override, simple_rule_action)


def main():
    train_path = ROOT / 'data/train_logs.jsonl'
    split_path = ROOT / 'artifacts/splits.json'
    splits = json.loads(split_path.read_text())
    train_hash = hashlib.sha256(train_path.read_bytes()).hexdigest()
    assert train_hash == splits['datasets']['train']['sha256']
    episodes = load_episodes(train_path)
    by_id = {ep[0]['episode_id']: ep for ep in episodes}
    roles = ('policy_fit', 'evaluator_fit', 'development_score')
    all_ids = [eid for role in roles for eid in splits[role]]
    assert len(all_ids) == len(set(all_ids)) == len(episodes)
    assert set(all_ids) == set(by_id)
    policies = {'simple_rule': simple_rule_action, 'always_check_in': always_check_in,
                'always_estimate_nudge': always_estimate_nudge, 'contextual_fallback': fallback_action}
    counts = {name: Counter() for name in policies}
    protected = 0
    poison = [{'case_id': 'unrelated', 'action': 'PARK_THREAD', 'reward': 10000,
               'done': True, 'next_observation': {'turn': 99}, 'info': {'outcome': 'sold'}}]
    for episode in episodes:
        validate_episode(episode)
        for row in episode:
            obs = row['observation']
            before = deepcopy(obs)
            for name, policy in policies.items():
                action = policy(obs, [])
                assert action in ACTIONS
                assert policy(obs, poison) == action
                counts[name][action] += 1
            assert simple_rule_action(obs, []) == simple_rule(obs, [])
            fallback = fallback_action(obs, [])
            assert fallback in eligible_actions(obs, [])
            assert len(eligible_actions(obs, [])) > 1
            override = service_override(obs, [])
            if override is not None:
                protected += 1
                assert fallback == override
            assert featurize(obs, []) == featurize(obs, poison)
            assert obs == before
    fit = [row['observation'] for eid in splits['policy_fit'] for row in by_id[eid]]
    dev = [row['observation'] for eid in splits['development_score'] for row in by_id[eid]]
    encoder = FeatureEncoder().fit(fit)
    medians = encoder.medians_.copy()
    encoded = encoder.transform(dev)
    assert np.isfinite(encoded).all()
    assert np.array_equal(medians, encoder.medians_)
    report = {
        'schema_version': 1, 'feature_schema_version': SCHEMA_VERSION,
        'train_sha256': train_hash,
        'splits_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
        'seed': splits['seed'], 'action_order': list(ACTIONS),
        'service_risk_threshold': DEFAULT_SERVICE_RISK_THRESHOLD,
        'episodes': len(episodes), 'rows': sum(map(len, episodes)),
        'role_counts': {role: len(splits[role]) for role in roles},
        'policy_action_counts': {name: dict(count) for name, count in counts.items()},
        'protected_observations': protected,
        'illegal_actions': 0, 'fallback_excluded': 0, 'service_violations': 0,
        'history_contamination': 0, 'observation_mutations': 0,
        'starter_reference_mismatches': 0,
        'encoder': {'fit_role': 'policy_fit', 'fit_rows': len(fit),
                    'transform_role': 'development_score', 'transform_rows': len(dev),
                    'features': encoded.shape[1], 'nonfinite_values': 0,
                    'medians_changed_on_transform': False},
        'limitations': 'Training-state replay and interface checks only; no policy-performance estimate, candidate learning, or validation scoring.',
    }
    output = ROOT / 'artifacts/task2_checks.json'
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
