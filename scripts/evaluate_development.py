"""Reproduce checkpoint-two baseline estimates using frozen training roles only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from submission.audit import ROLES, case_id
from submission.parse import load_episodes


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(episodes, manifest, train_path):
    """Refuse changed datasets, overlapping roles, and case contamination."""
    if sha256(train_path) != manifest['datasets']['train']['sha256']:
        raise ValueError('training dataset hash does not match frozen manifest')
    by_id = {ep[0]['episode_id']: ep for ep in episodes}
    assigned = [eid for role in ROLES for eid in manifest[role]]
    if len(by_id) != len(episodes) or len(assigned) != len(set(assigned)):
        raise ValueError('duplicate episodes or overlapping roles')
    if set(assigned) != set(by_id):
        raise ValueError('roles must cover all and only the training episodes')
    roles = {role: [by_id[eid] for eid in manifest[role]] for role in ROLES}
    if not roles['evaluator_fit'] or not roles['development_score']:
        raise ValueError('evaluator-fit and development-score roles must be nonempty')
    if {role: len(rows) for role, rows in roles.items()} != manifest['actual_counts']:
        raise ValueError('role counts differ from frozen manifest')
    cases = {}
    for role, members in roles.items():
        for ep in members:
            case = case_id(ep)
            if case is not None:
                if case in cases and cases[case] != role:
                    raise ValueError('case overlap between development roles')
                cases[case] = role
    return roles


def run(train_path, split_path, output_path):
    import numpy as np
    import sklearn
    from submission.evaluate import fit_fixed_policy_evaluator, score_episodes, summarize
    from submission.features import ACTIONS, SCHEMA_VERSION
    from submission.rules import (always_check_in, always_estimate_nudge,
                                  fallback_action, simple_rule_action)

    manifest = json.loads(Path(split_path).read_text())
    roles = validate_manifest(load_episodes(train_path), manifest, train_path)
    config = {'n_estimators': 100, 'max_depth': 8, 'min_samples_leaf': 30,
              'random_state': 20260621, 'n_jobs': 1}
    policies = {'always_check_in': always_check_in,
                'always_estimate_nudge': always_estimate_nudge,
                'simple_rule': simple_rule_action,
                'contextual_fallback': fallback_action}
    scores, evaluators = {}, {}
    for name, policy in policies.items():
        print(f'Fit and score fixed policy: {name}', flush=True)
        evaluator = fit_fixed_policy_evaluator(roles['evaluator_fit'], policy, config)
        scores[name] = score_episodes(policy, roles['development_score'], evaluator)
        evaluators[name] = evaluator.metadata
    print('Summarize paired full-episode bootstrap and slices', flush=True)
    summary = summarize(scores, seed=20260621, bootstrap_reps=1000)
    source_paths = ['submission/evaluate.py', 'submission/features.py',
                    'submission/rules.py', 'submission/parse.py', 'submission/audit.py',
                    'scripts/evaluate_development.py', 'docs/checkpoint-two-protocol.md']
    try:
        revision = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    result = {
        'schema_version': 1,
        'purpose': 'Development-only fixed-baseline evaluation; no policy promotion',
        'protocol': 'docs/checkpoint-two-protocol.md',
        'reproduction': '.venv/bin/python scripts/evaluate_development.py',
        'data': {'train_sha256': sha256(train_path), 'splits_sha256': sha256(split_path),
                 'roles': {role: manifest[role] for role in ROLES},
                 'role_counts': manifest['actual_counts'], 'validation_accessed': False},
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                    'scikit_learn': sklearn.__version__},
        'source': {'git_revision': revision,
                   'sha256': {path: sha256(ROOT / path) for path in source_paths}},
        'settings': {'evaluator': config, 'bootstrap_seed': 20260621,
                     'bootstrap_reps': 1000, 'gamma': 1.0,
                     'action_order': list(ACTIONS), 'feature_schema_version': SCHEMA_VERSION,
                     'service_risk_threshold': 0.78, 'cumulative_weight_caps': [10, 50, 100]},
        'evaluators': evaluators,
        'results': summary,
        'limitations': [
            'Intervals condition on fitted models and exclude evaluator-fit uncertainty.',
            'Only logged actions have observed outcomes; changed actions change later states.',
            'Propensity misspecification and hidden confounding can bias both PDIS and DR.',
            'Coarse action support does not prove continuous-state overlap.',
            'Importance-weight clipping adds bias; zero matching prefixes are not proof of zero value.',
            'Direct estimates depend entirely on the fitted model; unsupported DR can inherit extrapolation.',
            'No validation scoring, policy tuning, promotion, or production rollout is authorized by this result.',
        ],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(f'Wrote {output}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=ROOT / 'data/train_logs.jsonl')
    parser.add_argument('--splits', type=Path, default=ROOT / 'artifacts/splits.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/dev_baselines.json')
    args = parser.parse_args()
    run(args.train, args.splits, args.output)


if __name__ == '__main__':
    main()
