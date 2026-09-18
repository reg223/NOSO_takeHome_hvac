"""Catch accidental role leakage or data replacement before evaluator fitting."""
import hashlib
import json
import pytest


def fixture(tmp_path):
    episodes = [
        [{'episode_id': name, 'observation': {'case_id': name}}]
        for name in ('fit', 'score', 'policy')
    ]
    path = tmp_path / 'train.jsonl'
    path.write_text('fixture dataset\n')
    manifest = {
        'datasets': {'train': {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}},
        'evaluator_fit': ['fit'], 'development_score': ['score'], 'policy_fit': ['policy'],
        'actual_counts': {'evaluator_fit': 1, 'development_score': 1, 'policy_fit': 1},
    }
    return episodes, path, manifest


def test_runner_accepts_only_complete_disjoint_roles_and_matching_hash(tmp_path):
    from scripts import evaluate_development as runner
    episodes, path, manifest = fixture(tmp_path)
    roles = runner.validate_manifest(episodes, manifest, path)
    assert roles['evaluator_fit'][0][0]['episode_id'] == 'fit'
    assert roles['development_score'][0][0]['episode_id'] == 'score'
    path.write_text('replaced dataset\n')
    with pytest.raises(ValueError, match='hash'):
        runner.validate_manifest(episodes, manifest, path)


@pytest.mark.parametrize('change', ['overlap', 'missing', 'extra', 'case', 'empty'])
def test_runner_rejects_role_leakage_and_incomplete_manifests(tmp_path, change):
    from scripts import evaluate_development as runner
    episodes, path, manifest = fixture(tmp_path)
    if change == 'overlap':
        manifest['development_score'] = ['fit']
    elif change == 'missing':
        manifest['policy_fit'] = []
    elif change == 'extra':
        manifest['policy_fit'].append('unknown')
    elif change == 'case':
        episodes[1][0]['observation']['case_id'] = 'fit'
    else:
        manifest['evaluator_fit'] = []
    with pytest.raises(ValueError):
        runner.validate_manifest(episodes, manifest, path)


def test_complete_runner_writes_finite_results_for_all_frozen_baselines(tmp_path):
    from scripts import evaluate_development as runner
    rows = []
    for name in ('fit', 'score', 'policy'):
        obs = {'turn': 0, 'case_id': name, 'topic': 'open_estimate',
               'customer_signal': 'none', 'track_hint': 'deal', 'membership_status': 'none',
               'estimate_value_bucket': 'medium', 'ignored_count': 0, 'touch_count': 0,
               'user_patience': 2, 'urgency_score': 0.1, 'service_risk_score': 0.0,
               'relationship_score': 0.1, 'previous_actions': []}
        rows.append({'episode_id': name, 'step': 0, 'observation': obs,
                     'action': 'CHECK_IN', 'action_prob': 1.0, 'reward': 4.0,
                     'done': True, 'next_observation': None,
                     'info': {'outcome': 'max_steps', 'cost_cents': 0.2}})
    train = tmp_path / 'train.jsonl'
    train.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    manifest = {
        'datasets': {'train': {'sha256': hashlib.sha256(train.read_bytes()).hexdigest()}},
        'evaluator_fit': ['fit'], 'development_score': ['score'], 'policy_fit': ['policy'],
        'actual_counts': {'evaluator_fit': 1, 'development_score': 1, 'policy_fit': 1},
    }
    splits, output = tmp_path / 'splits.json', tmp_path / 'results.json'
    splits.write_text(json.dumps(manifest))
    runner.run(train, splits, output)
    result = json.loads(output.read_text())
    assert result['data']['validation_accessed'] is False
    assert set(result['results']['policies']) == {
        'always_check_in', 'always_estimate_nudge', 'simple_rule', 'contextual_fallback'}
    matched = result['results']['policies']['always_check_in']['estimates']
    assert matched['pdis']['reward']['mean'] == 4.0
    assert matched['dr']['reward']['mean'] == pytest.approx(4.0)
    assert matched['logged']['cost_cents']['mean'] == pytest.approx(0.2)
