import numpy as np
import pytest

from submission.audit import audit, make_splits, check_disjoint


def episode(i, case=None):
    return [{'episode_id': f'ep_{i:05}', 'step': 0,
             'observation': {'turn': 0, 'case_id': case or f'case_{i}',
                             'topic': 'open_estimate', 'customer_signal': 'none',
                             'days_since_last_outbound': 99, 'days_since_last_inbound': 99},
             'action': 'WAIT', 'action_prob': 0.25, 'reward': -7,
             'done': True, 'next_observation': {'turn': 1},
             'info': {'cost_cents': 0.1, 'outcome': 'max_steps'}}]


def test_audit_preserves_rewards_costs_and_terminal_states():
    result = audit([episode(0), episode(1)])
    assert result['episodes'] == 2
    assert result['rows'] == 2
    assert result['reward_quantiles']['0.0'] == -7
    assert result['cost_cents_total'] == pytest.approx(0.2)
    assert result['terminal_next_observation']['object'] == 2
    assert result['outcome_counts'] == {'max_steps': 2}
    assert result['recency']['days_since_last_outbound']['is_99_count'] == 2


def test_seeded_split_matches_declared_numpy_algorithm():
    episodes = [episode(i) for i in range(24000)]
    result = make_splits(episodes)
    expected = sorted(ep[0]['episode_id'] for ep in episodes)
    np.random.default_rng(20260621).shuffle(expected)
    assert result['policy_fit'] == expected[:16800]
    assert result['evaluator_fit'] == expected[16800:20400]
    assert result['development_score'] == expected[20400:]
    assert make_splits(list(reversed(episodes))) == result
    assert len(set(sum((result[k] for k in ('policy_fit', 'evaluator_fit', 'development_score')), []))) == 24000


def test_case_groups_never_cross_roles():
    episodes = [episode(i, f'case_{i // 3}') for i in range(100)]
    splits = make_splits(episodes)
    role_by_id = {eid: role for role in ('policy_fit', 'evaluator_fit', 'development_score') for eid in splits[role]}
    case_roles = {}
    for ep in episodes:
        case_roles.setdefault(ep[0]['observation']['case_id'], set()).add(role_by_id[ep[0]['episode_id']])
    assert all(len(roles) == 1 for roles in case_roles.values())
    assert sum(splits['actual_counts'].values()) == 100


def test_cross_dataset_case_and_episode_overlap_rejected():
    with pytest.raises(ValueError, match='overlap'):
        check_disjoint([episode(0)], [episode(0)])
    with pytest.raises(ValueError, match='overlap'):
        check_disjoint([episode(0, 'same')], [episode(1, 'same')])


def test_audit_refuses_invalid_cost():
    ep = episode(0)
    ep[0]['info']['cost_cents'] = float('nan')
    with pytest.raises(ValueError, match='cost'):
        audit([ep])


def test_cli_quarantines_incomplete_and_saves_hashes(tmp_path):
    import hashlib
    import json
    import subprocess
    import sys

    train = tmp_path / 'train.jsonl'
    valid = tmp_path / 'valid.jsonl'
    incomplete = episode(2)[0]
    incomplete['done'] = False
    train.write_text('\n'.join(json.dumps(row) for row in [episode(0)[0], incomplete]) + '\n')
    valid.write_text(json.dumps(episode(1)[0]) + '\n')
    output, splits_path = tmp_path / 'audit.json', tmp_path / 'splits.json'
    command = [sys.executable, '-m', 'submission.audit', '--train', str(train), '--valid', str(valid),
               '--output', str(output), '--splits', str(splits_path)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    report, splits = json.loads(output.read_text()), json.loads(splits_path.read_text())
    assert report['train']['episodes'] == 1
    assert report['quarantine']['train'][0]['episode_id'] == 'ep_00002'
    assert splits['quarantined_episode_ids'] == ['ep_00002']
    assert report['datasets']['train']['sha256'] == hashlib.sha256(train.read_bytes()).hexdigest()
    assert set(eid for role in ROLES_FOR_TEST for eid in splits[role]) == {'ep_00000'}
    saved = output.read_bytes(), splits_path.read_bytes()
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert saved == (output.read_bytes(), splits_path.read_bytes())


ROLES_FOR_TEST = ('policy_fit', 'evaluator_fit', 'development_score')


def test_case_ids_missing_initially_still_grouped():
    from submission.audit import case_id
    ep = episode(0)
    ep[0]['observation'].pop('case_id')
    later = episode(0, 'known')[0]
    later['step'] = later['observation']['turn'] = 1
    ep[0]['done'] = False
    ep[0]['next_observation'] = later['observation']
    ep.append(later)
    assert case_id(ep) == 'known'


def test_planned_id_split_interface_is_seeded_and_complete():
    from submission import audit as module
    assert hasattr(module, 'write_splits'), 'planned ID split interface is missing'
    ids = [f'ep_{i:05}' for i in range(24000)]
    result = module.write_splits(ids)
    grouped = make_splits([episode(i) for i in range(24000)])
    assert result == {role: grouped[role] for role in ROLES_FOR_TEST}
    assert module.write_splits(list(reversed(ids))) == result
    assert module.write_splits(ids, seed=7) != result
    with pytest.raises(ValueError, match='duplicate'):
        module.write_splits(['a', 'a'])


def test_path_audit_interface(tmp_path):
    import json
    train, valid = tmp_path / 'train.jsonl', tmp_path / 'valid.jsonl'
    train.write_text(json.dumps(episode(0)[0]) + '\n')
    valid.write_text(json.dumps(episode(1)[0]) + '\n')
    report = audit(str(train), str(valid))
    assert report['train']['episodes'] == report['valid']['episodes'] == 1
    assert report['cross_dataset_overlap'] == {'episode_ids': 0, 'case_ids': 0}
    valid.write_text(train.read_text())
    with pytest.raises(ValueError, match='overlap'):
        audit(str(train), str(valid))
