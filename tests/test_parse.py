import json
import math

import pytest

from submission.parse import load_episodes, load_episodes_with_quarantine


def row(
    episode_id="episode-a",
    step=0,
    *,
    action="WAIT",
    action_prob=0.5,
    reward=1.0,
    done=False,
    observation=None,
    next_observation=None,
):
    if observation is None:
        observation = {"turn": step}
    if next_observation is None and not done:
        next_observation = {"turn": step + 1}
    return {
        "episode_id": episode_id,
        "step": step,
        "observation": observation,
        "action": action,
        "action_prob": action_prob,
        "reward": reward,
        "done": done,
        "next_observation": next_observation,
        "info": {},
    }


def complete_episode(episode_id="episode-a", *, terminal_action="WAIT"):
    return [
        row(episode_id, 0),
        row(
            episode_id,
            1,
            action=terminal_action,
            done=True,
            observation={"turn": 1},
            next_observation=None,
        ),
    ]


def write_rows(tmp_path, rows):
    path = tmp_path / "logs.jsonl"
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
    return path


def test_load_episodes_returns_ordered_raw_rows_and_allows_terminal_null(tmp_path):
    first = complete_episode("episode-a")
    second = complete_episode("episode-b", terminal_action="PARK_THREAD")
    path = write_rows(tmp_path, first + second)

    assert load_episodes(path) == [first, second]


def test_load_episodes_allows_a_terminal_next_observation_object(tmp_path):
    rows = complete_episode()
    rows[-1]["next_observation"] = {"turn": 2, "terminal_reason": "max_steps"}
    path = write_rows(tmp_path, rows)

    assert load_episodes(path) == [rows]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("action", "SEND_EMAIL", "action"),
        ("action_prob", 0, "action_prob"),
        ("action_prob", math.inf, "action_prob"),
        ("reward", math.nan, "reward"),
    ],
)
def test_load_episodes_rejects_invalid_action_reward_and_probability(tmp_path, field, value, match):
    rows = complete_episode()
    rows[0][field] = value
    path = write_rows(tmp_path, rows)

    with pytest.raises(ValueError, match=rf"line 1.*{match}"):
        load_episodes(path)


def test_load_episodes_reports_missing_required_key_with_path_and_line(tmp_path):
    rows = complete_episode()
    del rows[0]["reward"]
    path = write_rows(tmp_path, rows)

    with pytest.raises(ValueError, match=rf"{path}.*line 1.*reward"):
        load_episodes(path)


def test_load_episodes_requires_contiguous_step_and_observation_turns(tmp_path):
    rows = complete_episode()
    rows[1]["step"] = 0
    rows[1]["observation"] = {"turn": 0}
    path = write_rows(tmp_path, rows)

    with pytest.raises(ValueError, match=r"line 2.*contiguous"):
        load_episodes(path)


def test_load_episodes_rejects_observation_turn_that_disagrees_with_step(tmp_path):
    rows = complete_episode()
    rows[0]["observation"] = {"turn": 1}
    path = write_rows(tmp_path, rows)

    with pytest.raises(ValueError, match=r"line 1.*turn.*step"):
        load_episodes(path)


def test_load_episodes_requires_done_to_be_boolean_and_terminal_actions_to_finish(tmp_path):
    rows = complete_episode()
    rows[0]["done"] = 1
    path = write_rows(tmp_path, rows)

    with pytest.raises(ValueError, match=r"line 1.*done"):
        load_episodes(path)

    rows = complete_episode()
    rows[0]["action"] = "ESCALATE_TO_HUMAN"
    path = write_rows(tmp_path, rows)

    with pytest.raises(ValueError, match=r"line 1.*ESCALATE_TO_HUMAN"):
        load_episodes(path)


def test_load_episodes_rejects_next_observation_that_does_not_match_next_row(tmp_path):
    rows = complete_episode()
    rows[0]["next_observation"] = {"turn": 9}
    path = write_rows(tmp_path, rows)

    with pytest.raises(ValueError, match=r"line 1.*next_observation"):
        load_episodes(path)


def test_load_episodes_rejects_terminal_row_before_another_row(tmp_path):
    rows = complete_episode()
    rows[0]["done"] = True
    path = write_rows(tmp_path, rows)

    with pytest.raises(ValueError, match=r"line 1.*terminal.*final"):
        load_episodes(path)


def test_load_episodes_rejects_episode_id_reused_after_another_episode(tmp_path):
    rows = [
        complete_episode("episode-a")[0],
        complete_episode("episode-b")[0],
        complete_episode("episode-a")[1],
    ]
    path = write_rows(tmp_path, rows)

    with pytest.raises(ValueError, match=r"line 3.*noncontiguous"):
        load_episodes(path)


def test_load_episodes_rejects_incomplete_episode_but_quarantine_returns_it(tmp_path):
    incomplete = [row("partial", 0)]
    complete = complete_episode("complete")
    path = write_rows(tmp_path, incomplete + complete)

    with pytest.raises(ValueError, match=r"line 1.*incomplete"):
        load_episodes(path)

    episodes, quarantined = load_episodes_with_quarantine(path)

    assert episodes == [complete]
    assert quarantined == [
        {
            "episode_id": "partial",
            "reason": "incomplete episode: final row is nonterminal",
            "path": str(path),
            "line": 1,
        }
    ]


def test_load_episodes_rejects_inconsistent_present_case_ids(tmp_path):
    rows = complete_episode()
    rows[0]["observation"]["case_id"] = "case-a"
    rows[0]["next_observation"]["case_id"] = "case-b"
    rows[1]["observation"]["case_id"] = "case-b"
    path = write_rows(tmp_path, rows)

    with pytest.raises(ValueError, match=r"line 2.*case_id"):
        load_episodes(path)


def test_load_episodes_rejects_nonterminal_final_turn_as_an_integrity_error(tmp_path):
    rows = [row("episode-a", 3, done=False, observation={"turn": 3})]
    path = write_rows(tmp_path, rows)

    with pytest.raises(ValueError, match=r"line 1.*turn 3.*terminal"):
        load_episodes_with_quarantine(path)


def test_load_episodes_rejects_invalid_json_with_path_and_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{path}.*line 1.*invalid JSON"):
        load_episodes(path)


def test_load_episodes_rejects_unhashable_action_and_invalid_case_id(tmp_path):
    rows = complete_episode()
    rows[0]["action"] = []
    path = write_rows(tmp_path, rows)

    with pytest.raises(ValueError, match=r"line 1.*action"):
        load_episodes(path)

    rows = complete_episode()
    rows[0]["observation"]["case_id"] = None
    path = write_rows(tmp_path, rows)

    with pytest.raises(ValueError, match=r"line 1.*case_id"):
        load_episodes(path)
