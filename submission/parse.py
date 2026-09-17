
"""Streaming validation and episode reconstruction for logged JSONL data."""

import json
import math


LEGAL_ACTIONS = frozenset(
    {
        "WAIT",
        "CHECK_IN",
        "ESTIMATE_NUDGE",
        "ASK_OBJECTION",
        "OFFER_SCHEDULING",
        "MEMBERSHIP_TOUCH",
        "PARK_THREAD",
        "ESCALATE_TO_HUMAN",
    }
)
TERMINATING_ACTIONS = frozenset({"PARK_THREAD", "ESCALATE_TO_HUMAN"})
REQUIRED_ROW_KEYS = frozenset(
    {
        "episode_id",
        "step",
        "observation",
        "action",
        "action_prob",
        "reward",
        "done",
        "next_observation",
        "info",
    }
)


def load_episodes(path):
    """Return complete, validated episodes from a JSONL log.

    Unlike :func:`load_episodes_with_quarantine`, this strict entry point
    rejects an episode whose final logged row is nonterminal.
    """
    episodes, quarantined = load_episodes_with_quarantine(path)
    if quarantined:
        item = quarantined[0]
        raise ValueError(
            f"{item['path']}: line {item['line']}: {item['reason']}"
        )
    return episodes


def load_episodes_with_quarantine(path):
    """Return ``(complete_episodes, quarantined_incomplete_episodes)``.

    A quarantine is reserved for a syntactically valid episode that simply
    ends with ``done == False``. All other integrity failures are raised
    immediately with the source path and line number.
    """
    source_path = str(path)
    episodes = []
    quarantined = []
    closed_episode_ids = set()
    current_rows = []
    current_id = None
    current_case_id = _MISSING

    def finish_current():
        nonlocal current_rows, current_id, current_case_id
        if not current_rows:
            return
        final_row, final_line = current_rows[-1]
        if final_row["done"]:
            episodes.append([row for row, _ in current_rows])
        else:
            quarantined.append(
                {
                    "episode_id": current_id,
                    "reason": "incomplete episode: final row is nonterminal",
                    "path": source_path,
                    "line": final_line,
                }
            )
        closed_episode_ids.add(current_id)
        current_rows = []
        current_id = None
        current_case_id = _MISSING

    with open(path, "r", encoding="utf-8") as stream:
        for line_number, text in enumerate(stream, start=1):
            row = _decode_row(text, source_path, line_number)
            _validate_row(row, source_path, line_number)
            episode_id = row["episode_id"]

            if not current_rows:
                if episode_id in closed_episode_ids:
                    _fail(source_path, line_number, "episode_id is noncontiguous/reused")
                current_id = episode_id
                current_case_id = _case_id(row)
                current_rows.append((row, line_number))
                _validate_first_step(row, source_path, line_number)
                continue

            if episode_id != current_id:
                finish_current()
                if episode_id in closed_episode_ids:
                    _fail(source_path, line_number, "episode_id is noncontiguous/reused")
                current_id = episode_id
                current_case_id = _case_id(row)
                current_rows.append((row, line_number))
                _validate_first_step(row, source_path, line_number)
                continue

            previous, previous_line = current_rows[-1]
            if previous["done"]:
                _fail(source_path, previous_line, "terminal row must be final in its episode")
            _validate_next_step(row, len(current_rows), source_path, line_number)
            current_case_id = _validate_case_id(
                current_case_id, row, source_path, line_number
            )
            if previous["next_observation"] != row["observation"]:
                _fail(
                    source_path,
                    previous_line,
                    "next_observation must equal the next row observation",
                )
            current_rows.append((row, line_number))

    finish_current()
    return episodes, quarantined


def parse_log(file):
    """Backward-compatible name for strict episode loading."""
    return load_episodes(file)


class _Missing:
    pass


_MISSING = _Missing()


def _decode_row(text, path, line_number):
    if not text.strip():
        _fail(path, line_number, "blank line is not valid JSONL")
    try:
        row = json.loads(text)
    except json.JSONDecodeError as error:
        _fail(path, line_number, f"invalid JSON: {error.msg}")
    if not isinstance(row, dict):
        _fail(path, line_number, "JSONL row must be an object")
    return row


def _validate_row(row, path, line_number):
    missing = sorted(REQUIRED_ROW_KEYS.difference(row))
    if missing:
        _fail(path, line_number, f"missing required key: {missing[0]}")

    episode_id = row["episode_id"]
    if not isinstance(episode_id, str) or not episode_id:
        _fail(path, line_number, "episode_id must be a non-empty string")
    _validate_turn(row["step"], "step", path, line_number)

    observation = row["observation"]
    if not isinstance(observation, dict):
        _fail(path, line_number, "observation must be an object")
    if "turn" not in observation:
        _fail(path, line_number, "observation is missing required key: turn")
    _validate_turn(observation["turn"], "observation.turn", path, line_number)
    if observation["turn"] != row["step"]:
        _fail(path, line_number, "observation.turn must equal step")
    if "case_id" in observation and (
        not isinstance(observation["case_id"], str) or not observation["case_id"]
    ):
        _fail(path, line_number, "case_id must be a non-empty string when present")

    action = row["action"]
    if not isinstance(action, str) or action not in LEGAL_ACTIONS:
        _fail(path, line_number, f"invalid action: {action!r}")
    if not _is_finite_number(row["action_prob"]) or not 0 < row["action_prob"] <= 1:
        _fail(path, line_number, "action_prob must be finite and in (0, 1]")
    if not _is_finite_number(row["reward"]):
        _fail(path, line_number, "reward must be finite")
    if not isinstance(row["done"], bool):
        _fail(path, line_number, "done must be a boolean")
    if row["step"] == 3 and not row["done"]:
        _fail(path, line_number, "turn 3 must be terminal")
    if action in TERMINATING_ACTIONS and not row["done"]:
        _fail(path, line_number, f"{action} must terminate the episode")
    if row["next_observation"] is not None and not isinstance(
        row["next_observation"], dict
    ):
        _fail(path, line_number, "next_observation must be an object or null")
    if not isinstance(row["info"], dict):
        _fail(path, line_number, "info must be an object")


def _validate_first_step(row, path, line_number):
    if row["step"] != 0:
        _fail(path, line_number, "steps must be contiguous and start at 0")


def _validate_next_step(row, expected_step, path, line_number):
    if row["step"] != expected_step:
        _fail(path, line_number, "steps must be contiguous within an episode")


def _validate_turn(value, name, path, line_number):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 3:
        _fail(path, line_number, f"{name} must be an integer in 0..3")


def _validate_case_id(current_case_id, row, path, line_number):
    case_id = _case_id(row)
    if case_id is _MISSING:
        return current_case_id
    if current_case_id is _MISSING:
        return case_id
    if case_id != current_case_id:
        _fail(path, line_number, "case_id must be consistent within an episode")
    return current_case_id


def _case_id(row):
    return row["observation"].get("case_id", _MISSING)


def _is_finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _fail(path, line_number, message):
    raise ValueError(f"{path}: line {line_number}: {message}")
