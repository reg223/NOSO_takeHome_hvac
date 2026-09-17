import copy

import pytest

from submission.features import ACTIONS
from submission.rules import eligible_actions, improved_rule, service_override


def observation(**overrides):
    """A neutral, safe decision state with literal values for each branch."""
    result = {
        "turn": 0,
        "topic": "weather_tip",
        "customer_signal": "none",
        "membership_status": "none",
        "estimate_value_bucket": "none",
        "ignored_count": 0,
        "user_patience": 2,
        "service_risk_score": 0.0,
        "days_since_last_inbound": 99,
        "days_since_last_outbound": 99,
        "previous_actions": [],
    }
    result.update(overrides)
    return result


@pytest.mark.parametrize(
    ("name", "state", "expected"),
    [
        ("generic threads wait", {}, "WAIT"),
        ("service issues check in", {"topic": "service_issue"}, "CHECK_IN"),
        (
            "estimate without a signal gets a nudge",
            {"topic": "open_estimate", "estimate_value_bucket": "medium"},
            "ESTIMATE_NUDGE",
        ),
        (
            "estimate objection asks a question",
            {"topic": "competitor_quote", "customer_signal": "competitor"},
            "ASK_OBJECTION",
        ),
        (
            "positive estimate signal offers scheduling",
            {"topic": "repair_vs_replace", "customer_signal": "positive"},
            "OFFER_SCHEDULING",
        ),
        (
            "expiring member receives a membership touch",
            {"topic": "weather_tip", "membership_status": "expiring"},
            "MEMBERSHIP_TOUCH",
        ),
        ("unanswered touches park", {"ignored_count": 2}, "PARK_THREAD"),
        ("unhappy customers escalate", {"customer_signal": "unhappy"}, "ESCALATE_TO_HUMAN"),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_rule_returns_each_supported_action_for_its_meaningful_context(name, state, expected):
    """Catches a missing fallback branch or an action chosen for the wrong context."""
    assert improved_rule(observation(**state), []) == expected


def test_risk_override_strictly_exceeds_threshold_and_outranks_parking():
    """Catches changing the safeguard boundary or checking outreach fatigue first."""
    at_boundary = observation(service_risk_score=0.78, ignored_count=2)
    above_boundary = observation(service_risk_score=0.78001, ignored_count=2)

    assert service_override(at_boundary) is None
    assert improved_rule(at_boundary, []) == "PARK_THREAD"
    assert service_override(above_boundary) == "ESCALATE_TO_HUMAN"
    assert improved_rule(above_boundary, []) == "ESCALATE_TO_HUMAN"


def test_zero_patience_does_not_park_a_relevant_estimate():
    """Catches treating the predictive patience feature as an automatic terminal rule."""
    state = observation(
        topic="open_estimate",
        estimate_value_bucket="high",
        customer_signal="price_objection",
        user_patience=0,
    )

    assert improved_rule(state, []) == "ASK_OBJECTION"


def test_recent_outbound_contact_waits_before_topic_follow_up():
    """Catches sending another touch when the outbound recency safeguard applies."""
    state = observation(
        topic="open_estimate",
        estimate_value_bucket="high",
        customer_signal="positive",
        days_since_last_outbound=0,
    )

    assert improved_rule(state, []) == "WAIT"


@pytest.mark.parametrize("topic", ["maintenance_checkin", "install_warranty", "plan_visit"])
def test_repeated_maintenance_touch_waits_when_an_inbound_reply_exists(topic):
    """Catches repeating a check-in when there has been no newer inbound signal."""
    state = observation(
        topic=topic,
        previous_actions=["CHECK_IN"],
        days_since_last_inbound=1,
    )

    assert improved_rule(state, []) == "WAIT"


def test_new_inbound_activity_permits_a_maintenance_check_in():
    """Catches suppressing care follow-up after the customer has newly engaged."""
    state = observation(
        topic="maintenance_checkin",
        previous_actions=["CHECK_IN"],
        days_since_last_inbound=0,
    )

    assert improved_rule(state, []) == "CHECK_IN"


def test_repeated_objection_question_waits_without_new_inbound_activity():
    """Catches repeating an objection prompt without a new customer signal."""
    state = observation(
        topic="open_estimate",
        estimate_value_bucket="medium",
        customer_signal="price_objection",
        previous_actions=["ASK_OBJECTION"],
        days_since_last_inbound=1,
    )

    assert improved_rule(state, []) == "WAIT"


def test_new_inbound_activity_permits_an_objection_question():
    """Catches suppressing an objection response after the customer has replied."""
    state = observation(
        topic="open_estimate",
        estimate_value_bucket="medium",
        customer_signal="price_objection",
        previous_actions=["ASK_OBJECTION"],
        days_since_last_inbound=0,
    )

    assert improved_rule(state, []) == "ASK_OBJECTION"


@pytest.mark.parametrize(
    ("topic", "last_action", "signal"),
    [
        ("maintenance_checkin", "CHECK_IN", "none"),
        ("open_estimate", "ASK_OBJECTION", "price_objection"),
    ],
)
def test_missing_inbound_recency_waits_before_repeating_an_outreach_action(topic, last_action, signal):
    """Catches sparse observations making the rule repeat an unsolicited touch."""
    state = observation(
        topic=topic,
        estimate_value_bucket="medium",
        customer_signal=signal,
        previous_actions=[last_action],
        days_since_last_inbound=None,
    )

    assert improved_rule(state, []) == "WAIT"


def test_second_estimate_contact_asks_when_a_nudge_had_no_signal():
    """Catches leaving an unresponsive estimate on the same nudge loop."""
    state = observation(
        topic="open_estimate",
        estimate_value_bucket="medium",
        previous_actions=["ESTIMATE_NUDGE"],
    )

    assert improved_rule(state, []) == "ASK_OBJECTION"


def test_explicit_last_action_is_used_when_visible_action_history_is_empty():
    """Catches ignoring the observation's last-action fallback on an empty history."""
    state = observation(
        topic="open_estimate",
        estimate_value_bucket="medium",
        previous_actions=[],
        last_action="ESTIMATE_NUDGE",
    )

    assert improved_rule(state, []) == "ASK_OBJECTION"


def test_aging_equipment_only_becomes_an_estimate_context_with_an_estimate_flag():
    """Catches treating generic ageing advice as a sales topic without evidence."""
    generic = observation(topic="aging_equipment")
    flagged = observation(topic="aging_equipment", estimate_value_bucket="low")

    assert improved_rule(generic, []) == "WAIT"
    assert improved_rule(flagged, []) == "ESTIMATE_NUDGE"


@pytest.mark.parametrize("topic", ["service_issue", "maintenance_checkin"])
def test_service_and_maintenance_priority_excludes_membership_touch(topic):
    """Catches membership status bypassing care-oriented topic precedence."""
    state = observation(topic=topic, membership_status="lapsed")

    assert improved_rule(state, []) == "CHECK_IN"


def test_history_and_identifier_do_not_change_the_rule_decision():
    """Catches accidental use of checker-supplied cross-case history or identifiers."""
    state = observation(topic="open_estimate", estimate_value_bucket="medium", case_id="case-a")
    changed_identifier = {**state, "case_id": "case-b"}
    unrelated_history = [{"reward": 9999, "action": "ESCALATE_TO_HUMAN", "case_id": "other"}]

    assert improved_rule(state, []) == "ESTIMATE_NUDGE"
    assert improved_rule(changed_identifier, unrelated_history) == "ESTIMATE_NUDGE"


def test_outcome_metadata_does_not_change_the_rule_decision():
    """Catches leaked logged outcomes or next states changing an online decision."""
    state = observation(topic="open_estimate", estimate_value_bucket="medium")
    poisoned = {
        **state,
        "episode_id": "other-episode",
        "step": 3,
        "action": "PARK_THREAD",
        "action_prob": 1.0,
        "reward": 1000,
        "done": True,
        "info": {"outcome": "sold_estimate"},
        "next_observation": {"customer_signal": "unhappy"},
    }

    assert improved_rule(state, []) == "ESTIMATE_NUDGE"
    assert improved_rule(poisoned, []) == "ESTIMATE_NUDGE"


def test_missing_values_default_to_safe_wait():
    """Catches malformed or sparse input triggering outreach or an exception."""
    assert improved_rule({}, []) == "WAIT"


def test_eligibility_keeps_safeguards_and_the_contextual_fallback():
    """Catches restrictive filtering that drops a safe action or the rule's choice."""
    estimate = observation(topic="open_estimate", estimate_value_bucket="medium")
    service = observation(topic="service_issue")

    assert eligible_actions(estimate) == [
        "WAIT",
        "ESTIMATE_NUDGE",
        "ASK_OBJECTION",
        "OFFER_SCHEDULING",
        "PARK_THREAD",
        "ESCALATE_TO_HUMAN",
    ]
    assert eligible_actions(service) == ["WAIT", "CHECK_IN", "PARK_THREAD", "ESCALATE_TO_HUMAN"]


def test_unfiltered_eligibility_returns_the_canonical_action_order():
    """Catches the no-filter ablation silently retaining any context filter."""
    assert eligible_actions(observation(topic="service_issue"), filter_actions=False) == list(ACTIONS)


@pytest.mark.parametrize('topic', ['service_issue', 'maintenance_checkin', 'install_warranty', 'plan_visit'])
def test_care_eligibility_excludes_membership_even_when_lapsed(topic):
    assert 'MEMBERSHIP_TOUCH' not in eligible_actions(observation(topic=topic, membership_status='lapsed'))


def test_membership_topic_is_eligible_but_fallback_requires_expiring_or_lapsed_status():
    state = observation(topic='membership_renewal', membership_status='active')
    assert 'MEMBERSHIP_TOUCH' in eligible_actions(state)
    assert improved_rule(state, []) == 'WAIT'
    assert improved_rule(dict(state, membership_status='lapsed'), []) == 'MEMBERSHIP_TOUCH'


def test_rule_and_eligibility_preserve_the_observation():
    state = observation(topic='open_estimate', previous_actions=['ESTIMATE_NUDGE'])
    before = copy.deepcopy(state)
    assert improved_rule(state, []) == 'ASK_OBJECTION'
    assert 'ASK_OBJECTION' in eligible_actions(state)
    assert state == before


def test_parking_outranks_recent_outbound_and_unhappy_outranks_both():
    state = observation(ignored_count=2, days_since_last_outbound=0)
    assert improved_rule(state, []) == 'PARK_THREAD'
    assert improved_rule(dict(state, customer_signal='unhappy'), []) == 'ESCALATE_TO_HUMAN'


def test_missing_signal_does_not_trigger_the_no_signal_nudge_follow_up():
    state = observation(topic='open_estimate', previous_actions=['ESTIMATE_NUDGE'])
    state.pop('customer_signal')
    assert improved_rule(state, []) == 'ESTIMATE_NUDGE'
