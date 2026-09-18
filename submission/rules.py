"""Conservative, explainable benchmark actions for HVAC follow-up.

This benchmark applies safeguards before commercial follow-up.  It treats
``maintenance_checkin``, ``install_warranty``, and ``plan_visit`` as
maintenance topics; ``service_issue`` as service; and ``open_estimate``,
``competitor_quote``, and ``repair_vs_replace`` as estimate topics.  A low,
medium, or high estimate-value bucket also establishes an estimate context.
``aging_equipment`` and ``weather_tip`` remain generic unless that estimate
flag is present.  A membership-renewal topic is eligible for membership
actions, but the fallback sends a membership touch only for an expiring or
lapsed membership.  Service and maintenance take priority over membership.

The service-risk cutoff is a product safeguard, not an empirically optimal
threshold.  Callers may use ``eligible_actions(..., filter_actions=False)``
to run an intentional no-filter ablation; it returns the canonical action
order unchanged.
"""
from __future__ import annotations

import math

from typing import Any, Dict, List

from submission.features import ACTIONS, featurize


DEFAULT_SERVICE_RISK_THRESHOLD = 0.78


MAINTENANCE_TOPICS = frozenset({
    "maintenance_checkin", "install_warranty", "plan_visit",
})
SERVICE_TOPICS = frozenset({"service_issue"})
ESTIMATE_TOPICS = frozenset({
    "open_estimate", "competitor_quote", "repair_vs_replace",
})
ESTIMATE_BUCKETS = frozenset({"low", "medium", "high"})
OBJECTION_SIGNALS = frozenset({
    "price_objection", "competitor", "ask_spouse", "timing_delay",
})
MEMBERSHIP_STATUSES = frozenset({"expiring", "lapsed"})


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _at_least(value, threshold):
    number = _number(value)
    return number is not None and number >= threshold


def _is_zero(value):
    number = _number(value)
    return number == 0


def _features(observation):
    """Normalize all rule inputs through the shared decision-time contract."""
    return featurize(observation if isinstance(observation, dict) else {})


def _is_estimate_context(features):
    return (features["topic"] in ESTIMATE_TOPICS
            or features["estimate_value_bucket"] in ESTIMATE_BUCKETS)


def _is_membership_relevant(features):
    return (features["topic"] == "membership_renewal"
            or features["membership_status"] in MEMBERSHIP_STATUSES)


def service_override(observation, history=None, *,
                     service_risk_threshold=DEFAULT_SERVICE_RISK_THRESHOLD):
    """Return the terminal escalation safeguard, or ``None`` when inapplicable."""
    del history
    if (isinstance(service_risk_threshold, bool)
            or not isinstance(service_risk_threshold, (int, float))
            or not math.isfinite(service_risk_threshold)
            or not 0 <= service_risk_threshold <= 1):
        raise ValueError('service_risk_threshold must be a finite number in [0, 1]')
    features = _features(observation)
    risk = _number(features["service_risk_score"])
    if features["customer_signal"] == "unhappy" or (risk is not None and risk > service_risk_threshold):
        return "ESCALATE_TO_HUMAN"
    return None


def fallback_action(observation, history=None, *,
                    service_risk_threshold=DEFAULT_SERVICE_RISK_THRESHOLD):
    """Choose the next action from visible observation fields only.

    ``history`` is intentionally ignored.  The supplied checker can combine
    prior records from unrelated cases, while ``previous_actions`` is the
    visible, episode-specific source of action history.
    """
    del history
    features = _features(observation)

    override = service_override(observation, service_risk_threshold=service_risk_threshold)
    if override is not None:
        return override
    if _at_least(features["ignored_count"], 2):
        return "PARK_THREAD"
    if _is_zero(features["days_since_last_outbound"]):
        return "WAIT"

    topic = features["topic"]
    last_action = features["last_action"]
    inbound = features["days_since_last_inbound"]
    if topic in SERVICE_TOPICS:
        return "CHECK_IN"
    if topic in MAINTENANCE_TOPICS:
        if last_action == "CHECK_IN" and not _is_zero(inbound):
            return "WAIT"
        return "CHECK_IN"

    if (features["membership_status"] in MEMBERSHIP_STATUSES
            and _is_membership_relevant(features)):
        return "MEMBERSHIP_TOUCH"

    if _is_estimate_context(features):
        signal = features["customer_signal"]
        if signal in OBJECTION_SIGNALS:
            if last_action == "ASK_OBJECTION" and not _is_zero(inbound):
                return "WAIT"
            return "ASK_OBJECTION"
        if signal == "positive":
            return "OFFER_SCHEDULING"
        if signal == "none" and last_action == "ESTIMATE_NUDGE":
            return "ASK_OBJECTION"
        return "ESTIMATE_NUDGE"

    return "WAIT"


def eligible_actions(observation, history=None, *, filter_actions=True,
                     service_risk_threshold=DEFAULT_SERVICE_RISK_THRESHOLD):
    """Return context-compatible actions while retaining terminal safeguards.

    Filtering is for learned-policy candidates.  The benchmark action is also
    included so the filter cannot make the fallback unavailable.
    """
    if not filter_actions:
        return list(ACTIONS)

    features = _features(observation)
    eligible = {"WAIT", "PARK_THREAD", "ESCALATE_TO_HUMAN"}
    if features["topic"] in SERVICE_TOPICS | MAINTENANCE_TOPICS:
        eligible.add("CHECK_IN")
    if (_is_membership_relevant(features)
            and features["topic"] not in SERVICE_TOPICS | MAINTENANCE_TOPICS):
        eligible.add("MEMBERSHIP_TOUCH")
    if _is_estimate_context(features):
        eligible.update(("ESTIMATE_NUDGE", "ASK_OBJECTION", "OFFER_SCHEDULING"))

    fallback = fallback_action(observation, history, service_risk_threshold=service_risk_threshold)
    eligible.add(fallback)
    return [action for action in ACTIONS if action in eligible]


# Backward-compatible name for existing benchmark callers.
improved_rule = fallback_action


def simple_rule_action(observation: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    """Intentionally decent but beatable hand-coded policy."""

    risk = observation["service_risk_score"]
    signal = observation["customer_signal"]
    track = observation["track_hint"]
    membership = observation["membership_status"]
    estimate = observation["estimate_value_bucket"]
    ignored = observation["ignored_count"]
    touches = observation["touch_count"]
    patience = observation["user_patience"]
    urgency = observation["urgency_score"]
    relationship = observation["relationship_score"]

    if signal == "unhappy" or risk > 0.78:
        return "ESCALATE_TO_HUMAN"
    if ignored >= 2:
        return "PARK_THREAD"
    if touches >= 4:
        return "WAIT"
    if track == "relationship":
        if membership in {"expiring", "lapsed"}:
            return "MEMBERSHIP_TOUCH"
        if relationship > 0.55:
            return "CHECK_IN"
    if estimate != "none":
        if signal in {"price_objection", "competitor", "ask_spouse", "timing_delay"} and patience > 0:
            return "ASK_OBJECTION"
        if signal == "positive" or urgency > 0.70:
            return "OFFER_SCHEDULING"
        return "ESTIMATE_NUDGE"
    if membership in {"active", "expiring", "lapsed"}:
        return "MEMBERSHIP_TOUCH"
    return "WAIT"



def always_check_in(observation, history=None):
    """Unconditional comparison reference."""
    return "CHECK_IN"


def always_estimate_nudge(observation, history=None):
    """Unconditional comparison reference."""
    return "ESTIMATE_NUDGE"
