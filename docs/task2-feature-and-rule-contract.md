# Task 2: feature and rule contract

`submission.features.featurize(observation)` is the shared, deterministic
decision-time representation. `FeatureEncoder` fits dense one-hot categories
and numeric medians on observations explicitly supplied to `fit`; future
training code must pass only its policy-fit subset. It accepts raw observation
dictionaries at both fit and transform time. Persist the fitted encoder with
the model rather than fitting a fresh encoder at inference.

The allowlist includes the README scalar fields and the observed inbound and
outbound recency and last-action fields. Identifiers, current logged action,
row step, rewards, propensities, terminal flags, info, hidden fields, and next
states are excluded. Previous actions contribute one count per legal action,
the trailing repeat count, and last action. A nonempty `previous_actions` list
takes precedence over a conflicting explicit `last_action`; otherwise the
explicit field is used. Absent action history yields missing counts, while
an explicitly empty list yields zero counts. Remaining decisions is `4-turn`.

Missing categorical values use `__missing__`. Each categorical field reserves
that level, and unseen nonmissing values become an all-zero one-hot block.
Every numeric field has a missing indicator, including fields with no missing
fit examples. Missing numeric values use the fit-set median; if the entire
fit column is missing, zero is used with the missing indicator retained.
Recency 99 remains numeric and also receives a separate `*_is_99` indicator.
The interpretation of 99 as absent contact is plausible but unproven.

## Rule priorities and topic mapping

`improved_rule(observation, history)` ignores caller-provided history. It uses
the observation contract shared with the encoder. The escalation threshold
of risk greater than 0.78 (or unhappy signal) is a product safeguard inherited
from the supplied baseline, not an empirically optimal threshold.

Fallback priority is escalation, parking after at least two ignored touches,
waiting after outbound recency zero, service check-in, maintenance check-in
or wait, expiring/lapsed membership touch, estimate logic, then wait.
For repeated maintenance check-ins, no observed new inbound response means
wait. Missing inbound recency also conservatively leads to wait after a repeat.
Zero patience alone does not force parking.

The following mapping uses exact topic names recorded by Task 1. It is an
explicit initial engineering choice to evaluate on development data, not an
estimate of action effectiveness:

| Context | Exact topics and additional conditions |
| --- | --- |
| Service | `service_issue` |
| Maintenance | `maintenance_checkin`, `install_warranty`, `plan_visit` |
| Estimate | `open_estimate`, `competitor_quote`, `repair_vs_replace`; also any observation with estimate bucket `low`, `medium`, or `high` |
| Membership relevance | `membership_renewal`, or status `expiring`/`lapsed`, outside service/maintenance priority |
| Generic | `aging_equipment`, `weather_tip`, or unknown topics unless another visible condition establishes relevance |

Membership fallback requires expiring/lapsed status; membership-renewal topic
with another status can still make membership touch eligible for a learned
candidate. Estimate objections trigger discovery unless the previous action
already asked and inbound recency is nonzero or missing. Otherwise positive
signal triggers scheduling, no signal after an estimate nudge triggers an
objection question, and other estimate contexts trigger a nudge.

`eligible_actions` returns unique actions in the canonical eight-action order.
It always retains wait, park, and escalation, adds context-specific actions,
and includes the fallback. `filter_actions=False` supplies the full action set
for the planned development ablation; it does not disable the service override.
Future learned selectors must apply that override before scoring or gating.
No no-filter performance comparison has been run; that requires Task 3's
evaluator and the subsequent learned candidates.

The supplied `simple_rule` and `always_check_in` benchmarks remain unchanged.
Task 2 establishes behavior and feature consistency; it does not establish
that the new rule improves outcomes.

## Verification

```sh
.venv/bin/python -m pytest tests/test_features.py tests/test_rules.py -q
.venv/bin/python -m pytest tests -q
```

Tests cover all eight actions, precedence and threshold boundaries, repeated
outreach, zero patience, arbitrary cross-case history, leakage, numeric
imputation, unknown categories, and serialized feature consistency.
