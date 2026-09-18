# Task 2 feature and contextual-rule contract

The runtime schema is version 2 in `submission/features.py`. Training and inference
call the same deterministic `featurize(observation, history=None)` function. The
allowlist contains topic, equipment type, season, track hint, heat, estimate value
bucket, membership status, customer signal, last action, turn, day, weather extreme,
ignored count, touch count, patience, urgency, price sensitivity, service risk,
relationship, inbound/outbound recency, and visible previous actions. Identifiers,
logged actions outside the observation, rewards, propensities, terminal metadata,
next observations, and outcomes are excluded even if supplied as extra keys.

## History and derived features

`history` is accepted for interface compatibility but intentionally ignored. This
is a documented deviation from the plan's suggestion to derive state from supplied
history: the checker may combine unrelated cases. Using it would contaminate the
current decision. Observation `previous_actions` is the sole source for action
counts, trailing repeated-action count, consecutive waits, and explicit occurred
indicators for objection questions, estimate nudges, and scheduling offers. Missing
or malformed previous actions produce missing derived values; an explicit empty
list produces zeros. A nonempty list takes precedence for last action; otherwise
an explicit observation last action is retained. Unknown actions are not counted
as canonical actions and break trailing runs. Remaining decisions is `4 - turn`.
Neither inputs nor fitted encoder statistics are mutated during extraction.

## Missingness and recency

Absent/invalid categorical values use `__missing__`. Non-finite or nonnumeric
numeric values become `None`; boolean weather indicators remain numeric. The
encoder adds a missing indicator for every numeric column and uses medians fitted
only on the designated training role. A wholly missing fit column is imputed to
zero while preserving its missing flag. Missing categories have explicit one-hot
levels; unseen categories produce an all-zero block. Numeric recency `99` is kept
literally and receives an explicit `is_99` flag. This encodes an observed value,
without asserting what it means in the real world. A missing recency's sentinel
flag is zero and its numeric missing flag distinguishes it from an observed value.
Persist the fitted encoder and schema version with future model bundles; version 1
bundles must not silently consume version 2 feature dimensions.

## Rule precedence and configuration

`fallback_action(observation, history=None)` is also available as `improved_rule`
for existing callers. It escalates an unhappy signal or risk strictly greater than
`DEFAULT_SERVICE_RISK_THRESHOLD = 0.78`, then parks after at least two ignored
outreach attempts, then waits after same-day outbound contact. Zero patience alone
never causes parking. The existing ignored-count exhaustion rule is retained;
it is an explainable heuristic, not an evidence-backed optimal terminal decision.

Service topics receive a check-in. Maintenance topics also receive a check-in,
but repeat check-ins wait unless inbound recency is zero; missing inbound recency
also waits. Service and maintenance precede membership. Expiring/lapsed membership
receives a membership touch. Estimate context is established by an estimate topic
or low/medium/high estimate bucket. Explicit objections receive an objection
question, but unchanged repeated questions wait unless inbound recency is zero.
Positive signals offer scheduling. An unresponsive prior nudge permits objection
discovery without an explicit objection; other estimate cases receive a nudge.
Generic contexts wait. Aging equipment alone does not establish estimate context.
These choices preserve the existing contextual benchmark rather than adding
untested urgency/relationship thresholds from the starter.

`eligible_actions(observation, history=None)` retains wait and both terminal
safeguards plus contextual actions and the fallback, in canonical action order.
Estimate contexts retain nudge, objection discovery, and scheduling even in
ambiguous cases. Care contexts exclude membership. Membership-renewal topics
allow membership touches, while the fallback still requires expiring/lapsed status.
Eligibility does not itself select a safe action: future learned selectors must
apply `service_override` before learned selection. `filter_actions=False` is a
development ablation and returns all eight actions.

The keyword-only `service_risk_threshold` allows explicit development experiments
in service override, fallback, and eligibility; its finite range is `[0, 1]`.
Ordinary calls retain 0.78. It is an initial product safeguard, not an empirically
optimal threshold. Future fitting must persist and freeze this configuration
before validation; validation must never tune it. No experiments were run here.

## Preserved comparison references and verification

`simple_rule_action` copies the starter's rule without changing its branches or
required visible keys. `always_check_in` and `always_estimate_nudge` are independent
constant references. All three live in `submission/rules.py` with no runtime
import of starter files. The starter remains unchanged. These reference policies
are comparisons, not evidence of performance or authorization for production.

Feature and rule tests cover forbidden fields, arbitrary history contamination,
explicit history missingness, wait runs, occurred indicators, recency flags,
fit-only imputation, unknown categories, serialization, contextual precedence,
service boundaries and explicit alternatives, plausible eligibility, and parity
with the preserved starter across its branches. No candidate training, scoring,
or validation-log access is part of Task 2.
