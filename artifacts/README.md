# Progress report: data audit, feature contract, and rule benchmark

## Task 1: data audit and development split

Generated from the supplied JSONL files using Python 3.9.23 and NumPy 2.0.2;
tested with pytest 8.4.2. From the project root:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest tests -q
.venv/bin/python -m submission.audit --train data/train_logs.jsonl --valid data/valid_logs.jsonl --output artifacts/audit.json --splits artifacts/splits.json
```

`audit.json` records SHA-256 dataset hashes, usable episode IDs, quarantine,
coverage counts (individual and joint), propensity/reward/return quantiles,
logged outcomes, total cost, terminal-state conventions, and recency cross-tabs.
Rewards are unchanged; cost is reported separately, not subtracted again.
These summaries describe the historical logging policy, not a candidate policy.

`splits.json` contains training episode IDs only. Sorted IDs are shuffled with
`numpy.random.default_rng(20260621)` (PCG64): the first 16,800 fit policies,
next 3,600 fit evaluators, and last 3,600 score development candidates. If a
future dataset repeats case IDs, whole case groups are assigned in first
shuffled occurrence order to cumulative 70%/85% boundaries; groups can overshoot
boundaries, and actual counts are recorded. Missing case IDs cannot establish
case independence. The supplied data has complete case IDs and no repeated cases.

## Findings

- Training: 24,000 complete episodes / 74,733 rows.
- Validation: 5,000 complete episodes / 15,565 rows.
- No incomplete episodes, unexpected integrity failures, or train/validation
  episode/case overlap; all supplied episodes are usable.
- All terminal next observations are null in these files. The loader also accepts
  terminal dictionaries and always uses `done` to determine continuation.
- All 1,277 training parks and 1,828 training escalations terminate. Validation
  has 297 parks and 350 escalations, all terminal.
- Training cost totals 38,431.0861 cents; validation totals 7,525.5064 cents.
- Outbound recency is 99 at every initial observation. In nonterminal transitions
  from 99 it stays 99 after WAIT and resets to 0 after outreach. Inbound recency
  also persists at 99 or resets to 0. This supports an absent-contact interpretation
  but does not prove it; retain the raw value and a separate `is_99` indicator
  when features are implemented in Task 2.

Strict `load_episodes` rejects incomplete episodes. The audit's explicit quarantine
loader excludes incomplete episodes from return summaries and all development
roles while recording their IDs and source lines. All other integrity failures
abort generation; they require documented treatment before training proceeds.

Validation structural checks and preliminary counts were already viewed during
planning. This task adds aggregate rewards, outcomes, costs, coverage and recency
inspection. No validation outcomes influenced split construction or model tuning.
No policy has been trained or evaluated by Task 1.

## Task 2: feature contract and rule benchmark

Completed `submission/features.py`, `submission/rules.py`, and their tests.
The detailed contract and topic choices are documented in
[`docs/task2-feature-and-rule-contract.md`](../docs/task2-feature-and-rule-contract.md).
The original supplied benchmarks remain unchanged.

Feature extraction uses an explicit decision-time allowlist. IDs, logged
actions, outcomes, rewards, propensities, terminal flags, and next states are
excluded. The visible previous-action list supplies last action, per-action
counts, and the consecutive repeat count; remaining decisions is `4-turn`.
Missing categories get an explicit level, unknown categories are ignored,
and numeric values use fit-set medians with stable missing indicators.
Recency 99 retains both its raw value and a separate indicator.

The rule prioritizes unhappy/high-risk escalation, ignored-outreach parking,
recent-outreach waiting, service and maintenance care, membership renewal,
then estimate follow-up. Zero patience alone does not cause parking. Repeated
check-ins or objection questions wait without an observed new inbound signal;
missing inbound recency also conservatively waits after a repeat.

The audited `plan_visit` topic joins `maintenance_checkin` and
`install_warranty` in the maintenance group. `competitor_quote` and
`repair_vs_replace` join `open_estimate`; a low/medium/high estimate bucket
also establishes estimate relevance. These are initial engineering choices
for later evaluation. A membership touch is eligible for membership-renewal
topics or expiring/lapsed status outside care priority, while the fallback
requires expiring/lapsed status.

The learned-candidate action filter always includes wait, park, escalation,
and the fallback. Its `filter_actions=False` switch preserves the planned
no-filter development experiment. The service override remains a separate
mandatory step for future learned selectors. Its risk threshold of 0.78 is
a product safeguard, not an estimated optimal threshold.

Verification results are recorded in [`task2_checks.json`](task2_checks.json):

- 45 Task 2 tests passed; 70 total tests passed.
- All 74,733 logged training observations returned legal actions; the filter
  excluded the fallback zero times.
- All 2,860 unhappy/high-risk observations escalated: zero service-override
  violations, including when parking or recent-contact conditions also applied.
- Arbitrary cross-case history changed zero decisions; observation mutation
  checks also found zero changes.
- The encoder fitted only the policy-fit role: 52,323 decision rows from
  16,800 episodes. It transformed 11,144 development-score rows from 3,600
  episodes into 105 dense features with no nonfinite values. Transforming
  held-out observations did not change fitted medians.
- Serialization tests reproduced the same feature values after loading.

For inspection only, the rule's actions on historical training states were:

| Action | Decision rows |
| --- | ---: |
| WAIT | 36,469 |
| CHECK_IN | 9,069 |
| ESTIMATE_NUDGE | 4,226 |
| ASK_OBJECTION | 6,755 |
| OFFER_SCHEDULING | 571 |
| MEMBERSHIP_TOUCH | 3,444 |
| PARK_THREAD | 11,339 |
| ESCALATE_TO_HUMAN | 2,860 |

These counts are static replay diagnostics on states visited by the historical
policy. They do not estimate the new rule's outcomes or future action mix:
changing actions changes later states, and a new terminal action would end
an episode earlier. No candidate reward model or policy-value estimator was
fitted, and no improvement over the baselines is claimed. Task 3's evaluator
is the next required step before making a performance comparison.
