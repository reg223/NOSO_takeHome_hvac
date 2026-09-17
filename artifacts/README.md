# Task 1: data audit and development split

Generated from the supplied JSONL files using Python 3.9.23 and NumPy 2.0.2;
tested with pytest 8.4.2. From the project root:

```sh
.venv/bin/python -m pip install numpy==2.0.2 pytest==8.4.2
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
