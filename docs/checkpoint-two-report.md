# Checkpoint 2: evaluator and development baselines

Checkpoint 2 implements and tests the evaluator needed before optimizing a
learned policy. The development results **do not establish an improvement of the
contextual fallback over the starter rule**.

## What was achieved

- Implemented complete-trajectory PDIS, sequential doubly robust (DR), and direct
  model estimates, with zero continuation after termination and no restarting
  importance weights after an action mismatch.
- Fitted a separate fixed-policy evaluator for each of four unchanged baselines
  on 3,600 evaluator-fit episodes; scored the separate 3,600 development episodes.
  Validation logs were not accessed.
- Added 1,000 paired episode bootstrap resamples, weight-cap sensitivities
  (10/50/100), four initial-state slices, support checks, ESS, costs, touches, and
  all eight observed terminal-outcome categories.
- Passed 109 tests, including hand-calculated two-action/two-step arithmetic,
  fixed-policy target construction, leakage resistance, role separation, and the
  end-to-end runner. Independent code review found no substantive issues.
- Reproduced the complete result artifact byte-for-byte from an unrelated
  working directory, including refitting all four evaluators.

## Results and implications

Undiscounted reward per episode; intervals are conditional 95% bootstrap intervals.

| Policy | Direct model | PDIS | DR [95% interval] |
|---|---:|---:|---:|
| Always check in | -19.47 | -12.93 | -22.84 [-25.81, -20.15] |
| Always nudge estimate | -34.11 | -50.47 | -51.73 [-68.96, -41.29] |
| Starter simple rule | -6.77 | 3.61 | 3.07 [-6.30, 14.25] |
| Contextual fallback | -5.79 | 13.06 | 9.20 [-3.64, 34.74] |

The historical logging policy averaged **-13.60** on these episodes. That is a
separate factual reference, not a paired experiment against these candidates.

The contextual-minus-starter paired DR difference is **+6.14 [-13.02, 33.94]**.
It changes to +1.94, +1.83, and -0.20 under caps of 10, 50, and 100 respectively;
each interval includes zero. The direct difference is +0.98, but the direct and
DR absolute estimates disagree substantially. A favorable point estimate is
insufficient evidence to promote the contextual rule.

| Initial-state slice | Episodes | Paired DR reward difference [95% interval] |
|---|---:|---:|
| Service/unhappy/high risk | 282 | +1.05 [-2.34, 4.58] |
| Open estimate with objection | 772 | +32.72 [-50.14, 148.27] |
| Cold or ignored | 1,906 | -1.11 [-9.04, 6.77] |
| Membership relevant | 1,124 | -11.61 [-43.82, 9.76] |

Every slice remains inconclusive. For the contextual rule, final-turn ESS is
**1.85**, from only **two matching trajectories**; one contributes 64% of the
turn's total weight. Also, 6.7% of its decisions on logged states have no matching
evaluator-fit action in their coarse support cell. Those states require model
extrapolation, and positive coarse counts elsewhere do not prove overlap.

Secondary DR estimates suggest a tradeoff against the starter: **0.42 fewer
outreach touches**, **7.34 percentage points fewer opt-outs**, but **6.58 points
fewer bookings** and **12.89 points fewer membership conversions** per episode.
Estimated cost falls by 0.17 cents, with an interval crossing zero. These are
uncertain off-policy estimates, not observed effects or reliable forecasts.
Touches count the five outreach actions; costs are reported separately without
subtracting them from reward again.

## Handoff

The evaluator is ready for Task 4's immediate-reward experiment with support and
advantage gates. Baselines, safeguard settings, and validation remain unchanged.
No policy is promoted and these results authorize no rollout. Bootstrap intervals
exclude evaluator-fitting uncertainty; hidden confounding, propensity error,
model bias, and changed future states remain limitations.

Reproduce with `.venv/bin/python scripts/evaluate_development.py`; run tests with
`.venv/bin/python -m pytest tests -q`. Exact estimates, paired contrasts,
sensitivities, role IDs, source/data hashes, and settings are in
[`artifacts/dev_baselines.json`](../artifacts/dev_baselines.json). Definitions and
assumptions are in [`checkpoint-two-protocol.md`](checkpoint-two-protocol.md).
