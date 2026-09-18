# Checkpoint two: development evaluation protocol

This protocol is specified before generating the development baseline results.
It implements Task 3 of the current implementation plan. It does not select or
tune a policy and does not use validation logs.

## Data roles and fixed comparisons

- Load only `data/train_logs.jsonl`, verifying its SHA-256 against the frozen
  `artifacts/splits.json` manifest.
- Fit each fixed-policy evaluator, including its encoder, on the 3,600
  `evaluator_fit` episodes. Do not fit on the policy-fit or development-score roles.
- Score the same 3,600 `development_score` episodes for `always_check_in`,
  `always_estimate_nudge`, `simple_rule`, and `contextual_fallback`.
- Preserve the contextual safeguard threshold of 0.78. No threshold search or
  baseline policy changes are part of this checkpoint.

The policy encoder in a future learned policy must fit on policy-fit data.
The separate evaluator encoder fits on evaluator-fit data. Both use the same
visible-field feature contract; fitted statistics need not be shared.

## Estimators

Use undiscounted complete-episode reward (`gamma = 1`). Report logged historical
episode returns separately from counterfactual estimates. Supplied rewards already
include their reward-accounting costs; separately estimate cost in cents without
subtracting that cost from reward again.

For a deterministic candidate, let `w_t` be the product through turn `t` of
`1[candidate action == logged action] / logged propensity`. A first mismatch
makes all subsequent weights zero. PDIS is `sum_t w_t * reward_t`. The direct
estimate is the fitted initial-state value. Sequential DR is
`V(s_0) + sum_t w_t * (reward_t + V(s_{t+1}) - Q(s_t, a_t))`, with zero
continuation after every logged terminal row, including `max_steps`.

Fit one multi-output ExtraTrees regressor per turn, backwards, with 100 trees,
maximum depth 8, minimum leaf size 30, seed 20260621, and one worker for
reproducibility. The target is the observed metric plus the next-turn value under
the same frozen candidate. Never maximize over next actions. Append a fixed
canonical action one-hot encoding to the shared decision-time state features.
Outcome indicators, cost, and outreach counts are targets only, never state inputs.

Report unclipped estimates as primary. Sensitivity estimates cap each raw
cumulative importance weight at 10, 50, or 100. Accumulate the original ratios
before applying each cap; do not recursively accumulate clipped weights. These
are importance-weight caps, not propensity probabilities or propensity floors.
Clipping reduces variance but introduces bias.

## Uncertainty and support

Resample complete episodes 1,000 times with seed 20260621 and report percentile
95% intervals. Align policy scores by episode ID and use identical episode samples
for policy contrasts. Intervals are conditional on fitted evaluators; the
bootstrap does not refit models or remove model bias, hidden confounding, or
propensity misspecification.

At each turn report nonzero matching-prefix counts, maximum cumulative weight,
maximum normalized weight, and ESS `(sum(w)**2) / sum(w**2)`. Report zero ESS
explicitly. Also report evaluator-fit action counts and coarse local counts for
`(turn, topic, customer_signal, action)`. Missing local support is an extrapolation
warning. Positive coarse counts do not establish overlap for every continuous
state. A zero weighted estimate or collapsed interval with no matching prefixes
does not establish that the candidate has zero value or zero risk.

The behavior policy's probability of alternative actions is not supplied; a
logged action's propensity cannot prove support for an unchosen target action.
Unclipped and clipped estimates, direct and DR estimates, and support diagnostics
must be read together. No automated promotion rule is applied at this checkpoint.

## Prespecified slices and interpretation

Use initial observations to define overlapping episode populations:

- Service: topic `service_issue`, signal `unhappy`, or service risk above 0.78.
- Estimate with objection: topic `open_estimate` and signal in `price_objection`,
  `competitor`, `ask_spouse`, or `timing_delay`.
- Cold/ignored: heat `cold` or at least one ignored outreach.
- Membership: topic `membership_renewal` or status `expiring`/`lapsed`.

Later-turn ignored count at least two or patience at most zero defines only a
historical-state replay diagnostic. It is not an initial-state causal population:
changing the policy changes which later states are reached.

Report supplied terminal outcomes, cost, and outreach alongside reward. Off-policy
outcome estimates and intervals can lie outside probability bounds; do not clip
them into apparent certainty. Baseline action agreement is not performance.

## Verification and handoff

Before running comparisons, exact synthetic tests must establish PDIS and DR
arithmetic, mismatch zeroing, clipping semantics, terminal handling, and support
failure behavior. Check that evaluator fitting follows the fixed candidate and
that policy inputs exclude identifiers and logged metadata. Record source/data
hashes, seeds, action order, schema, model settings, and role IDs in results.

Finish with `artifacts/dev_baselines.json` and a brief checkpoint report. Weak
support or inconsistent estimates should be reported as inconclusive evidence.
Task 4's immediate-reward learner remains a subsequent checkpoint.
