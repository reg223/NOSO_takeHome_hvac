# HVAC Follow-up Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a reproducible, offline `act(observation, history)` policy that improves on `simple_rule` only where logged evidence supports the improvement, while satisfying the README submission contract and preserving service safeguards.

**Architecture:** Build an episode-aware data/audit layer, an explicit decision-time feature contract, and an explainable contextual-rule fallback. Compare that fallback with an immediate-reward model and a four-turn backward value model. Put a small hard-safety layer around all candidates; learned actions must pass local support and conservative advantage gates, otherwise the policy falls back to the contextual rules.

**Tech Stack:** Python 3.9, standard-library JSON/path handling, NumPy, scikit-learn tree regressors, joblib, and pytest. No neural network, paid API, network access, or custom simulator is required for the first complete submission.

**Spec:** `/Users/reg223/Desktop/Y6S1/sideP/hvac_followup_policy_optimization/README.md`. The earlier plan at `/Users/reg223/Desktop/Y6S1/sideP/hvac_followup_policy_optimization/docs/superpowers/plans/2026-09-17-hvac-policy.md` is prior design input, not an authority that overrides the README.

## Global Constraints

- Return exactly one of the README actions: `WAIT`, `CHECK_IN`, `ESTIMATE_NUDGE`, `ASK_OBJECTION`, `OFFER_SCHEDULING`, `MEMBERSHIP_TOUCH`, `PARK_THREAD`, or `ESCALATE_TO_HUMAN`.
- Preserve `act(observation, history)` and use only information visible at the current decision step.
- Treat the supplied reward as undiscounted episode reward (`gamma = 1.0`) for the primary return analysis; report costs separately and do not subtract them twice.
- Train and tune only on `data/train_logs.jsonl`; use `data/valid_logs.jsonl` only after the candidate and selection rule are frozen.
- Keep all model files and runtime dependencies inside `submission/`; inference must work without network access and from an unrelated working directory.
- Preserve the starter policy and at least two simple baselines as comparison references. The local checker is an interface check, not a score.
- Never treat action agreement, training fit, or reward on rows where a candidate matches the logged action as policy-performance evidence.
- `case_id`, `episode_id`, `step`, `reward`, `action_prob`, `info`, `done`, `next_observation`, and final outcomes are not predictive inputs.
- Keep `99` recency values as explicit observed indicators; do not claim their real-world semantics from association alone.
- Do not interpret offline estimates as causal unless their support and assumptions are stated.
- This plan does not authorize a production rollout. Any 5% pilot requires a separate product decision and monitoring approval.

## What the Audit Establishes

The audit artifact reports 24,000 training episodes / 74,733 rows and 5,000 validation episodes / 15,565 rows. It reports zero train/validation overlap by case or episode ID, no quarantined episodes, no unexpected integrity failures, and terminal next observations as null for all included episodes. All eight actions occur in both splits.

The data is sequential: validation has 5,000 turn-0 rows, then 4,239, 3,563, and 2,763 rows on turns 1–3. Validation terminal outcomes include 388 booked appointments, 316 sold estimates, 114 membership conversions, 216 positive reviews, 1,264 opt-outs, 350 escalations, and 297 parked threads. The validation episode-return median is -12 and the 95th percentile is about 100.3, so the policy must be evaluated on full trajectories and outcome trade-offs rather than immediate reward alone.

Support is uneven. Validation contains 4,945 `ESTIMATE_NUDGE`, 3,201 `ASK_OBJECTION`, 2,846 `CHECK_IN`, 2,097 `OFFER_SCHEDULING`, 1,048 `WAIT`, 781 `MEMBERSHIP_TOUCH`, 350 `ESCALATE_TO_HUMAN`, and 297 `PARK_THREAD` rows. Logged propensities range from about 0.006 to 0.632. Therefore, low-propensity regions require support warnings, clipped/sensitivity estimates, or baseline fallback; they do not justify confident counterfactual claims.

## Final Design Reasoning

1. Customers can be assumed to be generally goal-directed, but not endlessly tolerant. The policy should not model unsupported adversarial bargaining or prompt attacks outside the eight-action space.
2. An observation category does not determine one action. A price objection can rationally lead to `ASK_OBJECTION`, `WAIT`, `ESTIMATE_NUDGE`, or scheduling depending on turn, patience, ignored/touch counts, recency, and previous actions.
3. The hard-rule layer must stay small. Genuine service recovery/high risk and clearly exhausted engagement deserve protection; most other behavior should remain a soft preference learned from trajectories.
4. Noisy fields such as `track_hint`, urgency, risk, price sensitivity, and relationship score are evidence, not ground truth. Contradictions must be features the model resolves.
5. `ASK_OBJECTION` has information value and `WAIT` has timing value. A policy that optimizes only immediate reward will systematically undervalue both.
6. `PARK_THREAD` and `ESCALATE_TO_HUMAN` are terminal and deserve a higher decision margin than reversible actions.
7. The final policy should optimize the supplied episode reward while separately reporting conversion/service outcomes, opt-outs, human escalation, touches, parking, and cost.

## File Map

| Path | Responsibility |
|---|---|
| `submission/parse.py` | Stream JSONL, validate rows, reconstruct ordered episodes |
| `submission/audit.py` | Integrity checks, coverage summaries, hashes, fixed train-only splits |
| `submission/features.py` | One visible-field schema and deterministic train/inference transforms |
| `submission/rules.py` | Contextual fallback, service safeguards, candidate action eligibility |
| `submission/support.py` | Training-only local state/action support and propensity diagnostics |
| `submission/value_models.py` | Immediate and backward sequential action-value models |
| `submission/train.py` | Reproducible fitting and artifact creation |
| `submission/evaluate.py` | Fixed-policy evaluators, PDIS/DR, bootstrap and slice reports |
| `submission/policy.py` | Portable public entry point and cached model loading |
| `submission/models/` | Selected model, schema, configuration, metadata, and hashes |
| `submission/README.md` | Exact reproduction and runtime instructions |
| `submission/report.md` | Answers to every README report question |
| `submission/requirements.txt` | Verified pinned runtime/reproduction dependencies |
| `tests/` | Data, leakage, value-target, evaluator, and packaging tests |
| `artifacts/` | Audit, split IDs, experiment tables, scores, and figures |

## Task 1: Reconstruct episodes and freeze development roles

**Files:** Create `submission/parse.py`, `submission/audit.py`, `tests/test_data.py`, and `artifacts/audit.json`, `artifacts/splits.json`.

**Interfaces:** `load_episodes(path: str) -> list[list[dict]]`; `validate_episode(rows: list[dict]) -> None`; `audit(train_path: str, valid_path: str) -> dict`; `write_splits(train_episode_ids: list[str], seed: int = 20260621) -> dict[str, list[str]]`.

- [x] Write tests for duplicate steps, noncontiguous steps, invalid actions, non-finite rewards, propensities outside `(0, 1]`, transition mismatches, early terminal rows, missing terminal rows, and cross-split ID overlap.
- [x] Implement line-numbered JSON decoding and ordered grouping by `episode_id`.
- [x] Validate `step == observation["turn"]`, contiguous steps from 0, `done` authority, terminal placement, and adjacent `next_observation` equality.
- [x] Confirm that `PARK_THREAD` and `ESCALATE_TO_HUMAN` terminate in the supplied data; do not hard-code a different terminal convention without documenting it.
- [x] Produce counts by turn/action/topic/signal, outcome counts, reward and propensity quantiles, costs, and recency sentinel summaries.
- [x] Shuffle sorted training episode IDs with seed `20260621` into 16,800 policy-fit, 3,600 evaluator-fit, and 3,600 development-score episodes. Keep repeated case IDs grouped if the audit finds any.
- [x] Save train/validation SHA-256 hashes, split IDs, and audit results. Never use validation outcomes for split construction.

Run: `python -m submission.audit --train data/train_logs.jsonl --valid data/valid_logs.jsonl --output artifacts/audit.json --splits artifacts/splits.json`.

**Done when:** all integrity tests pass, every usable episode belongs to exactly one development role, and the audit artifact is reproducible.

## Task 2: Define features and the contextual-rule benchmark

**Files:** Create `submission/features.py`, `submission/rules.py`, `tests/test_features.py`, and `tests/test_rules.py`.

**Interfaces:** `featurize(observation: dict, history: list[dict] | list[str]) -> dict[str, float | int | str]`; `fallback_action(observation: dict, history: list[dict] | list[str]) -> str`; `eligible_actions(observation: dict, history: list[dict] | list[str]) -> list[str]`; `service_override(observation: dict, history: list[dict] | list[str]) -> str | None`.

- [x] Encode only README-visible fields plus verified decision-time recency fields. Exclude IDs, rewards, propensities, `info`, `done`, next state, and final outcomes.
- [x] Encode missing categorical values as `__missing__`, numeric missingness explicitly, and recency `99` both as its observed value and as an indicator.
- [x] Derive turn, remaining decisions, last action, action counts, repeated-action counts, consecutive waits, and whether objection/nudge/scheduling has already occurred.
- [x] Use the supplied `history` only to derive prior actions and decision-time state. Ignore any reward or outcome fields in history. Test arbitrary cross-case history contamination.
- [x] Preserve the original `simple_rule` as `simple_rule_action`; add `always_check_in` and `always_estimate_nudge` as simple references.
- [x] Implement a contextual fallback that prioritizes service recovery, engagement exhaustion, relationship/membership context, then open-estimate intent. Keep multiple actions eligible in ambiguous cases.
- [x] Treat `service_risk_score > 0.78` only as an initial product-safeguard configuration, not as an empirically optimal threshold. Record the threshold and expose explicit alternatives. Unit tests verify configuration; performance experiments on development data await Task 3’s evaluator.
- [x] Do not make zero patience alone terminal. Combine patience with ignored outreach, recency, touch count, and history.
- [x] Allow objection discovery in an open-estimate context even without an explicit objection, but discourage unchanged repeated objection questions.

Example test:

```python
def test_identifier_is_not_a_feature():
    left = featurize({"turn": 0, "case_id": "a"}, [])
    right = featurize({"turn": 0, "case_id": "b"}, [])
    assert left == right
```

Run: `python -m pytest tests/test_features.py tests/test_rules.py -q`.

**Done when:** feature extraction is identical at train and inference, the fallback is explainable, and every ambiguous context retains more than one plausible action.

## Task 3: Build the evaluator before optimizing

**Files:** Create `submission/evaluate.py`, `tests/test_evaluate.py`, and `artifacts/dev_baselines.json`.

**Interfaces:** `score_episodes(policy, episodes, evaluator) -> list[dict]`; `fit_fixed_policy_evaluator(episodes, policy, config) -> FixedPolicyEvaluator`; `summarize(scores, seed: int = 20260621, bootstrap_reps: int = 1000) -> dict`.

- [ ] Report historical logged-policy episode return separately from candidate estimates.
- [ ] Implement per-decision importance sampling with cumulative ratio `1[action == logged_action] / action_prob`; do not restart after a mismatch.
- [ ] Implement sequential doubly robust evaluation using a separately fitted fixed-policy evaluator. The evaluator must follow the frozen candidate policy, not maximize over actions.
- [ ] Use zero continuation after `done`, including `max_steps`; distinguish terminal action outcomes from ordinary continuation.
- [ ] Report unclipped PDIS/DR, propensity clipping sensitivity at 10/50/100, maximum cumulative weights, and effective sample size `(sum(w) ** 2) / sum(w ** 2)` by turn.
- [ ] Bootstrap complete episodes 1,000 times with paired episode samples shared across policies. Label intervals conditional on fitted models; they do not remove hidden-confounding or extrapolation bias.
- [ ] Predefine slices: service issue/unhappy/high-risk, open estimate with objection, cold/ignored, and membership relevance. Treat later-turn patience/ignored slices as replay diagnostics rather than causal initial-state populations.
- [ ] Add an enumerable two-action/two-step fixture that verifies PDIS, DR, mismatch-zeroing, terminal continuation, and zero-support handling.

Evaluate at minimum: `always_check_in`, `always_estimate_nudge`, `simple_rule`, and the contextual fallback. Add a behavior-cloning reference only if it can be implemented without delaying the core policy.

**Done when:** the evaluator passes exact synthetic tests and produces paired baseline tables with outcomes, costs, intervals, support, and ESS.

## Task 4: Fit the immediate-reward candidate

**Files:** Create `submission/support.py`, `submission/value_models.py`, `submission/train.py`, `tests/test_support.py`, and `tests/test_value_models.py`.

**Interfaces:** `fit_support(episodes) -> SupportModel`; `SupportModel.count(observation, action) -> int`; `fit_policy_bundle(episodes, mode: str, config: dict) -> PolicyBundle`; `PolicyBundle.action_values(observation, history) -> dict[str, ValueEstimate]`; `PolicyBundle.select(observation, history) -> str`.

- [ ] Count local support by `(turn, topic, signal_group, action)` and report missing cells. Use broad counts as a warning, not proof of local coverage.
- [ ] Fit one per-turn `ExtraTreesRegressor(n_estimators=100, max_depth=8, min_samples_leaf=30, random_state=seed, n_jobs=-1)` using action as an input feature and immediate reward as the target.
- [ ] Fit five whole-episode bootstrap replicas with seeds 100–104. Use between-replica disagreement as an uncertainty heuristic, not a calibrated confidence interval.
- [ ] Permit a learned departure only when candidate and fallback support exceed the declared threshold and the lower confidence heuristic for candidate-minus-fallback value exceeds the declared margin.
- [ ] Use a small fixed development grid for `(uncertainty penalty, margin)`; do not search model capacity and gates simultaneously.
- [ ] Apply service safeguards before learned selection and keep `WAIT`, fallback, and terminal actions available according to the action-eligibility contract.
- [ ] Save model metadata, feature schema, split hashes, action order, support counts, gate settings, and fallback rate.

Run: `python -m submission.train --mode immediate --splits artifacts/splits.json --output artifacts/immediate`.

**Done when:** the immediate-reward candidate is reproducible, explainable, and evaluated using the same contract as the rule baselines.

## Task 5: Add backward sequential learning

**Files:** Extend `submission/value_models.py` and `submission/train.py`; extend `tests/test_value_models.py`.

- [ ] Reuse the same features, model capacity, support gates, safeguards, and development protocol as Task 4.
- [ ] Train turn 3 first using immediate reward targets, then train turns 2, 1, and 0 backward.
- [ ] Construct each target as `reward + next_value`, where `next_value` is the value of the action selected by the frozen safeguarded future policy; do not use an unrestricted max over unavailable or unsafe actions.
- [ ] Set `next_value = 0` when `done` is true, including `max_steps`.
- [ ] Test exact arithmetic: reward `-1` plus selected next value `6` must produce target `5`, even if a disallowed action has value `100`; terminal reward `-1` must produce target `-1`.
- [ ] Compare immediate and sequential candidates on the same development episodes. Inspect objection-question and waiting cases as predictions, not proven counterfactual facts.
- [ ] Run no-support-gate and no-action-filter ablations only on development data and never as deployable candidates.

Run: `python -m submission.train --mode sequential --splits artifacts/splits.json --output artifacts/sequential`.

**Done when:** the experiment isolates delayed-value learning and its targets match the behavior the deployed selector can actually produce.

## Task 6: Freeze selection and evaluate validation once

**Files:** Write `artifacts/selection.json`, `artifacts/final_results.json`, and `submission/models/metadata.json`.

- [ ] Freeze the contextual fallback, one immediate-reward candidate, one sequential candidate, baselines, feature schema, model settings, gates, and dataset hashes from development evidence.
- [ ] Refit selected policy candidates on the policy-fit plus development-score episodes (20,400 episodes) while retaining evaluator-fit episodes for independent fixed-policy evaluation. Record that not all training episodes were used for policy fitting.
- [ ] Freeze all artifacts before reading validation outcomes for model selection.
- [ ] Score validation once under the frozen protocol. Compare at least `always_check_in`, `always_estimate_nudge`, `simple_rule`, contextual fallback, immediate reward, and sequential hybrid.
- [ ] Promote the sequential model only if paired DR improvement over the best rule has a positive lower interval endpoint, the direct estimate agrees in sign, supported PDIS shows no credible degradation, and no protected action is violated. Apply the same rule to the immediate model if sequential learning fails.
- [ ] Treat low ESS, large normalized weights, small slices, or weak action support as inconclusive evidence, not as a reason to tune thresholds after validation.
- [ ] Keep raw logged outcome rates as historical-policy references. Estimate candidate return, terminal outcomes, and cost with clearly labeled methods and limitations.
- [ ] If evidence is inconclusive, select the contextual fallback and report the learned experiment honestly.

**Done when:** one candidate is selected by a predeclared rule and all validation claims include support and off-policy limitations.

## Task 7: Package the submission and write the report

**Files:** Create `submission/policy.py`, `submission/README.md`, `submission/report.md`, `submission/requirements.txt`, and `tests/test_policy.py`.

- [ ] Expose exactly `act(observation, history)` and resolve model paths from `Path(__file__).resolve().parent`.
- [ ] Load artifacts once, validate action order/schema, avoid mutating observations or history, and make corrupt/missing learned artifacts trigger a documented rule fallback.
- [ ] Add a clean-install test that proves the learned artifact loads when present; fallback must not silently hide packaging errors during verification.
- [ ] Test empty history, missing optional fields, unseen categories, every turn, deterministic repeat calls, arbitrary history contamination, and all validation observations returning legal actions.
- [ ] Run `python starter/check_policy.py --policy submission/policy.py --logs data/valid_logs.jsonl --limit 15565` and `python -m pytest tests -q`.
- [ ] Pin only dependencies actually required by the packaged policy and training commands. Verify runtime with network disabled.
- [ ] Write report sections matching the README: product action rationale; baselines; overall and at least three slice results; old-log limitations; visible fields and leakage; three failure modes; 5% rollout monitoring and rollback.
- [ ] State the central offline limitations: only logged actions have outcomes, propensities may be misspecified, hidden confounding remains, support is weak in some regions, and a changed policy changes later states.
- [ ] Make failure modes concrete: unsupported action overvaluation, noisy context causing pressure or missed recovery, and distribution/trajectory shift.
- [ ] Describe a future episode-level randomized pilot with the incumbent as control, runtime/service rollback triggers, and monitoring of conversion, opt-outs, complaints, escalation workload, cost, touches, parking, fallback rate, and latency. Mark thresholds as proposed product tolerances, not data-derived facts.

**Done when:** `submission/` is portable, reproducible, offline-capable, checker-valid, and its report answers every README question without overstating evidence.

## Execution Order and Checkpoints

1. Finish Tasks 1–2: audit, visible-field contract, contextual fallback, and baseline smoke tests.
2. Finish Task 3 before trusting any candidate comparison.
3. Finish Task 4 to establish whether immediate reward learning helps at all.
4. Finish Task 5 only after the immediate candidate is reproducible; it is the delayed-value experiment, not a prerequisite for a rule submission.
5. Finish Tasks 6–7 for frozen validation, packaging, and the interview-ready report.

If time is constrained, a tested contextual fallback with honest baseline/offline evaluation is preferable to an unverified neural or unconstrained offline-RL policy.

## Method References

- [Tree-Based Batch Mode Reinforcement Learning](https://jmlr.org/papers/v6/ernst05a.html) — precedent for successive supervised action-value regression.
- [Doubly Robust Off-policy Value Evaluation for Reinforcement Learning](https://proceedings.mlr.press/v48/jiang16.html) — sequential off-policy evaluation with explicit support assumptions.

## Checkpoint 1 completion notes (2026-09-17)

Tasks 1–2 are complete. Reproduction: run `.venv/bin/python -m pytest tests -q`,
the Task 1 audit command, and `.venv/bin/python scripts/check_checkpoint_one.py`.
Training-state smoke results are in `artifacts/task2_checks.json`; the feature,
history, safeguard, and eligibility contract is in
`docs/task2-feature-and-rule-contract.md`.

Documented implementation decisions:

- External history is accepted but ignored because the supplied checker can mix
  cases. Visible `previous_actions` supplies episode-specific history features.
- The streaming loader requires ordered, contiguous episode blocks and rejects
  reused IDs or out-of-order steps rather than silently repairing corrupt logs.
- Numeric missingness is `None` in scalar extraction and an explicit indicator
  plus fit-role imputation in encoded model inputs.
- The existing two-ignored-outreach exhaustion rule remains the contextual
  benchmark; zero patience alone never parks. It is a heuristic, not an optimized
  terminal decision. Context and prior outreach inform other branches.
- The ID-only `write_splits` helper cannot infer case groups; the production audit
  uses full episodes and `make_splits` to preserve case grouping.
- The already-inspected validation aggregate audit was regenerated without
  changing either frozen artifact. No candidate validation scoring or tuning
  occurred. Baseline action counts are smoke diagnostics, not performance evidence.
- Service-threshold alternatives are configurable and unit-tested. Comparing
  their value on development data requires Task 3 and remains deferred to it.

The next checkpoint is Task 3: exact synthetic evaluator tests and paired baseline
estimates with support, ESS, costs, and explicit off-policy limitations.
