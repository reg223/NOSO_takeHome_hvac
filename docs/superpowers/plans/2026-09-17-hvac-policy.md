# HVAC Follow-up Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking. This document plans implementation; it does not authorize a production rollout.

**Goal:** Deliver a reproducible, offline `act(observation, history)` policy with a defensible comparison against rules and simpler learned alternatives.

**Architecture:** Build an episode-aware data pipeline and evaluator first. Compare contextual rules, immediate-reward learning, and four-turn backward value learning. Learned policies share service safeguards and fall back to contextual rules when training support or estimated advantage is weak.

**Tech Stack:** Python, standard-library JSON, NumPy, scikit-learn, joblib, pytest. Use dense one-hot encoded categorical features and small tree ensembles; no neural network, paid API, or custom scoring simulator is needed initially. Pin the versions actually installed and verified during implementation.

**Spec:** `README.md`, informed by [Design ML Approach](https://chatgpt.com/g/g-p-6aaae38579008191a80902792aad0d28/c/6aaae3a0-02ec-83ea-a92e-68844cd96145) and [Reinforcement Learning Plan](https://chatgpt.com/g/g-p-6aaae38579008191a80902792aad0d28/c/6aabefa8-f78c-83ea-b6fb-3963ca2b93fe).

## Global constraints

- Return exactly one of the eight README action strings; episodes have at most four decisions.
- Use only decision-time observations as predictive inputs in version 1. Accept `history` but do not depend on it initially.
- Optimize the supplied undiscounted episode reward, with discount factor 1.0. Report cost separately; do not subtract it twice or clip large negative rewards.
- No network or training data dependencies at inference; models load relative to the policy file.
- Preserve supplied starter files as benchmarks. The checker is an interface check, not a score.
- Keep validation out of model fitting and tuning. Structural checks and preliminary aggregate counts have already been inspected; document that exposure rather than calling validation completely unseen.
- No performance claims from action agreement, average reward on matching rows, or a model's training fit.

## What is established locally

| Item | Training | Validation |
|---|---:|---:|
| Episodes | 24,000 | 5,000 |
| Decision rows | 74,733 | 15,565 |
| Turn-3 rows | 13,261 | 2,763 |
| Terminal rows | 24,000 | 5,000 |
| Observed actions | 8 | 8 |
| Minimum logged propensity | 0.00610296 | 0.00604721 |

There are no overlapping episode IDs or case IDs. `step` equals observation `turn` on all rows. Both splits have zero noncontiguous episodes, zero terminal-placement errors, and zero adjacent transition mismatches. Training action counts are: estimate nudge 23,433; ask objection 15,756; check in 13,409; scheduling 10,286; wait 5,190; membership 3,554; escalation 1,828; parking 1,277. Global coverage does not establish coverage in each customer context.

`submission/parse.py` currently reads lines but does not parse or return them. `starter/policy.py` simply delegates to `simple_rule`. The existing requirements contain torch, orjson, and pydantic despite the standard-library comment. There is no Git repository in the current directory, so do not assume commits or worktrees are available.

The checker combines unrelated observations into one rolling history and pairs candidate actions with logged next states. Version 1 will use `previous_actions`, `last_action`, and recency fields already present in the observation. A richer verified-history model is a later experiment, not a prerequisite.

## Decisions that refine the previous conversations

1. **Zero patience alone does not force parking.** Combine it with ignored outreach and recency. The earlier sample demonstrated that zero patience was not universally terminal.
2. **An explicit objection is not required to ask about one.** Allow discovery in open-estimate contexts while discouraging unchanged repeated questions.
3. **Model disagreement is a heuristic, not a calibrated confidence interval.** Use separate episode-level evaluation intervals for claims about policy improvement.
4. **Backward targets must follow the actual safeguarded future policy.** An unrestricted maximum would train for behavior that is never deployed.
5. **Use paired differences for selection.** Compare candidate minus rule return on the same episodes and bootstrap samples, rather than comparing two separate lower confidence bounds.
6. **Keep the first version small.** Rules plus immediate reward plus sequential reward are the core ladder. Behavior cloning and a custom simulator are optional and should not delay a complete submission.

## File map

| Path | Responsibility |
|---|---|
| `submission/parse.py` | Stream JSONL, validate rows, reconstruct episodes |
| `submission/audit.py` | Integrity, coverage, reward and sentinel summaries |
| `submission/features.py` | Explicit visible-field schema and deterministic transformations |
| `submission/rules.py` | Contextual fallback, service override, plausible action set |
| `submission/support.py` | Training-only state/action counts and eligibility |
| `submission/train.py` | Group split, fit candidates, write artifacts and metadata |
| `submission/value_models.py` | Shared action scoring and backward training |
| `submission/evaluate.py` | Fixed-policy value fitting, sequential estimators and intervals |
| `submission/policy.py` | Portable public entry point and cached model loading |
| `submission/models/` | Selected artifacts, configuration, feature schema, split metadata |
| `submission/README.md` | Installation, reproduction, interface and fallback behavior |
| `submission/report.md` | Findings, limitations and all seven required report answers |
| `submission/requirements.txt` | Verified pinned runtime and reproduction dependencies |
| `tests/` | Data, leakage, value-target, evaluator and packaging tests |
| `artifacts/` | Audit JSON, split IDs, experiment tables, episode scores and figures |

All commands below run from the repository directory and describe interfaces to implement. They are not available yet.

## Task 1: Produce the data audit and fixed development split

**Files:** Modify `submission/parse.py`; create `submission/audit.py`, `tests/test_data.py`, `artifacts/audit.json`, and `artifacts/splits.json`.

**Interfaces:** `load_episodes(path) -> list[list[dict]]`; inner lists are ordered logged rows. `audit(episodes) -> dict`. Store IDs rather than copies of raw logs in the split file.

- [x] Replace `readlines()` with line-by-line JSON decoding; identify errors by path and line number.
- [x] Check required keys, eight legal actions, finite rewards, finite propensities in `(0,1]`, turns 0–3, contiguous steps, exactly one final terminal row, and adjacent `next_observation` equality.
- [x] Treat `done` as authoritative for continuation. Inspect terminal `next_observation` conventions instead of imposing that terminal next states must be null. Confirm `PARK_THREAD` and `ESCALATE_TO_HUMAN` terminate; verify no earlier terminal occurs within an episode.
- [x] Quarantine incomplete episodes from return estimation. Do not silently label a missing final row as a terminal transition. Stop training if unexpected integrity failures exist until their treatment is documented.
- [x] Produce counts by turn/action/topic/signal, propensity quantiles, reward quantiles, outcome counts and cost totals. Inspect whether `99` consistently denotes absent contact; retain its indicator without assuming its meaning is proven.
- [x] Shuffle sorted training episode IDs with NumPy seed `20260621`: first 16,800 for policy fitting, next 3,600 for evaluator fitting, last 3,600 for development scoring. Repeated case IDs, if found within training, must remain grouped; adjust group sizes and record actual counts.
- [x] Save dataset hashes, split IDs and audit results. Never stratify using validation outcomes.

Tests must cover duplicated steps, cross-episode grouping, incomplete episodes, invalid propensities, and transition mismatch. For example:

```python
def test_incomplete_episode_is_rejected(tmp_path):
    path = tmp_path / 'bad.jsonl'
    row = {'episode_id': 'a', 'step': 0, 'observation': {'turn': 0},
           'action': 'WAIT', 'action_prob': 0.5, 'reward': 0,
           'done': False, 'next_observation': {'turn': 1}, 'info': {}}
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='incomplete'):
        load_episodes(path)
```

Run: `python -m submission.audit --train data/train_logs.jsonl --valid data/valid_logs.jsonl --output artifacts/audit.json --splits artifacts/splits.json`.

**Done when:** a saved audit establishes exactly which episodes can be used and every episode belongs to only one development role.

**Completed 2026-09-17:** 25 tests passed. Saved audit and SHA-256 hashes confirm 24,000 training and 5,000 validation episodes, zero quarantined episodes, no episode/case overlap, and disjoint 16,800/3,600/3,600 development roles. Reproduction commands and recency findings are in `artifacts/README.md`. Tasks 2–7 remain unstarted.

## Task 2: Build one feature contract and the rule benchmark

**Files:** Create `submission/features.py`, `submission/rules.py`, `tests/test_features.py`, `tests/test_rules.py`.

**Interfaces:** `featurize(observation) -> dict`; `improved_rule(observation, history) -> str`; `eligible_actions(observation) -> list[str]`; `service_override(observation) -> str | None`.

- [ ] Allow the documented observation fields plus the observed recency and last-action fields. Exclude `case_id`, row metadata, reward, propensity, `info`, `done`, and next state from features.
- [ ] Encode missing categories as `__missing__`; ignore unknown one-hot categories; impute numerical values using fit-set medians with missing indicators. Preserve the observed recency value and add `recency_is_99` indicators.
- [ ] Derive last action, each action's prior count, consecutive repeat count, and remaining decisions `4-turn`. Ignore passed history in version 1; do not read its rewards.
- [ ] Start with an explicit escalation override for unhappy signal or service risk greater than `0.78`, matching the supplied threshold. Label it a product safeguard, not an empirically optimal cutoff.
- [ ] Start the fallback in this order: service override; ignored count at least 2 → park; outbound recency 0 → wait; service issue → check in; maintenance/install-warranty topic → check in unless the last action was check in and inbound recency is nonzero, in which case wait; relevant expiring/lapsed membership → membership touch; open estimate → objection/readiness/nudge logic; otherwise wait.
- [ ] For open estimates: explicit objection → ask unless last action was ask and inbound recency is nonzero, then wait; positive signal → scheduling; no signal and last action estimate nudge → ask; otherwise estimate nudge. Patience zero is retained as a learned feature, not a standalone override.
- [ ] Membership relevance means membership-renewal topic or expiring/lapsed status outside service/maintenance priority. Extend exact topic names only after the audit identifies them and record the mapping.
- [ ] For learned candidates, always retain wait and park; allow escalation; add check in for service/maintenance/install-warranty contexts; add membership touch for membership relevance; add nudge/ask/scheduling for open-estimate context. Always include fallback action. Keep a no-filter ablation on development data to expose damage from overly restrictive rules.
- [ ] Verify invariance to changing forbidden fields and arbitrary cross-case history. Verify all eight actions have meaningful fixtures, escalation outranks parking, and zero patience does not force park.

Example contract test:

```python
def test_identifier_is_not_a_feature():
    assert featurize({'turn': 0, 'case_id': 'a'}) == featurize({'turn': 0, 'case_id': 'b'})
```

Run: `python -m pytest tests/test_features.py tests/test_rules.py -q`.

**Done when:** rules are explainable and feature extraction is identical at training and inference. Preserve original `simple_rule` and `always_check_in` as two additional benchmarks.

## Task 3: Build a trustworthy evaluator before optimizing

**Files:** Create `submission/evaluate.py`, `tests/test_evaluate.py`.

**Interfaces:** `score_episodes(policy, episodes, evaluator) -> list[dict]`; each record contains episode ID, initial-state slice labels, PDIS, DR, direct-value prediction, cumulative weights and replay action counts. `summarize(scores, seed=20260621, bootstrap_reps=1000) -> dict`.

- [ ] Report observed logging-policy return using complete episode reward sums as a reference, clearly separated from candidate estimates.
- [ ] Implement per-decision importance sampling (PDIS). For a deterministic candidate, set `rho_t = int(candidate_action == logged_action) / action_prob`; accumulate the product from episode start. Never restart the weight after a mismatch.
- [ ] Implement sequential doubly robust scoring using a separately fitted fixed-policy value evaluator:

```python
weight = 1.0
pdis = 0.0
dr = evaluator.value(rows[0]['observation'])
for row in rows:
    obs = row['observation']
    action = policy(obs, [])
    weight *= float(action == row['action']) / row['action_prob']
    future = 0.0 if row['done'] else evaluator.value(row['next_observation'])
    pdis += weight * row['reward']
    dr += weight * (row['reward'] + future - evaluator.q(obs, row['action']))
```

- [ ] Fit that evaluator backward on the evaluator-fit split, with target `reward + Q_next(next_observation, fixed_policy(next_observation))`, zero after termination. It evaluates a frozen policy; it does not maximize actions. Fit a distinct evaluator per policy.
- [ ] Score only the development-score split. Keep policy training, evaluator fitting and final scoring separate. Do not reuse the policy's maximizing Q-values as its evaluator.
- [ ] Report unclipped PDIS and DR, cumulative-weight clipping sensitivity at 10/50/100, maximum weights, and effective sample size `(sum(w)**2)/sum(w**2)` per turn. Zero denominator means zero ESS and an unsupported estimate. Clipped estimates are biased sensitivity analyses.
- [ ] Bootstrap complete scored episodes 1,000 times; use the same sampled IDs for every policy so return differences are paired. Label these intervals conditional on the fitted models; they do not capture all model bias or training uncertainty.
- [ ] Predefine initial-state slices: service issue/unhappy/risk above 0.78; open estimate with explicit objection; cold or already ignored; membership relevance. Slice definitions may overlap. Analyze later-turn patience/ignored counts as replay diagnostics, not initial-population causal slices.
- [ ] Add a tiny enumerable two-action/two-step fixture with known behavior probabilities and known target value. Assert weighted PDIS and DR equal that value in exact expectation. Also verify first-step mismatch zeros subsequent corrections, terminal future value is zero, and zero support is reported without division errors.

**Done when:** the evaluator passes exact synthetic checks and produces baseline tables with counts, intervals and support diagnostics. Synthetic tests validate mathematics; they are not a replacement HVAC simulator.

## Task 4: Fit the immediate-reward candidate and support gate

**Files:** Create `submission/support.py`, `submission/value_models.py`, `submission/train.py`, `tests/test_support.py`, `tests/test_value_models.py`.

**Interfaces:** `fit_support(episodes) -> SupportModel`; `SupportModel.count(observation, action) -> int`; `fit_models(episodes, mode, config) -> PolicyBundle`; `PolicyBundle.select(observation) -> str`.

- [ ] Count training support in cells `(turn, topic, signal_group, action)`. Signal groups: unhappy; positive; explicit objection; none/other. Preserve these counts in the artifact. Missing cells have zero support; broad counts are only a heuristic for local coverage.
- [ ] Fit one immediate-reward regressor per turn with candidate action as a one-hot feature. Start with `ExtraTreesRegressor(n_estimators=100, max_depth=8, min_samples_leaf=30, random_state=seed, n_jobs=-1)` and squared-error loss. Fit preprocessing only on the policy-fit set.
- [ ] Fit five replicas using whole-episode bootstrap samples and seeds 100–104. Score candidate action means and paired action-versus-fallback prediction differences across replicas. Do not use individual-tree variability as an episode bootstrap.
- [ ] Permit a learned departure only when both the candidate and fallback have at least 30 cell examples and `mean(Q_candidate-Q_fallback) - lambda*std(Q_candidate-Q_fallback) > margin`. Otherwise return fallback. Service overrides precede this gate. If the fallback itself lacks support, label that fallback decision unsupported rather than calling it statistically safe.
- [ ] Try only four gate settings initially: `(lambda, margin)` in `(0,0), (1,0), (1,1), (2,1)`. These are development configurations in reward units, not guaranteed confidence levels. Fix the model capacity and support threshold initially to limit the search.
- [ ] Save per-policy action distribution, fallback rate, support-failure rate and dev evaluation table. Select one immediate-reward configuration on paired dev evidence.

Run: `python -m submission.train --mode immediate --splits artifacts/splits.json --output artifacts/immediate`.

**Done when:** a reproducible learned baseline can be evaluated under the same rules and feature contract as the eventual sequential policy.

## Task 5: Add backward sequential learning

**Files:** Extend `submission/value_models.py`, `submission/train.py`, `tests/test_value_models.py`.

- [ ] Reuse identical feature, model, support and gate settings for a fair comparison. Train turn 3 first using immediate reward targets.
- [ ] Freeze the complete turn-3 selector, including overrides and fallback. Train turn 2 on reward plus predicted turn-3 value of the action that selector actually chooses. Repeat for turns 1 and 0.
- [ ] Use the ensemble mean Q at the chosen next action as continuation value. The gate controls selection; do not subtract its uncertainty penalty from the reward target.
- [ ] Set continuation to zero whenever `done` is true, including `max_steps`; reject a nonterminal turn-3 row as inconsistent with the four-step contract.
- [ ] Train each bootstrap replica from its resampled episodes. Bootstrap disagreement remains a heuristic because common support rules and shared data can induce correlated errors.
- [ ] Test exact target arithmetic and terminal handling. With reward `-1`, next chosen action value `6`, and a disallowed alternative worth `100`, assert target `5`, not `99`. For the same terminal row assert target `-1`.
- [ ] Compare the same four gate configurations on development data. Inspect cases where asking an objection is preferred by sequential learning but not immediate reward; describe predictions, not proven counterfactual outcomes.
- [ ] Run no-support-gate and no-action-filter ablations only on development data. Do not remove service safeguards for a deployable candidate.

Run: `python -m submission.train --mode sequential --splits artifacts/splits.json --output artifacts/sequential`.

**Done when:** the experiment isolates the value of accounting for future decisions, with a tested connection between training targets and deployed behavior.

## Task 6: Freeze, evaluate, and choose the submission

**Files:** Write `artifacts/selection.json`, `artifacts/final_results.json`, `submission/models/metadata.json`.

- [ ] Freeze the strongest rule configuration, one immediate-reward candidate, one sequential candidate, and the provided benchmarks based on development results. Write chosen parameters and dataset hashes before final validation.
- [ ] Refit selected policy candidates on the union of policy-fit and development-score episodes (20,400 episodes); retain 3,600 evaluator-fit episodes for independently fitting each final evaluator. Freeze these artifacts before scoring validation. Report that not all training episodes fit the policy.
- [ ] Score validation once under the frozen protocol. Compare simple rule, always check in, improved rule, immediate reward and sequential hybrid. Include the logging-policy observed return reference.
- [ ] Promote the sequential model over improved rules only if the paired DR improvement interval has lower endpoint greater than zero, its direct estimate has the same sign, and supported PDIS does not show credible degradation. Use the same rule to consider immediate reward if sequential fails.
- [ ] Treat overall minimum per-turn ESS below 100 or any normalized individual weight above 10% as a warning that importance-weighted evidence is insufficient for promotion. These are declared engineering heuristics, not statistical guarantees. Repeat interpretation at ESS thresholds 50 and 200 in the report without retuning policies.
- [ ] Require zero replay violations of explicit service overrides. Report small or poorly supported slices as inconclusive; do not manufacture outcome rates from action replay. If the learned-policy case is inconclusive, ship improved rules and report the learned experiment honestly.
- [ ] Keep outcome estimates separate: for opt-out/conversion use sequential evaluation with a terminal event indicator as reward; for cost use per-step logged cost. Raw logged outcome rates belong only to the historical-policy reference.

**Done when:** one policy is selected under a documented rule, all estimates have limitations attached, and there has been no validation-driven parameter tuning. Validation is used for this frozen final selection, so it is not an independent test of that selection; the hidden evaluator provides the next external check.

## Task 7: Package and write the interview-ready report

**Files:** Create `submission/policy.py`, `submission/README.md`, `submission/report.md`, `submission/requirements.txt`, and `tests/test_policy.py`.

- [ ] Expose exactly `act(observation, history)`. Resolve sibling modules and model paths from `Path(__file__).resolve().parent`; make imports work with the supplied path-based checker and from an unrelated working directory.
- [ ] Load artifacts once; validate action order and feature schema. Missing/corrupt artifacts must trigger a documented rule fallback without repeated loading. A clean-install test must assert that a learned submission actually loaded its model so fallback cannot conceal packaging errors.
- [ ] Verify empty history, contaminated history, missing optional fields, unseen categories, every turn, deterministic repeat calls, no observation mutation, and all validation observations returning a legal action.
- [ ] Run `python starter/check_policy.py --policy submission/policy.py --logs data/valid_logs.jsonl --limit 15565` and `python -m pytest tests -q`.
- [ ] Verify in a fresh environment with network disabled at runtime. Include exact dependency versions, training command, dataset hashes, seeds, selected configuration, model sizes and measured inference latency. Do not claim offline installation unless dependencies are already available or vendored.
- [ ] Write seven report sections matching the README: action rationale; baseline comparison; overall and slice estimates; old-log limitations; visible fields and leakage; three failure modes; 5% rollout monitoring and rollback.
- [ ] Explain offline assumptions explicitly: recorded propensities must be correct conditional logging probabilities; hidden confounding may remain; support can be weak; changed policies change later states; evaluation models can extrapolate incorrectly. The logs cannot establish all these assumptions.
- [ ] Make the three failure modes concrete: unsupported action overvaluation, noisy context causing inappropriate pressure or missed recovery, and changed trajectories/customer mix invalidating offline estimates.
- [ ] Propose episode-level randomized assignment for a future 5% pilot, with the incumbent as control. Immediate rollback for invalid/runtime failures or a confirmed critical service-handling miss. As provisional business thresholds for approval before launch: pause if a predefined sequential statistical boundary supports opt-out harm over 1 percentage point, complaints over 0.5 points, or escalation workload over 20% relative; confirm tolerances and monitoring design before traffic. Monitor conversion, cost, touches, parking, fallback and inference latency as well. These are proposed tolerances, not data-derived facts or authorization to launch.

**Done when:** the required directory is portable, results are reproducible, and every report question is answered with evidence or an explicit limitation.

## Suggested working order

First checkpoint: Tasks 1–2 produce an audit and an explainable candidate. Second: Task 3 makes comparisons credible. Third: Task 4 establishes whether learning helps at all. Fourth: Task 5 tests delayed value. Final: Tasks 6–7 freeze the result and deliver the submission. If time is constrained, finish a tested rule policy and its evaluation/report before adding bootstrap sequential models.

The most useful next action is **Task 1: complete and save the audit**, followed immediately by the feature/rule contract. Do not begin by training torch models or optimizing on validation.

## Method references

- [Tree-Based Batch Mode Reinforcement Learning](https://jmlr.org/papers/v6/ernst05a.html): precedent for fitting action values through successive supervised regression problems.
- [Doubly Robust Off-policy Value Evaluation for Reinforcement Learning](https://proceedings.mlr.press/v48/jiang16.html): sequential off-policy evaluation combining value prediction and importance-weighted corrections. Its guarantees depend on assumptions; finite samples and poor support still limit conclusions.
