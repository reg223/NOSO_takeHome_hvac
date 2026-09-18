# Project contribution rules

## Commit messages

Every commit message must begin with a lowercase tag of 3–5 characters,
followed immediately by a colon and a space, then a concise description of the
work:

```text
<tag>: <description>
```

Use a tag that names the kind of work, for example:

- `misc:` documentation, requirements, configuration, or maintenance
- `data:` dataset parsing, validation, or audit work
- `feat:` new product or policy behavior
- `fix:` a bug correction
- `test:` tests and test infrastructure
- `train:` model training, fitting, or training artifacts
- `eval:` evaluation, scoring, or analysis

Tags must be lowercase and 3–5 characters long. Keep the description specific
and written in the imperative mood where practical.

When a new category of work is introduced, add its tag and meaning to this
list before using it in a commit. Commit categories are metadata, so they do
not correspond to `.gitignore` patterns; add a path to `.gitignore` only when
the category also introduces generated or local-only files.

## Project constraints

- Keep the public policy interface as `act(observation, history)` and return
  exactly one of the eight actions defined in `README.md`.
- Use only decision-time information. Never use identifiers, rewards,
  propensities, terminal metadata, next observations, final outcomes, or
  other hidden fields as predictive inputs.
- Treat `99` recency values as observed values with explicit indicators; do not
  infer their real-world meaning from association alone.
- Train and tune only on `data/train_logs.jsonl`. Freeze candidates and
  selection rules before using `data/valid_logs.jsonl` for the one-time final
  evaluation.
- Evaluate complete trajectories with undiscounted episode reward (`gamma =
  1.0`), and report costs separately without subtracting them twice.
- Use support checks, propensity diagnostics, paired episode comparisons, and
  clearly labeled PDIS/DR limitations. Agreement with logged actions and
  training fit are not policy-performance evidence.
- Preserve the starter policy and at least two simple baselines as comparison
  references. The local checker validates the interface; it is not a score.
- Keep runtime dependencies, model artifacts, and path resolution inside
  `submission/`; inference must work offline and from an unrelated working
  directory.
- Keep service safeguards ahead of learned selection. Unsupported or
  inconclusive learned decisions must fall back to the contextual rules.
- Do not authorize or imply a production rollout. Any 5% pilot requires a
  separate product decision, monitoring plan, and rollback approval.

## Local workflow

- Follow the implementation plan in
  `docs/superpowers/plans/hvac-offline-rl-policy-implementation-plan.md` as the
  current source of project sequencing and file responsibilities.
- Build the evaluator before trusting candidate comparisons; keep policy-fit,
  evaluator-fit, development-score, and validation roles separate.
- Use only visible-field, deterministic features shared by training and
  inference. Keep contextual rules explainable and retain multiple plausible
  actions in ambiguous contexts.
- Test leakage resistance, arbitrary history contamination, exact sequential
  target arithmetic, terminal handling, support failures, packaging, and
  offline runtime behavior.
- Record hashes, seeds, schemas, action order, gate settings, and fallback
  rates in reproducible artifacts. Do not tune thresholds after validation.
