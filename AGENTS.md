# Project instructions

## AVNav evaluation integrity

These rules apply to every AVNav experiment, including data collection,
offline-RL training, validation, and test evaluation.

### No rule-based policy improvements

- Evaluate the policy's own action outputs. Do not add hand-written control
  rules or action post-processing to improve SR, SPL, or other reported
  metrics.
- In particular, do not override actions after collisions, count collisions
  to trigger turns or recovery sequences, choose directions with depth-based
  heuristics, or tune a STOP-confidence threshold to improve results.
- Do not apply different rule-based settings to different checkpoints or
  splits. Validation and test must use the same predeclared evaluation
  protocol and the unmodified policy outputs.
- If the benchmark explicitly requires an action adapter, treat its documented
  behavior as part of the fixed benchmark protocol: record it before running,
  apply it identically to every model and split, and never tune it against
  validation or test metrics. Do not introduce a new adapter as an experimental
  improvement.

### Keep data splits isolated

- Training, reward shaping, replay buffers, data augmentation, and fitted
  preprocessing may use training-split data only. Never copy, merge, or derive
  training examples from validation or test episodes.
- Use validation data only to select checkpoints and tune experiment settings.
  Do not train or fine-tune on validation data.
- Keep test data for final evaluation after the model, checkpoint, and
  settings are frozen. Do not use test observations, actions, rewards, labels,
  trajectories, or metrics to train, augment, tune, select, or debug a candidate.
- Preserve split manifests and episode/scene IDs. Before training, check and
  record the provenance and overlap of all data sources against the train,
  validation, and test manifests. Any validation/test leakage invalidates the
  affected run; do not report it as a valid improvement.

### Record the evaluation protocol

- Record the exact checkpoint, configuration, action handling, data-source
  manifests, split IDs, and checkpoint-selection rule for each run.
- Select checkpoints using validation only, then evaluate the selected
  checkpoint on the complete test splits. Report per-split episode counts and
  evaluation errors alongside SR/SPL.
- If a run violates either the no-rule-based-improvement rule or split
  isolation, label its metrics non-compliant and do not use them to claim an
  improvement.
