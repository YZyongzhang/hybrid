# AVNav RGB Single-Frame Foundation Ablation

This isolated ablation keeps the AVNav audio encoder, transformer blocks,
fusion layers, and action head while removing depth and the previous RGB frame.
The visual input is only the current RGB frame. The existing `avnav` and
`avnav_real` implementations are not modified.

Train with:

```bash
python -m experiment.avnav_real_single_frame.train_foundation \
  --config experiment/avnav_real_single_frame/config/foundation_single_frame.yaml
```

Run the validation-only closed-loop diagnostic with:

```bash
python -m experiment.avnav_real_single_frame.evaluate_foundation \
  --config experiment/avnav_real_single_frame/config/foundation_single_frame.yaml
```

The evaluator accepts validation splits only. It runs the foundation action
head's raw argmax, without an offline-RL actor, collision predictor, or action
recovery.
