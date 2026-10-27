# AVWAN without the depth image input

This is a separate AVWAN foundation-training experiment. It keeps AVWAN's
audio, visual-patch and visual/fusion Transformer, geometry and acoustic map
encoders, waypoint mask, prediction heads, supervised objectives, optimizer,
and validation checkpoint selection. The only model-input change is that the
visual encoder receives the previous and current RGB frames (6 channels) and
does not receive the two-frame depth image.

The existing AVWAN sources are imported as shared components and are not
modified. The dedicated data converter stores RGB, spectrogram, geometry map,
acoustic map, and labels in a separate cache; it does not copy depth arrays to
that cache. `gm` and the other AVWAN inputs remain enabled to preserve the
existing architecture's other modalities.

## Train

From the repository root, edit `config/foundation.yaml` if the local training
and validation source paths differ, then run:

```bash
python experiments/avwan-no-depth/train.py --stage all
```

`--stage prepare` builds the isolated cache, while `--stage train` trains from a
completed cache. The defaults mirror the AVWAN foundation training settings;
the cache and output directories are unique to this experiment. They are
`data_cache/foundation_no_depth/` and `runs/foundation_no_depth/` beneath this
directory.

Before conversion, the entry point checks train/validation episode IDs and
checks the train source against the configured test manifests for scene and
episode overlap. It checks validation/test episode IDs as well. The test audit
reads only split metadata for these comparisons, not observations, actions,
rewards, labels, trajectories, or metrics. The audit report is recorded in
`dataset.json`; the cache's `sources.json` retains episode-relative source
paths. Do not point the training source at a validation or test split.

Training writes `config.json`, `dataset.json`, and `protocol.json`; it selects
`best.pt` by minimum validation combined action and angle loss. Test evaluation
is intentionally not part of this training entry point. Freeze the selected
checkpoint and evaluation protocol before evaluating a test split.
