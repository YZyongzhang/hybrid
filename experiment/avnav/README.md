# AVNav foundation, offline RL, and online evaluation

## Independent AVNav run (2026-10-17)

The staged audio → action → offline CQL run and its 2,000-episode full test are
archived under `result_avnav/2026-10-17-avnav-staged-cql/`. Its model layers,
SoundSpaces task config, test episode loader, and online evaluator are all
implemented in `experiment/avnav`; the AVNav evaluator does not import AV-WaN
evaluation code.

Run the same protocol from the project root with:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/getuanhui/anaconda3/envs/ss/bin/python run_avnav.py \
  --stage staged_all \
  --staged-foundation-config experiment/avnav/config/foundation_avnav_20261017.yaml \
  --offline-config experiment/avnav/config/offline_rl_avnav_20261017.yaml
```

The multiple-source pickle episodes live at `/data/Getuanhui/avnav-hybrid-mutiple`.
They were converted into episode-preserving PT shards at `pt/train`, `pt/val`,
and `pt/val_unheard`. The checked-in defaults point at those prepared shards.

The legacy single-stage foundation -> offline CQL -> online workflow can still be
run with:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/getuanhui/anaconda3/envs/ss/bin/python run_avnav.py \
  --config experiment/avnav/config/experiment.yaml \
  --offline-config experiment/avnav/config/offline_rl.yaml --stage all
```

To continue from an already-trained foundation, use `--stage offline_rl` and
`--stage evaluate_online`. These stages are also available as separate module
commands below.

Run from the repository root with the SoundSpaces `ss` Python environment, which
provides CUDA, Habitat, and SoundSpaces:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/getuanhui/anaconda3/envs/ss/bin/python run_avnav.py \
  --config experiment/avnav/config/experiment.yaml --stage train

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/getuanhui/anaconda3/envs/ss/bin/python -m experiment.avnav.offline_rl \
  --config experiment/avnav/config/offline_rl.yaml

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/getuanhui/anaconda3/envs/ss/bin/python -m experiment.avnav.evaluate_online \
  --config experiment/avnav/config/offline_rl.yaml
```

The legacy foundation trainer starts from random weights, visits all training
transitions, and selects `best.pth` on the heard validation split. The model uses
per-observation audio normalization so feature extraction and online inference use
the same inputs.

Offline RL reads the trusted episode pickles' recorded reward, done, and action
sequences. It extracts frozen 256-dimensional features with the new foundation,
trains a discrete SAC-CQL policy with a behavior-cloning term, and selects the
policy by held-out heard action NLL. The unheard split is reserved for later
evaluation. The earlier single-stage result is preserved under
`result/2026-10-13-avnav-offline-rl/`.

Online evaluation runs the learned four-action policy in Habitat/SoundSpaces on
`test_multiple` and `test_multiple_unheard`. It must execute the actor's raw
argmax action at every step. Do not add collision-triggered turns/recovery, depth
turn selection, or STOP-confidence overrides. Historic rule-modified runs are
non-compliant diagnostics and must not be cited as policy performance. Official
full tests use 1,000 episodes per split and save per-episode rollouts and aggregate
metrics under their result directory.

The workflow uses the AVNAV network in `experiment/avnav/model`, not the AV-WaN
81-waypoint model. Relative output paths are resolved from the project root.

## Staged AVNAV foundation

`config/foundation_staged.yaml` and `train_staged_foundation.py` implement a new
two-stage foundation run. Audio training combines 8-way direction prediction with
the AV-WaN gradient reversal sound-domain objective. Half of the heard training
sound classes are assigned to a deterministic pseudo-unheard domain. The second
stage loads the best audio checkpoint, freezes the complete audio encoder, and
trains the visual, fusion, and action layers. Validation selects checkpoints by
the worst loss across the independent heard and unheard validation sets.

The 2026-10-13 follow-up lives under
`result/2026-10-13-avnav-staged-grl-gamma/`. Offline CQL candidates use gamma
0.50, 0.65, and 0.80. IQL was also tried at gamma 0.65. CQL with gamma 0.80 and
BC weight 2.0 was first evaluated online; after poor results, BC weight 5.0 was
selected by held-out action NLL and evaluated in a second full online run. These
are learned policies with no hand-coded action rules. See `EXPERIMENT.md` and
`summary.json` for validation and online results, and the output directories for
configs, checkpoints, metrics, and 1,000-episode-per-split JSONL files. Online
episodes are appended as they finish so an interrupted run can resume.

Start staged foundation training with:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/getuanhui/anaconda3/envs/ss/bin/python run_avnav.py \
  --staged-foundation-config experiment/avnav/config/foundation_staged.yaml \
  --stage staged_foundation
```
