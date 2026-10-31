# AVNav RGB + Audio real experiment

The default model is yesterday's validation-selected `corrective_epoch_015`
from `result/2026-10-28-avnav-codex_attemp_rgb`. Use these checkpoints together:

- `checkpoints/foundation_rgb_best.pth` — source `checkpoints/foundation_rgb/best.pth`
- `checkpoints/corrective_epoch_015_best_policy.pt` — source `checkpoints/corrective_epoch_015/best_policy.pt`

`model_manifest.json` records SHA256 hashes, source files, the training config,
selection rule, and the previous RGB-D model's backup directory. The frozen
selection and data/split audits are copied into `provenance/`. Selection used
validation only: 500 episodes per split, SPL 0.322087 (heard) and 0.164163
(unheard), with zero errors. These are simulator validation results; real
hardware performance has not been measured by this migration.

## Run

From the project root, in the `ss` Python environment:

```bash
python experiment/real_experiment/inference_server.py --device cuda --port 8080
```

This command loads the new paired checkpoints by default. `/health` identifies
`corrective_epoch_015` and the audio/RGB modalities. The existing robot client
can continue sending its legacy depth fields; they are optional and are never
passed to the model. Requests without depth fields are also accepted. The
request route, session sequencing, action IDs, and audio/RGB archives are
unchanged.

For Python inference:

```python
from experiment.real_experiment import AVNavRealPolicy

policy = AVNavRealPolicy(device="cuda")  # or "cpu"
policy.reset()
action_id = policy.act_observation(observation)
# Alternatively: action_id = policy.act(spectrogram, rgb)
```

## Inputs and protocol

- `rgb`: HWC RGB `[128, 128, 3]`, uint8 0–255 or float 0–1.
- `spectrogram`: binaural `[65, 26, 2]`. `act_observation` also accepts a
  single-batch array or the simulator's `(spectrogram, metadata)` tuple.
- Vision concatenates previous and current RGB along the channel axis into
  `[128, 128, 6]`. The first previous frame is zero.
- The SAC-LSTM receives the latest five 256-dimensional embeddings, left-padded
  with zeros. Call `reset()` at each new episode; HTTP sequence 0 resets history.
- Actions are the actor's deterministic argmax with original AVNav IDs. There
  are no STOP thresholds, collision recovery rules, or action overrides.

The HTTP frontend resizes RGB to 128×128 and divides by 255. Its existing audio
frontend takes the latest one second of 48 kHz stereo int16 PCM, left-pads short
clips, resamples to 16 kHz, and produces the SoundSpaces-compatible log1p STFT
block-mean spectrogram. The foundation and offline model definitions are exact
copies of the frozen RGB+audio experiment source and load both state dicts
strictly. Runtime dependencies: PyTorch, NumPy, SciPy, and OpenCV.

## Inference records

The server saves RGB PNGs, 48 kHz stereo PCM WAVs, and unchanged action IDs to
`inference_records/`, or the path supplied with `--record-dir`. `requests.jsonl`
and `commands.jsonl` reference session, sequence, request, and action IDs.
Every 30 steps, raw audio clips are concatenated under `audio_batches/`, with
source steps recorded in `audio_batches.jsonl`. Batching is archival only;
inference uses each step's current audio.
