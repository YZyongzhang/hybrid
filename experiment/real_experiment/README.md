# AVNav RGB-D + Audio Candidate

This directory contains one paired foundation/offline-RL checkpoint candidate
and the model code needed to run its policy inference.

## Checkpoints

- `checkpoints/foundation_epoch_125.pth`
- `checkpoints/saccql_epoch_80.pth`

Use the two files together. They were evaluated as one stack in
`val_multiple_unheard` with 500 episodes, deterministic SAC actions, and a
maximum of 200 steps per episode. The recorded results were average reward
7.6199, SPL 0.2670, success rate 0.3660, and average 132.21 steps. The archived
log is `../logs/eval_OfflineRL_v1_5_20260604_143241.log`.

This is a validation result, not a final test-split or real-hardware result.
The historical record does not document a formal checkpoint sweep/selection
rule, so treat this as a promising integration candidate rather than a
benchmark-certified deployment model.

## Inputs and action handling

`AVNavRealPolicy.act_observation(obs)` consumes the AVNav observation fields:

- `rgb`: HWC RGB image, uint8 `[H, W, 3]` or float `[H, W, 3]`
- `depth`: `[H, W]` or HWC `[H, W, 1]`
- `spectrogram`: same spectrogram representation as the simulator; the wrapper
  selects `spectrogram[0]`, matching the archived evaluator

The policy concatenates the previous and current RGB/depth frames along image
width, encodes the result with audio, and keeps the latest five embeddings for
the SAC-LSTM. Call `policy.reset()` at the beginning of each episode. The
returned integer is the original policy action ID; the wrapper applies no
recovery or action-override rule. Keep the simulator's original action-ID
mapping when connecting it to the robot.

The foundation model API has optional pose and ego-map arguments. The archived
v1.5 evaluator does not pass either argument, and this wrapper follows that
same path.

## Example

```python
from real_experiment import AVNavRealPolicy

policy = AVNavRealPolicy(
    "real_experiment/checkpoints/foundation_epoch_125.pth",
    "real_experiment/checkpoints/saccql_epoch_80.pth",
    device="cuda",  # or "cpu"
)
policy.reset()
action_id = policy.act_observation(observation)
```

The model definitions are in `models/foundation.py` and
`models/offline_sac_v1_5.py`. They are copied from the matching `finnal_exp`
network implementations. Runtime dependencies are PyTorch and NumPy, plus the
sensor frontend that produces the simulator-compatible spectrogram.

## Inference records

The HTTP inference server stores each successfully evaluated observation under
`real_experiment/inference_records/`. RGB images are saved as PNG files in
`images/`, and the received stereo audio is saved as 48 kHz, 16-bit PCM WAV
files in `audio/`. Both filenames end with `_a<action_id>` (for example,
`_a3.png` and `_a3.wav`) using the policy's returned action ID unchanged.
`requests.jsonl` and `commands.jsonl` record the same action ID and map each
input to its session, sequence, and request IDs.

Every 30 steps, the server also concatenates those 30 raw PCM clips into a WAV
under `audio_batches/`. `audio_batches.jsonl` records the session, original
step sequence numbers, per-step WAV paths, action IDs, and sample counts for
each batch.
The individual WAV files remain available under `audio/`. This aggregation is
for debugging only: inference still uses the current step's audio and the
policy's inputs and outputs are unchanged.

Use `--record-dir PATH` with `inference_server.py` to choose a different
archive location. The default is `real_experiment/inference_records/`.
