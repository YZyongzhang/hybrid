#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/getuanhui/project/hybrid_offline_RL
PY=/home/getuanhui/anaconda3/envs/ss/bin/python
DATA=/data2/getuanhui/avnav-codex
OUT="$PROJECT/result_avnav/2026-10-27-avnav-recollected-collision-reward-scaling"
CFG="$PROJECT/experiment/avnav/config/recollected_collision_reward_scaling_20261027.yaml"
REPORT="$DATA/collection_report.json"

mkdir -p "$OUT"
if ! mkdir "$OUT/.pipeline.lock" 2>/dev/null; then
  echo "Pipeline lock exists at $OUT/.pipeline.lock; refusing a duplicate run."
  exit 2
fi
exec >> "$OUT/runner.log" 2>&1
cd "$PROJECT"

collection_alive() {
  "$PY" - "$DATA" <<'PY'
import pathlib, sys
needle = b"experiment.avnav.collect_expert_diverse_collision_data"
root = sys.argv[1].encode()
for proc in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
    try:
        cmd = proc.read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if needle in cmd and root in cmd:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

echo "[$(date -Is)] Waiting for train-only data collection at $DATA."
while [[ ! -s "$REPORT" ]]; do
  if ! collection_alive; then
    echo "[$(date -Is)] Collector exited before writing $REPORT."
    exit 1
  fi
  sleep 60
done

"$PY" - "$REPORT" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
expected = {'greedy': 5700, 'deverse': 5700, 'collision': 2850}
if report.get('status') != 'completed':
    raise SystemExit(f"Collection is incomplete: status={report.get('status')}")
for item in report.get('category_summaries', []):
    name = item['category']
    if item['target_tasks'] != expected[name] or item['saved_episodes'] != expected[name]:
        raise SystemExit(f"{name}: expected {expected[name]} saved episodes, got {item}")
    if item.get('errors') or item.get('shortfalls'):
        raise SystemExit(f"{name}: collection errors or shortfalls: {item}")
print('Collection counts and status verified.')
PY

echo "[$(date -Is)] Auditing source provenance against validation and test manifests."
"$PY" -m experiment.avnav.audit_data_splits \
  --config "$CFG" \
  --output-dir "$OUT/audits" \
  --additional-provenance \
    "$DATA/greedy/manifest.jsonl" \
    "$DATA/deverse/manifest.jsonl" \
    "$DATA/collision/manifest.jsonl"

echo "[$(date -Is)] Freezing the recollected-data experiment protocol."
"$PY" - "$PROJECT" "$DATA" "$OUT" "$CFG" <<'PY'
import hashlib, json, pathlib, sys, yaml
project, data, out, cfg = map(pathlib.Path, sys.argv[1:])
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()
config = yaml.safe_load(cfg.read_text())
report = data / 'collection_report.json'
source = json.loads(report.read_text())
summary = {
    'status': 'frozen_before_validation_selection',
    'experiment': 'AVNav CQL-LSTM collision reward shaping on recollected train-only data',
    'collection_report': str(report.resolve()),
    'collection_report_sha256': digest(report),
    'collection_protocol': str((data / 'collection_protocol.json').resolve()),
    'collection_protocol_sha256': digest(data / 'collection_protocol.json'),
    'collision_repair_protocol': str((data / 'collision_repair_protocol.json').resolve()),
    'collision_repair_protocol_sha256': digest(data / 'collision_repair_protocol.json'),
    'selection_manifest': str((data / 'selection_manifest.jsonl').resolve()),
    'selection_manifest_sha256': digest(data / 'selection_manifest.jsonl'),
    'category_summaries': source['category_summaries'],
    'training_config': str(cfg.resolve()),
    'training_config_sha256': digest(cfg),
    'data_source_audit': str((out / 'audits/data_source_audit.json').resolve()),
    'data_source_audit_sha256': digest(out / 'audits/data_source_audit.json'),
    'training_sources': [str(pathlib.Path(item).resolve())
                         for item in config['DATA']['TRAIN_EPISODES']],
    'feature_cache_root': config['OFFLINE_RL']['SEQUENCE_CACHE_ROOT'],
    'training_output_root': config['OFFLINE_RL']['OUTPUT'],
    'storage_amendment': json.loads(
        (out / 'experiment_protocol_amendment.json').read_text()),
    'reward_shaping_in_collected_data': json.loads(
        (data / 'collection_protocol.json').read_text())['reward_shaping'],
    'additional_train_time_collision_penalties':
        config['SAFETY']['COLLISION_REWARD_PENALTIES'],
    'training': {
        'seed': config['SEED'], 'algorithm': config['OFFLINE_RL']['ALGORITHM'],
        'epochs': config['OFFLINE_RL']['EPOCHS'],
        'updates_per_epoch': config['OFFLINE_RL']['UPDATES_PER_EPOCH'],
        'batch_size': config['OFFLINE_RL']['BATCH_SIZE'],
        'foundation_checkpoint': str(pathlib.Path(
            config['FOUNDATION']['CHECKPOINT']).resolve()),
        'collision_predictor_checkpoint': str(pathlib.Path(
            config['OFFLINE_RL']['COLLISION_PREDICTOR_CHECKPOINT']).resolve()),
    },
    'validation_selection': {
        'splits': ['val_multiple', 'val_multiple_unheard'],
        'screen_episodes_per_split': config['SAFETY']['SCREEN_EPISODES_PER_SPLIT'],
        'confirmation_episodes_per_split': config['SAFETY']['CONFIRMATION_EPISODES_PER_SPLIT'],
        'score': config['SAFETY']['VALIDATION_SCORE'],
        'action_handling': 'raw policy actor argmax; no action override or recovery',
    },
    'test': {
        'splits': ['test_multiple', 'test_multiple_unheard'],
        'episodes_per_split': config['SAFETY']['TEST_EPISODES_PER_SPLIT'],
        'starts_after_validation_checkpoint_freeze': True,
        'test_data_used_for_training_or_selection': False,
    },
}
path = out / 'experiment_protocol.json'
if path.exists() and json.loads(path.read_text()) != summary:
    raise SystemExit(f'Frozen protocol mismatch: {path}')
path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
PY

echo "[$(date -Is)] Building frozen train/validation feature cache."
FEATURE_BUILD="$OUT/feature_build_summary.json"
"$PY" -m experiment.avnav.prepare_safety_lstm_features \
  --config "$CFG" --output-root "$DATA/rl_experiment/features_safety" \
  --sequence-length 5 --device cuda:0 > "$FEATURE_BUILD"
CACHE_ROOT=$("$PY" - "$FEATURE_BUILD" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))['cache_root'])
PY
)

echo "[$(date -Is)] Training predeclared collision reward penalty grid: ${COLLISION_REWARD_PENALTIES:-0.0 0.1 0.25 0.5}."
for PENALTY in 0.0 0.1 0.25 0.5; do
  TOKEN=${PENALTY/./p}
  CANDIDATE_OUT="$DATA/rl_experiment/offline/penalty_$TOKEN"
  if [[ -s "$CANDIDATE_OUT/training_summary.json" ]]; then
    echo "Reusing completed training candidate penalty_$TOKEN."
    continue
  fi
  if [[ -d "$CANDIDATE_OUT" ]] && find "$CANDIDATE_OUT" -mindepth 1 -print -quit | rg -q .; then
    echo "Incomplete candidate output exists: $CANDIDATE_OUT"
    exit 1
  fi
  "$PY" -m experiment.avnav.train_safety_cql_lstm \
    --config "$CFG" --feature-cache "$CACHE_ROOT" \
    --output "$CANDIDATE_OUT" --collision-reward-penalty "$PENALTY"
done

echo "[$(date -Is)] Freezing validation candidate manifest."
"$PY" - "$CFG" "$OUT" "$DATA" <<'PY'
import hashlib, json, pathlib, sys, yaml
cfg, out, data = map(pathlib.Path, sys.argv[1:])
root = data / 'rl_experiment/offline'
payload = {
    'status': 'candidate_grid_frozen_before_validation',
    'objective': 'recollected train-only CQL-LSTM collision reward scaling',
    'training_config': str(cfg.resolve()),
    'training_seed': yaml.safe_load(cfg.read_text())['SEED'],
    'penalty_grid': [0.0, 0.1, 0.25, 0.5],
    'screened_epochs': [6, 12, 18, 24],
    'validation_splits': ['val_multiple', 'val_multiple_unheard'],
    'test_splits': ['test_multiple', 'test_multiple_unheard'],
    'test_data_used_before_freeze': False,
    'action_handling': 'raw policy actor argmax; no action override or recovery',
    'candidates': [],
}
for penalty in payload['penalty_grid']:
    token = str(penalty).replace('.', 'p')
    variant = f'penalty_{token}'
    for epoch in payload['screened_epochs']:
        checkpoint = root / variant / f'epoch_{epoch:03d}.pt'
        if not checkpoint.is_file():
            raise SystemExit(f'Missing candidate checkpoint: {checkpoint}')
        payload['candidates'].append({
            'name': f'collision_reward_{variant}_epoch_{epoch:03d}',
            'variant': variant, 'collision_reward_penalty': penalty,
            'epoch': epoch, 'config': str(cfg.resolve()),
            'checkpoint': str(checkpoint.resolve()),
        })
path = out / 'validation_candidates.json'
if path.exists() and json.loads(path.read_text()) != payload:
    raise SystemExit(f'Candidate manifest mismatch: {path}')
path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
PY

echo "[$(date -Is)] Selecting checkpoints with validation only."
"$PY" -m experiment.avnav.select_safety_cql_checkpoints \
  --candidates "$OUT/validation_candidates.json" \
  --output "$OUT/validation" \
  --screen-episodes 30 --confirmation-episodes 500 --top-k 4 \
  --workers 4 --collision-rate-coefficient 0.1

echo "[$(date -Is)] Running complete test on the frozen validation winner."
"$PY" -m experiment.avnav.run_collision_reward_full_test \
  --selection "$OUT/validation/validation_selection.json" \
  --output "$OUT/test" --episodes 1000 --workers 4

echo "[$(date -Is)] Recollected-data training, validation selection, and full test completed."
