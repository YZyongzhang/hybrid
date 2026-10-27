#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/getuanhui/project/hybrid_offline_RL
PY=/home/getuanhui/anaconda3/envs/ss/bin/python
OUT="$PROJECT/result_avnav/2026-10-19-avnav-generalization/iteration8"
CFG="$PROJECT/experiment/avnav/config/offline_rl_round8_base4_successonly_20261020.yaml"
COLLECTIONS=(
  /data/Getuanhui/avnav_diverse_round8a_20261020
  /data/Getuanhui/avnav_diverse_round8b_20261020
  /data/Getuanhui/avnav_diverse_round8c_20261020
  /data/Getuanhui/avnav_diverse_round8d_20261020
)
mkdir -p "$OUT/offline"
exec >> "$OUT/offline/pipeline.log" 2>&1

printf '[%s] Waiting for four round8 collectors to finish.\n' "$(date -Is)"
while true; do
  STATE=$("$PY" - "${COLLECTIONS[@]}" <<'PY'
import json, pathlib, sys
states=[]
for name in sys.argv[1:]:
    p=pathlib.Path(name)/'manifest.json'
    if not p.is_file():
        states.append('waiting')
        continue
    d=json.loads(p.read_text())
    if int(d.get('errors',0)):
        print('error'); raise SystemExit
    if int(d.get('completed',0)) != int(d.get('planned',-1)):
        states.append('waiting')
    else:
        states.append('done')
print('ready' if all(x=='done' for x in states) else 'waiting')
PY
  )
  if [[ "$STATE" == error ]]; then
    printf '[%s] A round8 collector reported errors; stopping.\n' "$(date -Is)"
    exit 1
  fi
  [[ "$STATE" == ready ]] && break
  sleep 60
done

printf '[%s] Preparing selected successful round8 trajectories.\n' "$(date -Is)"
"$PY" -m experiment.avnav.prepare_selected_success_sources \
  --collections "${COLLECTIONS[@]}" \
  --audit "$OUT/data/round8_selected_source_audit.json"

printf '[%s] Training IQL gamma 0.80 and 0.95 candidates.\n' "$(date -Is)"
"$PY" -m experiment.avnav.sweep_offline_rl \
  --config "$CFG" \
  --output-root "$OUT/offline/candidates" \
  --epochs 24 --updates-per-epoch 250 \
  --candidate-names iql_g0.80_beta3.0 iql_g0.95_beta3.0

PIDS=()
for NAME in iql_g0.80_beta3.0 iql_g0.95_beta3.0; do
  CAND="$OUT/offline/candidates/$NAME"
  for CHECKPOINT_NAME in best_policy epoch_021; do
    CHECKPOINT="$CAND/$CHECKPOINT_NAME.pt"
    EVAL_OUT="$CAND/validation_pinned_20261021_${CHECKPOINT_NAME}_raw_policy"
    mkdir -p "$EVAL_OUT"
    printf '[%s] Validating %s (%s).\n' "$(date -Is)" "$NAME" "$CHECKPOINT_NAME"
    "$PY" -m experiment.avnav.evaluate_online \
      --config "$CAND/candidate_config.yaml" \
      --checkpoint "$CHECKPOINT" --output "$EVAL_OUT" --episodes 100 \
      --splits val_multiple_unheard \
      > "$EVAL_OUT/evaluation.log" 2>&1 &
    PIDS+=("$!")
  done
done
BASELINE_OUT="$OUT/diagnostics/epoch21_raw_policy_pinned20261021_100"
mkdir -p "$BASELINE_OUT"
printf '[%s] Evaluating the iteration7 control on the current validation episode set.\n' "$(date -Is)"
"$PY" -m experiment.avnav.evaluate_online \
  --config "$PROJECT/result_avnav/2026-10-19-avnav-generalization/iteration6/offline/base4_successonly_iql_g0.80_beta3.0/config.yaml" \
  --checkpoint "$PROJECT/result_avnav/2026-10-19-avnav-generalization/iteration6/offline/base4_successonly_iql_g0.80_beta3.0/epoch_021.pt" \
  --output "$BASELINE_OUT" --episodes 100 --splits val_multiple_unheard \
  > "$BASELINE_OUT/evaluation.log" 2>&1 &
PIDS+=("$!")
FAILED=0
for PID in "${PIDS[@]}"; do
  if wait "$PID"; then
    printf '[%s] Validation worker %s completed.\n' "$(date -Is)" "$PID"
  else
    printf '[%s] Validation worker %s failed.\n' "$(date -Is)" "$PID"
    FAILED=1
  fi
done
(( FAILED == 0 )) || exit 1

SELECTION="$OUT/validation_selection_pinned_20261021.json"
"$PY" - "$PROJECT" "$OUT" "$SELECTION" <<'PY'
import json, pathlib, sys
project, out, selection_path = map(pathlib.Path, sys.argv[1:])
names=('iql_g0.80_beta3.0','iql_g0.95_beta3.0')
prior_path=out/'diagnostics/epoch21_raw_policy_pinned20261021_100/val_multiple_unheard.json'
prior=json.loads(prior_path.read_text())
rows=[]
for name in names:
    for checkpoint_name in ('best_policy','epoch_021'):
        path=out/'offline/candidates'/name/f'validation_pinned_20261021_{checkpoint_name}_raw_policy/val_multiple_unheard.json'
        metrics=json.loads(path.read_text())
        rows.append(dict(name=name, checkpoint=checkpoint_name,
                         checkpoint_path=str((out/'offline/candidates'/name/f'{checkpoint_name}.pt').resolve()),
                         validation=metrics))
eligible=[row for row in rows if int(row['validation']['errors'])==0]
best=max(eligible,key=lambda row:float(row['validation']['spl'])) if eligible else None
passed=best is not None and float(best['validation']['spl'])>float(prior['spl'])
record=dict(status='passed' if passed else 'held_out_validation_gate_not_improved',
            prior_best_validation=prior,candidates=rows,
            selected_candidate=best if passed else None,full_test_started=False)
selection_path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
PY

if ! rg -q '"status": "passed"' "$SELECTION"; then
  printf '[%s] Round8 candidates did not improve the unseen validation SPL.\n' "$(date -Is)"
  exit 3
fi

readarray -t BEST < <("$PY" - "$SELECTION" <<'PY'
import json, sys
d=json.loads(open(sys.argv[1]).read())
r=d['selected_candidate']
print(r['name'])
print(r['checkpoint_path'])
PY
)
BEST_NAME="${BEST[0]}"
BEST_CHECKPOINT="${BEST[1]}"
BEST_CFG="$OUT/offline/candidates/$BEST_NAME/candidate_config.yaml"
RUN="$OUT/full_test_${BEST_NAME}_epoch_policy_raw_parallel"
mkdir -p "$RUN/shards"
PIDS=()
for RANGE in '0 250' '250 500' '500 750' '750 1000'; do
  read -r START END <<< "$RANGE"
  SHARD="$RUN/shards/unheard_${START}_${END}"
  mkdir -p "$SHARD"
  printf '[%s] Starting full-test shard [%s,%s).\n' "$(date -Is)" "$START" "$END"
  "$PY" -m experiment.avnav.evaluate_online \
    --config "$BEST_CFG" --checkpoint "$BEST_CHECKPOINT" --output "$SHARD" \
    --episodes 1000 --splits test_multiple_unheard --index-range "$START" "$END" \
    > "$SHARD/evaluation.log" 2>&1 &
  PIDS+=("$!")
done
FAILED=0
for PID in "${PIDS[@]}"; do
  if wait "$PID"; then
    printf '[%s] Full-test worker %s completed.\n' "$(date -Is)" "$PID"
  else
    printf '[%s] Full-test worker %s failed.\n' "$(date -Is)" "$PID"
    FAILED=1
  fi
done
(( FAILED == 0 )) || exit 1

mkdir -p "$RUN/final"
"$PY" -m experiment.avnav.aggregate_online_shards \
  --output "$RUN/final" --shards "$RUN/shards" \
  --episodes 1000 --splits test_multiple_unheard
"$PY" - "$PROJECT" "$OUT" "$RUN" "$SELECTION" <<'PY'
import json, pathlib, sys, yaml
project, out, run, selection_path=map(pathlib.Path,sys.argv[1:])
prior=json.loads((project/'result_avnav/2026-10-19-avnav-generalization/iteration7/full_test_summary.json').read_text())
selection=json.loads(selection_path.read_text())
report=json.loads((run/'final/report.json').read_text())
full=next(row for row in report['splits'] if row['split']=='test_multiple_unheard')
best=selection['selected_candidate']
candidate_config=yaml.safe_load((out/'offline/candidates'/best['name']/'candidate_config.yaml').read_text())
training_sources=candidate_config['DATA']['TRAIN_EPISODES']
attempt=dict(foundation='iteration4/foundation/best.pth',
             checkpoint=best['checkpoint_path'],
             training_data='Configured TRAIN_EPISODES sources (heard_train_multiple, hard-longrange, all-scenes, and rounds2_to8 collections)',
             training_data_sources=training_sources,
             validation=best['validation'],
             action_handling='raw actor argmax',
             full_test={'test_multiple_unheard':full},
             report=str((run/'final/report.json').resolve()),
             shards=str((run/'shards').resolve()))
best_spl=max(float(prior['best_unheard_spl']),float(full['spl']))
summary=dict(status='target_met' if best_spl>0.1 else 'completed_below_target',
             test_multiple_unheard_spl_target=0.1,
             prior_best_attempt=prior.get('epoch21_left2_attempt',prior.get('iteration7_control_attempt')),
             validation_selection=selection,iteration8_attempt=attempt,
             best_unheard_spl=best_spl,spl_target_met=best_spl>0.1)
(out/'full_test_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
PY
printf '[%s] Round8 training, selection, and full test completed.\n' "$(date -Is)"
