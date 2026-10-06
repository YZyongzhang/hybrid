"""Resumable coordinator; all actual executions and intermediate results retained."""
import json
import time
from .stop_study import OUT,DATA,PYTHON,EXPERIMENTS,SPLITS,prepare
from .run_gamma_study import write,command_run,now,sha
from .evaluate_stop_study import evaluate,checkpoint_for


def select_evaluation_checkpoints():
    bests={name:json.loads((OUT/name/'training/best.json').read_text()) for name,mode,_ in EXPERIMENTS if mode!='baseline'}
    all_unchanged=all(r['epoch']==0 for r in bests.values())
    use_final=all_unchanged or (OUT/'protocol_amendment_01.json').exists()
    if all_unchanged and not (OUT/'protocol_amendment_01.json').exists():
        write(OUT/'protocol_amendment_01.json',dict(decided_at=now(),
            trigger='All four surrogate-validation-selected checkpoints are unchanged epoch 0.',
            reason='Evaluate the actual training intervention rather than repeat the same policy under four names.',
            rule='Use the fixed final epoch 40 for every trained arm; choose configuration by independent validation navigation SPL.',
            test_results_used=False,original_protocol_preserved=True,original_selected_epochs={n:r['epoch'] for n,r in bests.items()}))
    for name in bests:
        path=OUT/name/'training'/('latest.pt' if use_final else 'best_policy.pt')
        write(OUT/name/'evaluation_checkpoint.json',dict(checkpoint=str(path),sha256=sha(path),
            epoch=40 if use_final else bests[name]['epoch'],
            selection='fixed final epoch 40 under protocol_amendment_01' if use_final else 'original validation surrogate',
            original_surrogate_best_epoch=bests[name]['epoch']))


def main():
    prepare()
    while not (DATA/'collection_complete.json').exists():
        write(OUT/'status.json',dict(stage='waiting_for_collection',updated_at=now()))
        time.sleep(20)
    write(OUT/'status.json',dict(stage='feature_preparation',updated_at=now()))
    if not (OUT/'data_audit.json').exists():
        command_run([PYTHON,'-m','experiment.avwan.train_stop_rl','--prepare-features'],OUT,'prepare_features')
    for name,mode,penalty in EXPERIMENTS:
        if mode!='baseline':
            write(OUT/'status.json',dict(stage='training',experiment=name,updated_at=now()))
            command_run([PYTHON,'-m','experiment.avwan.train_stop_rl','--name',name],OUT,name+'_train')
    select_evaluation_checkpoints()
    # Complete independent validation for all trained candidates before tests.
    for split in SPLITS:
        for name,_,_ in EXPERIMENTS:
            write(OUT/'status.json',dict(stage='evaluation',experiment=name,split=split,updated_at=now()))
            print(json.dumps(evaluate(name,split)),flush=True)
        if split=='val_multiple_unheard':
            results=[json.loads((OUT/name/split/'results.json').read_text()) for name,_,_ in EXPERIMENTS]
            best=max(results,key=lambda row:row['spl'])
            selection_path=OUT/'navigation_validation_selection.json'
            if selection_path.exists():
                previous=json.loads(selection_path.read_text())
                assert previous['selected']==best['name'] and previous['validation_spl']==best['spl']
            else:
                write(selection_path,dict(selected=best['name'],validation_spl=best['spl'],
                    criterion='maximum independent unheard validation navigation SPL',decided_at=now(),test_scores_used=False))
    for name,_,_ in EXPERIMENTS:
        command_run([PYTHON,'-m','experiment.avwan.audit_offline_memory','--checkpoint',str(checkpoint_for(name)),
                     '--cache',str(DATA/'frozen_ratio_features'),'--output',str(OUT/name/'memory_audit.json')],
                    OUT/name,'memory_audit')
    write(OUT/'status.json',dict(stage='evaluations_complete',updated_at=now()))


if __name__=='__main__':main()
