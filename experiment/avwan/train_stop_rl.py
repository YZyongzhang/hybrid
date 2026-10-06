"""Matched-step SAC-CQL ablations with immutable negative STOP transitions."""
import argparse
import copy
import gzip
import json
import pickle
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from .stop_study import OUT,DATA,PREVIOUS,PARENT,FOUNDATION,BASE_CACHE,TRAIN_SEED,EXPERIMENTS
from .run_gamma_study import write,sha,now
from .offline.features import extract
from .offline.dataset import FeatureEpisodes,collate_episodes
from .offline.sac_cql import RecurrentSACCQL,masked_policy
from .offline.trainer import save_checkpoint


def annotated_episode(ep, failed_stop, penalty, expert_control=None):
    """Training-only reward transform; the loaded features and raw file stay intact."""
    bad=torch.as_tensor(failed_stop,dtype=torch.bool)
    allowed=~bad
    if expert_control is not None:
        expert=torch.as_tensor(expert_control,dtype=torch.bool)
        assert expert.shape==allowed.shape
        allowed=allowed & expert
    return dict(ep,bc_mask=allowed,reward=ep['reward'].clone()-penalty*bad.float())


def prepare_features():
    assert (DATA/'collection_complete.json').exists()
    source=DATA/'feature_source'
    for kind in ['policy','recovery']:
        for path in (DATA/kind/'episodes').glob('*/*.pkl.gz'):
            dest=source/'episodes'/path.parent.name/(kind+'_'+path.name)
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():
                # /data is mounted on a filesystem without symlink support.
                temporary=dest.with_suffix(dest.suffix+'.tmp')
                shutil.copyfile(path,temporary);temporary.replace(dest)
    write(OUT/'feature_source_storage.json',dict(method='immutable file copies',
        reason='/data filesystem rejects symlinks with ENOSYS',source=str(source)))
    state=torch.load(PARENT,map_location='cpu',weights_only=False)
    cfg=copy.deepcopy(state['config']);cfg['features'].update(source=str(source),checkpoint=str(FOUNDATION),
        cache=str(DATA/'frozen_ratio_features'),val_source=str(PREVIOUS/'validation_demonstrations'))
    write(DATA/'feature_config.json',cfg);torch.set_num_threads(4)
    manifest=extract(cfg,torch.device('cuda:0'))
    metadata=[]
    for entry in manifest['splits']['train']:
        path=source/entry['source']
        with gzip.open(path,'rb') as f:raw=pickle.load(f)
        ts=raw['transitions'];bad=[bool(t['false_stop']) for t in ts]
        for t,failed in zip(ts,bad):
            assert t['bc_allowed']==(not failed)
            if failed:assert t['action']==40 and t['terminated'] and t['discount']==0 and t['done']
            if t['correct_stop']:assert t['action']==40 and t['terminated'] and not failed
        original=Path(raw['metadata']['file']);raw_hash=sha(original)
        assert sha(path)==raw_hash
        metadata.append(dict(source=entry['source'],kind=raw['metadata']['kind'],false_stop=bad,
            raw_sha256=raw_hash,path=str(original),feature_source_copy=str(path),raw_rewards=[float(t['reward']) for t in ts],
            actual_actions=[int(t['action']) for t in ts],interventions=raw['metadata']['interventions']))
    write(DATA/'training_annotations.json',metadata)
    write(OUT/'data_audit.json',dict(passed=True,training_only=True,episodes=len(metadata),
        negative_stops=sum(sum(r['false_stop']) for r in metadata),terminal_bootstrap_zero=True,
        actual_actions_retained=True,raw_rewards_retained=True,annotations_sha256=sha(DATA/'training_annotations.json'),
        feature_manifest_sha256=sha(DATA/'frozen_ratio_features/manifest.json')))


def expert_annotations():
    path=DATA/'expert_bc_annotations.json'
    if path.exists():return json.loads(path.read_text())
    annotations=json.loads((DATA/'training_annotations.json').read_text());result={}
    for meta in annotations:
        if meta['kind']=='policy' and not any(meta['false_stop']):continue
        raw_path=Path(meta['path']);assert sha(raw_path)==meta['raw_sha256']
        with gzip.open(raw_path,'rb') as f:raw=pickle.load(f)
        flags=[bool(t['expert_control']) for t in raw['transitions']]
        assert all((t['teacher_action']==t['action']) if flag else t['teacher_action'] is None
                   for t,flag in zip(raw['transitions'],flags))
        assert len(flags)==len(meta['actual_actions'])
        result[meta['source']]=flags
    write(path,result)
    return result


def load_pools(penalty, expert_only=False):
    base_manifest=json.loads((BASE_CACHE/'manifest.json').read_text())
    manifest=json.loads((DATA/'frozen_ratio_features/manifest.json').read_text())
    assert base_manifest['identity']['checkpoint_sha256']==manifest['identity']['checkpoint_sha256']==sha(FOUNDATION)
    assert base_manifest['feature_dim']==manifest['feature_dim'] and base_manifest['action_count']==manifest['action_count']
    annotations=json.loads((DATA/'training_annotations.json').read_text())
    expert_flags=expert_annotations() if expert_only else {}
    base=FeatureEpisodes(BASE_CACHE,base_manifest,'train').episodes
    additional=FeatureEpisodes(DATA/'frozen_ratio_features',manifest,'train').episodes
    val=FeatureEpisodes(DATA/'frozen_ratio_features',manifest,'val')
    assert len(additional)==len(annotations)
    negatives=[];recoveries=[]
    for ep,meta in zip(additional,annotations):
        bad=torch.tensor(meta['false_stop'],dtype=torch.bool)
        if expert_only and (meta['kind']=='recovery' or bool(bad.any())):
            assert meta['source'] in expert_flags, 'Missing executed-expert BC provenance'
        assert torch.equal(ep['action'],torch.tensor(meta['actual_actions']))
        assert torch.allclose(ep['reward'],torch.tensor(meta['raw_rewards']))
        ep=annotated_episode(ep,bad,penalty,expert_flags.get(meta['source']) if expert_only else None)
        if meta['kind']=='policy' and bad.any():negatives.append(ep)
        elif meta['kind']=='recovery':recoveries.append(ep)
    assert negatives and recoveries
    return [base,negatives,recoveries],val,manifest


@torch.no_grad()
def validate(model,dataset,device):
    model.eval();all_nll=[];stop_pos=[];stop_neg=[];correct=0;count=0;tp=fp=fn=0
    for batch in DataLoader(dataset,batch_size=32,collate_fn=collate_episodes):
        b={k:v.to(device) for k,v in batch.items()};logits,_=model.actor(b['features'][:,:-1])
        probs,lp=masked_policy(logits,b['action_mask'][:,:-1]);valid=b['valid'];y=b['action']==40
        all_nll.extend((-lp.gather(-1,b['action'][...,None]).squeeze(-1))[valid].cpu().tolist())
        ps=probs[...,40].clamp(1e-7,1-1e-7)
        stop_pos.extend((-ps.log())[valid & y].cpu().tolist());stop_neg.extend((-torch.log1p(-ps))[valid & ~y].cpu().tolist())
        pred=probs.argmax(-1);correct+=int(((pred==b['action'])&valid).sum());count+=int(valid.sum())
        tp+=int(((pred==40)&y&valid).sum());fp+=int(((pred==40)&~y&valid).sum());fn+=int(((pred!=40)&y&valid).sum())
    nll=float(np.mean(all_nll));balanced=.5*(float(np.mean(stop_pos))+float(np.mean(stop_neg)))
    return dict(policy_nll=nll,balanced_stop_nll=balanced,selection_score=balanced+.1*nll,
        action_accuracy=correct/count,transitions=count,stop_true_positive=tp,stop_false_positive=fp,
        stop_false_negative=fn,stop_precision=tp/max(tp+fp,1),stop_recall=tp/max(tp+fn,1))


def train(name,mode,penalty):
    folder=OUT/name;folder.mkdir(exist_ok=True);output=folder/'training'
    if (output/'completed.json').exists():return
    random.seed(TRAIN_SEED);np.random.seed(TRAIN_SEED);torch.manual_seed(TRAIN_SEED);torch.set_num_threads(4)
    parent=torch.load(PARENT,map_location='cpu',weights_only=False);cfg=copy.deepcopy(parent['config'])
    cfg['seed']=TRAIN_SEED;cfg['training'].update(epochs=40,output=str(output),learning_rate=1e-4)
    cfg['features'].update(checkpoint=str(FOUNDATION),cache=str(DATA/'frozen_ratio_features'),
        source=str(DATA/'feature_source'),val_source=str(PREVIOUS/'validation_demonstrations'))
    assert cfg['training']['gamma']==.01 and cfg['training']['bc_weight']==5. and cfg['training']['terminal_mask']
    expert_only=mode=='recovery_expert_bc';sampling_mode='recovery' if expert_only else mode
    cfg['stop_pipeline']=dict(mode=mode,failed_stop_penalty=penalty,bc_masked=True,parent=str(PARENT),
        sampling={'original':[32,0,0],'negative':[24,8,0],'recovery':[16,8,8]}[sampling_mode],steps_per_epoch=186,
        effective_optimizer='Adam, learning_rate=1e-4, weight_decay=0 (same as parent SAC-CQL implementation)',
        checkpoint_selection='balanced STOP NLL + 0.1*action NLL on independent validation demonstrations')
    pools,val,manifest=load_pools(penalty,expert_only)
    if expert_only:
        cfg['stop_pipeline']['new_data_bc_scope']='only actually executed expert actions; original pool unchanged'
        cfg['stop_pipeline']['expert_annotation_sha256']=sha(DATA/'expert_bc_annotations.json')
    signature=copy.deepcopy(parent['signature']);signature['features']=copy.deepcopy(manifest['identity'])
    signature['stop_annotations_sha256']=sha(DATA/'training_annotations.json')
    if expert_only:signature['expert_bc_annotation_sha256']=cfg['stop_pipeline']['expert_annotation_sha256']
    device=torch.device('cuda:0');model=RecurrentSACCQL(signature['feature_dim'],signature['action_count'],cfg['training']).to(device)
    model.load_state_dict(parent['model']);optimizers=model.make_optimizers();output.mkdir(exist_ok=True)
    write(output/'config.json',cfg);(folder/'config.yaml').write_text(yaml.safe_dump(cfg))
    write(output/'dataset.json',dict(base_cache=str(BASE_CACHE),new_cache=str(DATA/'frozen_ratio_features'),
        pool_sizes=[len(p) for p in pools],sampling=cfg['stop_pipeline']['sampling'],annotations_sha256=signature['stop_annotations_sha256']))
    epoch_start=0;best=float('inf')
    def checkpoint(epoch,metrics,path):
        save_checkpoint(path,dict(epoch=epoch,model=model.state_dict(),signature=signature,config=cfg,
            validation=metrics,best_score=best,optimizers={k:v.state_dict() for k,v in optimizers.items()},
            initialization_checkpoint_sha256=sha(PARENT),selection=cfg['stop_pipeline']['checkpoint_selection']))
    if (output/'latest.pt').exists():
        state=torch.load(output/'latest.pt',map_location=device,weights_only=False);assert state['config']==cfg
        model.load_state_dict(state['model']);epoch_start=state['epoch'];best=state['best_score']
        for k,opt in optimizers.items():opt.load_state_dict(state['optimizers'][k])
    else:
        initial=validate(model,val,device);best=initial['selection_score'];write(output/'initial_validation.json',initial)
        checkpoint(0,initial,output/'best_policy.pt');write(output/'best.json',dict(epoch=0,**initial))
    write(folder/'status.json',dict(status='training',mode=mode,started_at=now()))
    for epoch in range(epoch_start+1,41):
        model.train();rng=np.random.default_rng(TRAIN_SEED+epoch);totals={};valid_count=bc_count=0;start=time.monotonic()
        for _ in range(186):
            episodes=[]
            for pool,n in zip(pools,cfg['stop_pipeline']['sampling']):
                episodes.extend(pool[int(i)] for i in rng.integers(len(pool),size=n))
            batch={k:v.to(device) for k,v in collate_episodes(episodes).items()}
            metrics=model.update(batch,optimizers)
            n=int(batch['valid'].sum());valid_count+=n;bc_count+=int((batch['valid']&batch['bc_mask']).sum())
            for k,v in metrics.items():totals[k]=totals.get(k,0.)+v*n
        training={k:v/valid_count for k,v in totals.items()};training.update(transitions=valid_count,bc_allowed=bc_count,
            bc_excluded=valid_count-bc_count,seconds=time.monotonic()-start)
        metrics=validate(model,val,device)
        if metrics['selection_score']<best:
            best=metrics['selection_score'];checkpoint(epoch,metrics,output/'best_policy.pt');write(output/'best.json',dict(epoch=epoch,**metrics))
        checkpoint(epoch,metrics,output/'latest.pt')
        record=dict(epoch=epoch,train=training,val=metrics)
        with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    write(output/'completed.json',dict(epochs=40,best_score=best,completed_at=now()))
    write(folder/'status.json',dict(status='trained',completed_at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare-features',action='store_true');p.add_argument('--name');args=p.parse_args()
    if args.prepare_features:prepare_features()
    else:
        spec=next(e for e in EXPERIMENTS if e[0]==args.name);train(*spec)
