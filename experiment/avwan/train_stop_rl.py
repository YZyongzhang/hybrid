"""Matched-step SAC-CQL ablations with immutable negative STOP transitions."""
import argparse
import copy
import gzip
import json
import math
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
from .offline.features import extract,load_foundation
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


def load_pools(penalty, expert_only=False, base_cache=None, additional_cache=None,
               validation_cache=None, foundation_checkpoint=None):
    base_cache=Path(base_cache) if base_cache is not None else BASE_CACHE
    additional_cache=Path(additional_cache) if additional_cache is not None else DATA/'frozen_ratio_features'
    validation_cache=Path(validation_cache) if validation_cache is not None else additional_cache
    foundation_checkpoint=Path(foundation_checkpoint) if foundation_checkpoint is not None else FOUNDATION
    base_manifest=json.loads((base_cache/'manifest.json').read_text())
    manifest=json.loads((additional_cache/'manifest.json').read_text())
    val_manifest=json.loads((validation_cache/'manifest.json').read_text())
    assert (base_manifest['identity']['checkpoint_sha256']==manifest['identity']['checkpoint_sha256']
            ==val_manifest['identity']['checkpoint_sha256']==sha(foundation_checkpoint))
    assert base_manifest['feature_dim']==manifest['feature_dim'] and base_manifest['action_count']==manifest['action_count']
    assert manifest['feature_dim']==val_manifest['feature_dim'] and manifest['action_count']==val_manifest['action_count']
    annotations=json.loads((DATA/'training_annotations.json').read_text())
    expert_flags=expert_annotations() if expert_only else {}
    base=FeatureEpisodes(base_cache,base_manifest,'train').episodes
    additional=FeatureEpisodes(additional_cache,manifest,'train').episodes
    val=FeatureEpisodes(validation_cache,val_manifest,'val' if 'val' in val_manifest['splits'] else 'train')
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


def train(name,mode,penalty,base_cache=None,steps_per_epoch=186,epochs=40,sampling=None,
          foundation_checkpoint=None,parent_checkpoint=None,additional_cache=None,validation_cache=None,
          training_overrides=None,device='cuda:0',cpu_threads=4):
    folder=OUT/name;folder.mkdir(exist_ok=True);output=folder/'training'
    if not isinstance(cpu_threads, int) or cpu_threads < 1:
        raise ValueError('cpu_threads must be a positive integer')
    custom_base_cache=base_cache is not None
    base_cache=Path(base_cache).resolve() if base_cache is not None else BASE_CACHE
    foundation_checkpoint=Path(foundation_checkpoint).resolve() if foundation_checkpoint is not None else FOUNDATION
    parent_checkpoint=Path(parent_checkpoint).resolve() if parent_checkpoint is not None else PARENT
    additional_cache=Path(additional_cache).resolve() if additional_cache is not None else DATA/'frozen_ratio_features'
    validation_cache=Path(validation_cache).resolve() if validation_cache is not None else additional_cache
    random.seed(TRAIN_SEED);np.random.seed(TRAIN_SEED);torch.manual_seed(TRAIN_SEED);torch.set_num_threads(cpu_threads)
    parent=torch.load(parent_checkpoint,map_location='cpu',weights_only=False);cfg=copy.deepcopy(parent['config'])
    cfg['seed']=TRAIN_SEED;cfg['training'].update(epochs=epochs,output=str(output))
    overrides = dict(training_overrides or {})
    allowed = {'batch_size','gamma','bc_weight','actor_rl_weight','cql_weight','cql_temperature',
               'entropy_ratio','initial_alpha','tau','learning_rate','weight_decay','grad_clip',
               'hidden_dim','lstm_layers','terminal_mask','foundation_prior'}
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f'Unsupported SAC-CQL training overrides: {sorted(unknown)}')
    cfg['training'].update(overrides)
    if cfg['training']['hidden_dim'] != parent['config']['training']['hidden_dim'] or \
            cfg['training']['lstm_layers'] != parent['config']['training']['lstm_layers']:
        raise ValueError('hidden_dim and lstm_layers must match the parent actor architecture')
    for key in ('gamma', 'entropy_ratio'):
        if not 0 <= cfg['training'][key] <= 1:
            raise ValueError(f'{key} must be in [0, 1]')
    for key in ('learning_rate','initial_alpha','cql_temperature','grad_clip'):
        if cfg['training'][key] <= 0:
            raise ValueError(f'{key} must be positive')
    for key in ('bc_weight','actor_rl_weight','cql_weight','weight_decay'):
        if cfg['training'][key] < 0:
            raise ValueError(f'{key} must be nonnegative')
    if not 0 < cfg['training']['tau'] <= 1:
        raise ValueError('tau must be in (0, 1]')
    if not isinstance(cfg['training']['terminal_mask'], bool) or \
            not isinstance(cfg['training']['foundation_prior'], bool):
        raise ValueError('terminal_mask and foundation_prior must be booleans')
    cfg['features'].update(checkpoint=str(foundation_checkpoint),cache=str(additional_cache),
        source=str(DATA/'feature_source'),val_source=str(PREVIOUS/'validation_demonstrations'))
    expert_only=mode=='recovery_expert_bc';sampling_mode='recovery' if expert_only else mode
    default_sampling={'original':[32,0,0],'negative':[24,8,0],'recovery':[16,8,8]}[sampling_mode]
    batch_size = cfg['training']['batch_size']
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    sampling = list(sampling) if sampling is not None else default_sampling
    if len(sampling)!=3 or any(n<0 for n in sampling) or sum(sampling)!=batch_size:
        raise ValueError(f'Sampling must contain three nonnegative counts totaling batch_size={batch_size}')
    if penalty < 0:
        raise ValueError('failed STOP penalty must be nonnegative')
    cfg['stop_pipeline']=dict(mode=mode,failed_stop_penalty=penalty,bc_masked=True,parent=str(parent_checkpoint),
        sampling=sampling,steps_per_epoch=steps_per_epoch,
        effective_optimizer=f"Adam, learning_rate={cfg['training']['learning_rate']}, "
                             f"weight_decay={cfg['training'].get('weight_decay', 0.)}",
        checkpoint_selection='balanced STOP NLL + 0.1*action NLL on independent validation demonstrations')
    if custom_base_cache:
        cfg['stop_pipeline']['base_cache']=str(base_cache)
    if additional_cache != DATA/'frozen_ratio_features':
        cfg['stop_pipeline']['additional_cache']=str(additional_cache)
    if validation_cache != additional_cache:
        cfg['stop_pipeline']['validation_cache']=str(validation_cache)
    if foundation_checkpoint != FOUNDATION:
        cfg['stop_pipeline']['prior_replaced_with_foundation_sha256']=sha(foundation_checkpoint)
    pools,val,manifest=load_pools(penalty,expert_only,base_cache,additional_cache,
                                  validation_cache,foundation_checkpoint)
    if expert_only:
        cfg['stop_pipeline']['new_data_bc_scope']='only actually executed expert actions; original pool unchanged'
        cfg['stop_pipeline']['expert_annotation_sha256']=sha(DATA/'expert_bc_annotations.json')
    if (output/'completed.json').exists():
        saved=json.loads((output/'config.json').read_text())
        if saved != cfg:
            raise ValueError('This run is already complete with different settings; change RUN.NAME for a new run')
        print(f'Run already complete with matching configuration: {output}', flush=True)
        return
    signature=copy.deepcopy(parent['signature']);signature['features']=copy.deepcopy(manifest['identity'])
    signature['stop_annotations_sha256']=sha(DATA/'training_annotations.json')
    if expert_only:signature['expert_bc_annotation_sha256']=cfg['stop_pipeline']['expert_annotation_sha256']
    device=torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested for offline RL but is unavailable')
    model=RecurrentSACCQL(signature['feature_dim'],signature['action_count'],cfg['training']).to(device)
    incompatible = model.load_state_dict(parent['model'], strict=cfg['training']['foundation_prior'])
    if not cfg['training']['foundation_prior']:
        unexpected = [key for key in incompatible.unexpected_keys if not key.startswith('actor.prior.')]
        if incompatible.missing_keys or unexpected:
            raise ValueError(f'Parent actor weights do not match configured architecture: {incompatible}')
    if cfg['training']['foundation_prior']:
        fresh=load_foundation(foundation_checkpoint,device)
        if model.actor.prior is None:
            raise ValueError('foundation_prior is enabled but actor prior layers are missing')
        model.actor.prior.load_state_dict(fresh.final.action_head.state_dict())
        del fresh
    with torch.no_grad():
        model.log_alpha.fill_(math.log(cfg['training']['initial_alpha']))
    optimizers=model.make_optimizers();output.mkdir(exist_ok=True)
    write(output/'config.json',cfg);(folder/'config.yaml').write_text(yaml.safe_dump(cfg))
    write(output/'dataset.json',dict(base_cache=str(base_cache),new_cache=str(additional_cache),
        validation_cache=str(validation_cache),foundation=str(foundation_checkpoint),
        pool_sizes=[len(p) for p in pools],sampling=cfg['stop_pipeline']['sampling'],annotations_sha256=signature['stop_annotations_sha256']))
    epoch_start=0;best=float('inf')
    def checkpoint(epoch,metrics,path):
        save_checkpoint(path,dict(epoch=epoch,model=model.state_dict(),signature=signature,config=cfg,
            validation=metrics,best_score=best,optimizers={k:v.state_dict() for k,v in optimizers.items()},
            initialization_checkpoint_sha256=sha(parent_checkpoint),selection=cfg['stop_pipeline']['checkpoint_selection']))
    if (output/'latest.pt').exists():
        state=torch.load(output/'latest.pt',map_location=device,weights_only=False);assert state['config']==cfg
        model.load_state_dict(state['model']);epoch_start=state['epoch'];best=state['best_score']
        for k,opt in optimizers.items():opt.load_state_dict(state['optimizers'][k])
    else:
        initial=validate(model,val,device);best=initial['selection_score'];write(output/'initial_validation.json',initial)
        checkpoint(0,initial,output/'best_policy.pt');write(output/'best.json',dict(epoch=0,**initial))
    write(folder/'status.json',dict(status='training',mode=mode,started_at=now()))
    for epoch in range(epoch_start+1,epochs+1):
        model.train();rng=np.random.default_rng(TRAIN_SEED+epoch);totals={};valid_count=bc_count=0;start=time.monotonic()
        for _ in range(steps_per_epoch):
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
    write(output/'completed.json',dict(epochs=epochs,best_score=best,completed_at=now()))
    write(folder/'status.json',dict(status='trained',completed_at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare-features',action='store_true');p.add_argument('--name');args=p.parse_args()
    if args.prepare_features:prepare_features()
    else:
        spec=next(e for e in EXPERIMENTS if e[0]==args.name);train(*spec)
