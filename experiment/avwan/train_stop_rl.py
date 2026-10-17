"""Matched-step SAC-CQL ablations with immutable negative STOP transitions."""
import argparse
import copy
import gzip
import json
import math
import pickle
import random
import resource
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
from .offline.reward_shaping import shape_training_pools
from .offline.sac_cql import RecurrentSACCQL,masked_policy,behavior_cloning_loss
from .offline.trainer import save_checkpoint


def memory_guard(max_rss_mib=12000, min_available_gib=12.0):
    available_kib = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:')))
    available_gib = available_kib / (1024 * 1024)
    rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if rss_mib >= max_rss_mib or available_gib <= min_available_gib:
        raise MemoryError(f'Resource guard stopped offline RL: RSS={rss_mib:.0f} MiB, '
                          f'available RAM={available_gib:.1f} GiB')


def annotated_episode(ep, failed_stop, penalty, expert_control=None):
    """Training-only reward transform; the loaded features and raw file stay intact."""
    bad=torch.as_tensor(failed_stop,dtype=torch.bool)
    allowed=~bad
    if expert_control is not None:
        expert=torch.as_tensor(expert_control,dtype=torch.bool)
        assert expert.shape==allowed.shape
        allowed=allowed & expert
    return dict(ep,bc_mask=allowed,reward=ep['reward'].clone()-penalty*bad.float())


def annotate_stop_targets(episode, success):
    """Mark terminal STOP as positive only for successful source episodes."""
    episode['stop_target'] = (episode['action'] == 40) & bool(success)
    return episode


def initialize_parent_weights(model, parent, strict=True):
    """Migrate the v1 categorical actor projection into motion and STOP heads."""
    parent_state = parent['model']
    parent_arch = parent['signature']['architecture']
    factorized_arch = 'independent_lstm_discrete_sac_cql_stop_factorized_v2'
    if not model.factorized_stop_head or parent_arch == factorized_arch:
        return model.load_state_dict(parent_state, strict=strict)
    if parent_arch != 'independent_lstm_discrete_sac_cql_v1':
        raise ValueError(f'Cannot initialize factorized STOP actor from {parent_arch!r}')
    action_count = model.actor.action_count
    stop_action = model.actor.stop_action
    if parent['signature']['action_count'] != action_count or action_count != 81:
        raise ValueError('Factorized STOP migration expects the audited 81-action AVWAN space')
    motion_indices = torch.tensor([i for i in range(action_count) if i != stop_action], dtype=torch.long)
    migrated = model.state_dict()
    for key, destination in migrated.items():
        if key.startswith('actor.motion_head.'):
            old_key = key.replace('actor.motion_head.', 'actor.head.2.', 1)
            source = parent_state[old_key].index_select(0, motion_indices)
        elif key.startswith('actor.stop_head.'):
            old_key = key.replace('actor.stop_head.', 'actor.head.2.', 1)
            source = parent_state[old_key][stop_action:stop_action + 1]
        else:
            source = parent_state.get(key)
            if source is None:
                raise ValueError(f'Parent checkpoint is missing compatible weight {key}')
        if source.shape != destination.shape:
            raise ValueError(f'Parent weight shape mismatch for {key}: {source.shape} vs {destination.shape}')
        migrated[key] = source
    return model.load_state_dict(migrated, strict=True)


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
               validation_cache=None, foundation_checkpoint=None, longrange_sources=None,
               hard_correction_cache=None, hard_correction_source=None,
               curriculum=False, separate_hard_correction_pool=False, reward_shaping=None,
               stop_supervision=False):
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
    base_entries=base_manifest['splits']['train']
    base_roots=base_manifest['identity']['source']
    base_roots=base_roots if isinstance(base_roots,list) else [base_roots]
    longrange_indices=[]
    for root in (longrange_sources or []):
        resolved=Path(root).resolve()
        matches=[i for i,value in enumerate(base_roots) if Path(value).resolve()==resolved]
        if len(matches)!=1:
            raise ValueError(f'Long-range source must occur once in the base cache: {resolved}')
        longrange_indices.extend(matches)
    if len(set(longrange_indices))!=len(longrange_indices):
        raise ValueError('Duplicate long-range source roots')
    # Single-source feature caches store entries as ``episodes/...``. Only
    # multi-source snapshots use the leading numeric source index.
    is_longrange=(lambda entry: int(entry['source'].split('/',1)[0]) in longrange_indices
                  if longrange_indices else False)
    longrange_entries=[entry for entry in base_entries if is_longrange(entry)]
    original_entries=[entry for entry in base_entries if not is_longrange(entry)]
    if longrange_indices and (not longrange_entries or not original_entries):
        raise ValueError('Long-range and original feature pools must both be nonempty')
    def read_subset(entries):
        subset=dict(base_manifest)
        subset['splits']=dict(base_manifest['splits'],train=entries)
        return FeatureEpisodes(base_cache,subset,'train').episodes
    base=read_subset(original_entries)
    longrange=read_subset(longrange_entries) if longrange_indices else None
    if stop_supervision:
        success_by_path = {}
        for root in base_roots:
            index_path = Path(root) / 'episodes.jsonl'
            if not index_path.is_file():
                raise FileNotFoundError(f'Missing base index for STOP supervision: {index_path}')
            for line in index_path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get('saved'):
                    success_by_path[(Path(root) / row['file']).resolve()] = bool(row.get('success', False))
        for episode in base + (longrange or []):
            source_path = Path(episode['_source_path']).resolve()
            if source_path not in success_by_path:
                raise ValueError(f'Missing source outcome for STOP target: {source_path}')
            annotate_stop_targets(episode, success_by_path[source_path])
    hard_correction_count = 0
    hard_correction_manifest_sha256 = None
    hard = None
    if hard_correction_cache is not None:
        if longrange is None and not separate_hard_correction_pool:
            raise ValueError('Hard correction episodes require the separately sampled long-range pool')
        hard_correction_cache=Path(hard_correction_cache).resolve()
        hard_manifest=json.loads((hard_correction_cache/'manifest.json').read_text())
        if hard_manifest['identity']['checkpoint_sha256'] != sha(foundation_checkpoint):
            raise ValueError('Hard correction features use a different foundation checkpoint')
        if (hard_manifest['feature_dim'] != manifest['feature_dim'] or
                hard_manifest['action_count'] != manifest['action_count']):
            raise ValueError('Hard correction feature/action dimensions do not match')
        hard=FeatureEpisodes(hard_correction_cache,hard_manifest,'train').episodes
        if not hard:
            raise ValueError('Hard correction feature cache is empty')
        hard_correction_count=len(hard)
        hard_correction_manifest_sha256=sha(hard_correction_cache/'manifest.json')
        if not curriculum and not separate_hard_correction_pool:
            longrange.extend(hard)
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
        if stop_supervision:
            # An explicitly failed STOP is a negative example for the STOP gate.
            ep['stop_target'] = (ep['action'] == 40) & ~bad
        if meta['kind']=='policy' and bad.any():negatives.append(ep)
        elif meta['kind']=='recovery':recoveries.append(ep)
    assert negatives and recoveries
    curriculum_pool_names = None
    if curriculum:
        if hard_correction_cache is None or hard_correction_source is None:
            raise ValueError('Curriculum requires the audited hard-correction source and cache')
        roots = base_roots
        row_lookup = {}
        for source_index, root in enumerate(roots):
            index_path = Path(root) / 'episodes.jsonl'
            if not index_path.is_file():
                raise FileNotFoundError(f'Missing base training index: {index_path}')
            for line in index_path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                row_lookup[(source_index, row['scene'], str(row['episode_id']))] = row
        easy_entries, medium_entries, long_entries = [], [], []
        for entry in base_entries:
            source_index_text, relative = entry['source'].split('/', 1)
            source_index = int(source_index_text)
            source_path = Path(relative)
            scene = source_path.parent.name
            episode_id = source_path.name.removesuffix('.pkl.gz')
            row = row_lookup.get((source_index, scene, episode_id))
            if row is None or not row.get('saved'):
                raise ValueError(f'Cannot map cached episode to a saved training row: {entry["source"]}')
            distance = float(row['initial_distance'])
            if distance < 16:
                easy_entries.append(entry)
            elif distance < 24:
                medium_entries.append(entry)
            else:
                long_entries.append(entry)
        if min(len(easy_entries), len(medium_entries), len(long_entries)) == 0:
            raise ValueError('Curriculum distance pools must each contain training episodes')

        def mark_anchor(episodes, enabled):
            for episode in episodes:
                n = len(episode['action'])
                episode['anchor_mask'] = torch.full((n,), enabled, dtype=torch.bool)

        easy = read_subset(easy_entries)
        medium = read_subset(medium_entries)
        long = read_subset(long_entries)
        for episode in easy + medium + long:
            mark_anchor([episode], True)

        hard_index_path = Path(hard_correction_source) / 'episodes.jsonl'
        correction_rows = {}
        for line in hard_index_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row['scene'], str(row['episode_id']))
            if key in correction_rows:
                raise ValueError(f'Duplicate hard-correction episode in index: {key}')
            if not str(row.get('split', '')).startswith('train'):
                raise ValueError(f'Hard-correction episode is not a training row: {key}')
            correction_rows[key] = row
        hard_feature_manifest = json.loads((Path(hard_correction_cache) / 'manifest.json').read_text())
        hard_roots = hard_feature_manifest['identity']['source']
        hard_roots = hard_roots if isinstance(hard_roots, list) else [hard_roots]
        if len(hard_roots) != 1 or Path(hard_roots[0]).resolve() != Path(hard_correction_source).resolve():
            raise ValueError('Hard-correction features do not match the declared raw source')
        if len(hard) != len(hard_feature_manifest['splits']['train']):
            raise ValueError('Hard-correction feature count mismatch')
        expert_bc_transitions = 0
        for episode, entry in zip(hard, hard_feature_manifest['splits']['train']):
            source_parts = entry['source'].split('/', 1)
            if source_parts[0].isdigit():
                source_index, relative = int(source_parts[0]), source_parts[1]
            elif len(hard_roots) == 1:
                source_index, relative = 0, entry['source']
            else:
                raise ValueError(f'Cannot resolve hard-correction cache source: {entry["source"]}')
            raw_path = Path(hard_roots[source_index]) / relative
            with gzip.open(raw_path, 'rb') as handle:
                raw_episode = pickle.load(handle)
            transitions = raw_episode['transitions']
            raw_scene = transitions[0]['scene_id']
            raw_episode_id = str(transitions[0]['episode_id'])
            meta = correction_rows.get((Path(raw_scene).stem, raw_episode_id))
            if meta is None or len(transitions) != len(episode['action']):
                raise ValueError(f'Hard-correction transition index mismatch: {raw_path}')
            raw_actions = torch.as_tensor([int(transition['action']) for transition in transitions])
            if not torch.equal(episode['action'], raw_actions):
                raise ValueError(f'Hard-correction actions changed in the feature cache: {raw_path}')
            expert_mask = []
            for transition in transitions:
                useful = (bool(transition['correction_progress']) or
                          (bool(meta['success']) and int(transition['action']) == 40 and
                           bool(transition['done'])))
                expert_mask.append(bool(transition['expert_control']) and useful and
                                   int(transition['collision_count']) == 0 and
                                   int(transition['correction_collision_count']) == 0)
            episode['bc_mask'] = torch.as_tensor(expert_mask, dtype=torch.bool)
            if stop_supervision:
                annotate_stop_targets(episode, bool(meta['success']))
            mark_anchor([episode], False)
            expert_bc_transitions += int(episode['bc_mask'].sum())
        pools = [easy, medium, long, hard, negatives, recoveries]
        curriculum_pool_names = [
            'base_under_16m', 'base_16_to_24m', 'base_at_least_24m',
            'useful_executed_expert_corrections', 'actual_failed_STOP_episodes', 'recovery_episodes']
        manifest['curriculum'] = dict(pool_names=curriculum_pool_names,
            pool_sizes=[len(pool) for pool in pools], expert_bc_transitions=expert_bc_transitions,
            hard_correction_episodes=len(hard),
            distance_boundaries_m=[16, 24],
            parent_anchor_episodes=[len(easy), len(medium), len(long)])
    else:
        if separate_hard_correction_pool:
            if hard is None:
                raise ValueError('A separate correction pool requires correction episodes')
            hard_index_path = Path(hard_correction_source) / 'episodes.jsonl'
            correction_rows = {}
            for line in hard_index_path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row['scene'], str(row['episode_id']))
                if key in correction_rows or not str(row.get('split', '')).startswith('train'):
                    raise ValueError(f'Invalid/duplicate training correction row: {key}')
                correction_rows[key] = row
            hard_feature_manifest = json.loads((Path(hard_correction_cache) / 'manifest.json').read_text())
            hard_roots = hard_feature_manifest['identity']['source']
            hard_roots = hard_roots if isinstance(hard_roots, list) else [hard_roots]
            if len(hard_roots) != 1 or Path(hard_roots[0]).resolve() != Path(hard_correction_source).resolve():
                raise ValueError('Correction features do not match the declared raw source')
            expert_bc_transitions = 0
            for episode, entry in zip(hard, hard_feature_manifest['splits']['train']):
                source_parts = entry['source'].split('/', 1)
                if source_parts[0].isdigit():
                    source_index, relative = int(source_parts[0]), source_parts[1]
                else:
                    source_index, relative = 0, entry['source']
                raw_path = Path(hard_roots[source_index]) / relative
                with gzip.open(raw_path, 'rb') as handle:
                    raw_episode = pickle.load(handle)
                transitions = raw_episode['transitions']
                raw_scene = transitions[0]['scene_id']
                raw_episode_id = str(transitions[0]['episode_id'])
                meta = correction_rows.get((Path(raw_scene).stem, raw_episode_id))
                if meta is None or len(transitions) != len(episode['action']):
                    raise ValueError(f'Correction transition index mismatch: {raw_path}')
                raw_actions = torch.as_tensor([int(t['action']) for t in transitions])
                if not torch.equal(episode['action'], raw_actions):
                    raise ValueError(f'Correction actions changed in the feature cache: {raw_path}')
                allowed = []
                for transition in transitions:
                    useful = (bool(transition['correction_progress']) or
                              (bool(meta['success']) and int(transition['action']) == 40 and
                               bool(transition['done'])))
                    allowed.append(bool(transition['expert_control']) and useful and
                                   int(transition['collision_count']) == 0 and
                                   int(transition['correction_collision_count']) == 0)
                episode['bc_mask'] = torch.as_tensor(allowed, dtype=torch.bool)
                if stop_supervision:
                    annotate_stop_targets(episode, bool(meta['success']))
                episode['anchor_mask'] = torch.zeros(len(episode['action']), dtype=torch.bool)
                expert_bc_transitions += int(episode['bc_mask'].sum())
            pools=([base,longrange,hard,negatives,recoveries] if longrange is not None else
                   [base,hard,negatives,recoveries])
            names=(['base','legacy_short_distance_augment','productive_parent_failure_corrections',
                    'actual_failed_STOP_episodes','recovery_episodes'] if longrange is not None else
                   ['base_including_unrebalanced_augment','productive_parent_failure_corrections',
                    'actual_failed_STOP_episodes','recovery_episodes'])
            manifest['separate_correction_pool'] = dict(
                pool_names=names,
                pool_sizes=[len(pool) for pool in pools],
                bc_transitions=expert_bc_transitions)
        else:
            pools=([base,longrange,negatives,recoveries] if longrange is not None
                   else [base,negatives,recoveries])
    manifest['hard_correction'] = dict(episodes=hard_correction_count,
        manifest_sha256=hard_correction_manifest_sha256)
    if reward_shaping:
        manifest['reward_shaping'] = dict(config=reward_shaping,
            training_stats=shape_training_pools(pools, manifest, reward_shaping))
    return pools,val,manifest


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
          training_overrides=None,longrange_sources=None,device='cuda:0',cpu_threads=4,gpu_memory_fraction=.12,
          max_rss_mib=12000,min_available_gib=12.0,random_initialization=False,
          hard_correction_cache=None,hard_correction_source=None,curriculum_schedule=None,
          training_method='sac_cql',separate_hard_correction_pool=False,reward_shaping=None):
    folder=OUT/name;folder.mkdir(exist_ok=True);output=folder/'training'
    if not isinstance(cpu_threads, int) or cpu_threads < 1:
        raise ValueError('cpu_threads must be a positive integer')
    custom_base_cache=base_cache is not None
    base_cache=Path(base_cache).resolve() if base_cache is not None else BASE_CACHE
    foundation_checkpoint=Path(foundation_checkpoint).resolve() if foundation_checkpoint is not None else FOUNDATION
    parent_checkpoint=(Path(parent_checkpoint).resolve() if parent_checkpoint is not None
                       else (None if random_initialization else PARENT))
    additional_cache=Path(additional_cache).resolve() if additional_cache is not None else DATA/'frozen_ratio_features'
    validation_cache=Path(validation_cache).resolve() if validation_cache is not None else additional_cache
    random.seed(TRAIN_SEED);np.random.seed(TRAIN_SEED);torch.manual_seed(TRAIN_SEED);torch.set_num_threads(cpu_threads)
    memory_guard(max_rss_mib,min_available_gib)
    if parent_checkpoint is None:
        parent=None
        cfg={
            'seed': TRAIN_SEED,
            'features': {},
            'training': {
                'epochs': epochs, 'output': str(output), 'batch_size': 32,
                'hidden_dim': 256, 'lstm_layers': 1, 'learning_rate': 1e-4,
                'gamma': .01, 'initial_alpha': .03, 'entropy_ratio': .1,
                'tau': .005, 'cql_weight': 1., 'cql_temperature': 1.,
                'grad_clip': 10., 'log_every': 100, 'tensorboard': False,
                'terminal_mask': True, 'foundation_prior': True,
                'bc_weight': 5., 'actor_rl_weight': 1., 'weight_decay': 0.,
            },
        }
    else:
        parent=torch.load(parent_checkpoint,map_location='cpu',weights_only=False)
        cfg=copy.deepcopy(parent['config'])
    cfg['seed']=TRAIN_SEED;cfg['training'].update(epochs=epochs,output=str(output))
    if training_method not in ('sac_cql', 'actor_only_bc'):
        raise ValueError(f'Unknown training method: {training_method}')
    overrides = dict(training_overrides or {})
    allowed = {'batch_size','gamma','bc_weight','actor_rl_weight','cql_weight','cql_temperature',
               'entropy_ratio','initial_alpha','tau','learning_rate','weight_decay','grad_clip',
               'hidden_dim','lstm_layers','terminal_mask','foundation_prior','parent_anchor_weight',
               'discount_mode','factorized_stop_head','stop_aux_weight'}
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f'Unsupported SAC-CQL training overrides: {sorted(unknown)}')
    cfg['training'].update(overrides)
    if parent is not None and (
            cfg['training']['hidden_dim'] != parent['config']['training']['hidden_dim'] or
            cfg['training']['lstm_layers'] != parent['config']['training']['lstm_layers']):
        raise ValueError('hidden_dim and lstm_layers must match the parent actor architecture')
    for key in ('gamma', 'entropy_ratio'):
        if not 0 <= cfg['training'][key] <= 1:
            raise ValueError(f'{key} must be in [0, 1]')
    if cfg['training'].get('discount_mode', 'smdp') not in ('smdp', 'fixed_gamma'):
        raise ValueError("discount_mode must be 'smdp' or 'fixed_gamma'")
    for key in ('learning_rate','initial_alpha','cql_temperature','grad_clip'):
        if cfg['training'][key] <= 0:
            raise ValueError(f'{key} must be positive')
    for key in ('bc_weight','actor_rl_weight','cql_weight','weight_decay'):
        if cfg['training'][key] < 0:
            raise ValueError(f'{key} must be nonnegative')
    if cfg['training'].get('stop_aux_weight', 0.) < 0:
        raise ValueError('stop_aux_weight must be nonnegative')
    if not isinstance(cfg['training'].get('factorized_stop_head', False), bool):
        raise ValueError('factorized_stop_head must be boolean')
    if cfg['training'].get('stop_aux_weight', 0.) > 0 and not cfg['training'].get('factorized_stop_head', False):
        raise ValueError('stop_aux_weight requires factorized_stop_head')
    if not 0 < cfg['training']['tau'] <= 1:
        raise ValueError('tau must be in (0, 1]')
    if not isinstance(cfg['training']['terminal_mask'], bool) or \
            not isinstance(cfg['training']['foundation_prior'], bool):
        raise ValueError('terminal_mask and foundation_prior must be booleans')
    if cfg['training'].get('parent_anchor_weight', 0.) < 0:
        raise ValueError('parent_anchor_weight must be nonnegative')
    cfg['features'].update(checkpoint=str(foundation_checkpoint),cache=str(additional_cache),
        source=str(DATA/'feature_source'),val_source=str(PREVIOUS/'validation_demonstrations'))
    expert_only=mode=='recovery_expert_bc';sampling_mode='recovery' if expert_only else mode
    default_sampling={'original':[32,0,0],'negative':[24,8,0],'recovery':[16,8,8]}[sampling_mode]
    batch_size = cfg['training']['batch_size']
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    sampling = list(sampling) if sampling is not None else default_sampling
    expected_pool_counts = ((6,) if curriculum_schedule is not None else
                            ((5,) if longrange_sources else (4,))
                            if separate_hard_correction_pool else (3, 4))
    if len(sampling) not in expected_pool_counts or any(n<0 for n in sampling) or sum(sampling)!=batch_size:
        raise ValueError(
            f'Sampling must contain {expected_pool_counts} pool counts totaling batch_size={batch_size}')
    if len(sampling)==4 and not longrange_sources and not separate_hard_correction_pool:
        raise ValueError('Four-pool sampling requires longrange_sources')
    if len(sampling)==3 and longrange_sources:
        raise ValueError('Long-range sampling requires four pool counts')
    if separate_hard_correction_pool:
        expected_separate = 5 if longrange_sources else 4
        if len(sampling) != expected_separate:
            raise ValueError('Separate correction sampling requires four base pools or five with a longrange pool')
    if penalty < 0:
        raise ValueError('failed STOP penalty must be nonnegative')
    if reward_shaping is not None:
        required = {'long_distance_threshold', 'long_success_bonus', 'failure_penalty',
                    'progress_reward_per_meter', 'away_penalty_per_meter'}
        if set(reward_shaping) != required:
            raise ValueError(f'reward_shaping must define exactly {sorted(required)}')
        if any(not math.isfinite(float(value)) for value in reward_shaping.values()):
            raise ValueError('Reward-shaping values must be finite')
        if (float(reward_shaping['long_distance_threshold']) <= 0 or
                any(float(reward_shaping[key]) < 0 for key in required - {'long_distance_threshold'})):
            raise ValueError('Reward bonuses, penalties and scales must be nonnegative')
    cfg['stop_pipeline']=dict(mode=mode,failed_stop_penalty=penalty,bc_masked=True,parent=str(parent_checkpoint),
        sampling=sampling,steps_per_epoch=steps_per_epoch,
        resource_limits=dict(gpu_memory_fraction=gpu_memory_fraction,
                             max_rss_mib=max_rss_mib,
                             min_available_gib=min_available_gib),
        effective_optimizer=f"Adam, learning_rate={cfg['training']['learning_rate']}, "
                             f"weight_decay={cfg['training'].get('weight_decay', 0.)}",
        checkpoint_selection='balanced STOP NLL + 0.1*action NLL on independent validation demonstrations')
    cfg['stop_pipeline']['reward_shaping'] = reward_shaping
    cfg['stop_pipeline']['parent'] = str(parent_checkpoint) if parent_checkpoint is not None else None
    cfg['stop_pipeline']['actor_initialization'] = ('parent_checkpoint' if parent_checkpoint is not None
                                                    else 'random')
    if custom_base_cache:
        cfg['stop_pipeline']['base_cache']=str(base_cache)
    if additional_cache != DATA/'frozen_ratio_features':
        cfg['stop_pipeline']['additional_cache']=str(additional_cache)
    if validation_cache != additional_cache:
        cfg['stop_pipeline']['validation_cache']=str(validation_cache)
    if foundation_checkpoint != FOUNDATION:
        cfg['stop_pipeline']['prior_replaced_with_foundation_sha256']=sha(foundation_checkpoint)
    hard_correction_cache=Path(hard_correction_cache).resolve() if hard_correction_cache else None
    is_curriculum = curriculum_schedule is not None
    cfg['stop_pipeline']['training_method'] = training_method
    cfg['stop_pipeline']['separate_hard_correction_pool'] = bool(separate_hard_correction_pool)
    pools,val,manifest=load_pools(penalty,expert_only,base_cache,additional_cache,
        validation_cache,foundation_checkpoint,longrange_sources,hard_correction_cache,
        hard_correction_source,curriculum=is_curriculum,
        separate_hard_correction_pool=separate_hard_correction_pool,
        reward_shaping=reward_shaping,
        stop_supervision=bool(cfg['training'].get('factorized_stop_head', False)))
    hard_info=manifest.get('hard_correction',dict(episodes=0,manifest_sha256=None))
    if is_curriculum:
        if len(sampling) != len(pools):
            raise ValueError(f'{len(pools)} data pools require the same number of sampling counts')
        if not parent_checkpoint:
            raise ValueError('Parent-anchored curriculum requires a parent checkpoint')
        if longrange_sources is None or not longrange_sources:
            raise ValueError('Curriculum requires separately identified long-range training sources')
        normalized_schedule=[]
        for stage in curriculum_schedule:
            stage_name=str(stage['name'])
            stage_epochs=int(stage['epochs'])
            stage_sampling=[int(value) for value in stage['sampling']]
            if stage_epochs < 1 or len(stage_sampling) != len(pools) or any(value < 0 for value in stage_sampling):
                raise ValueError(f'Invalid curriculum stage: {stage}')
            if sum(stage_sampling) != batch_size:
                raise ValueError(f'Curriculum stage {stage_name} sampling must sum to {batch_size}')
            if any(count > 0 and len(pool) == 0 for pool, count in zip(pools, stage_sampling)):
                raise ValueError(f'Curriculum stage {stage_name} samples an empty pool')
            normalized_schedule.append(dict(name=stage_name, epochs=stage_epochs, sampling=stage_sampling))
        if not normalized_schedule or sum(stage['epochs'] for stage in normalized_schedule) != epochs:
            raise ValueError('Curriculum stage epochs must sum to training.epochs')
        curriculum_schedule=normalized_schedule
        sampling=[stage['sampling'] for stage in curriculum_schedule]
        cfg['stop_pipeline']['sampling']=sampling[0]
        cfg['stop_pipeline']['curriculum_schedule']=curriculum_schedule
        cfg['stop_pipeline']['sampling_order']=manifest['curriculum']['pool_names']
        cfg['stop_pipeline']['parent_anchor_weight']=cfg['training'].get('parent_anchor_weight',0.)
    elif len(pools)!=len(sampling):
        raise ValueError(f'{len(pools)} data pools require the same number of sampling counts')
    if separate_hard_correction_pool and not is_curriculum:
        if len(sampling) != len(pools):
            raise ValueError('Separate correction sampling count does not match the data pools')
        correction_index = 2 if longrange_sources else 1
        anchor_pools = pools[:correction_index]
        correction_pool = pools[correction_index]
        for episode in [ep for pool in anchor_pools for ep in pool]:
            episode['anchor_mask'] = torch.ones(len(episode['action']), dtype=torch.bool)
        for episode in correction_pool:
            episode['anchor_mask'] = torch.zeros(len(episode['action']), dtype=torch.bool)
        for episode in [ep for pool in pools[correction_index+1:] for ep in pool]:
            episode['anchor_mask'] = torch.zeros(len(episode['action']), dtype=torch.bool)
        if training_method == 'actor_only_bc':
            for pool_index, pool in enumerate(pools):
                for episode in pool:
                    if pool_index != correction_index:
                        episode['bc_mask'] = torch.zeros(len(episode['action']), dtype=torch.bool)
    if training_method == 'actor_only_bc' and not separate_hard_correction_pool:
        raise ValueError('actor_only_bc requires a separate productive correction pool')
    if longrange_sources:
        cfg['stop_pipeline']['longrange_sources']=[str(Path(path).resolve()) for path in longrange_sources]
        if not is_curriculum:
            cfg['stop_pipeline']['sampling_order']=(
                ['original','existing_longrange','productive_parent_failure_corrections',
                 'actual_failed_STOP_episodes','recovery_episodes']
                if separate_hard_correction_pool else
                ['original','longrange','actual_failed_STOP_episodes','recovery_episodes'])
    elif separate_hard_correction_pool and not is_curriculum:
        cfg['stop_pipeline']['sampling_order']=[
            'original_including_unrebalanced_augment','productive_parent_failure_corrections',
            'actual_failed_STOP_episodes','recovery_episodes']
    if hard_correction_cache is not None:
        cfg['stop_pipeline']['hard_correction_cache']=str(hard_correction_cache)
        cfg['stop_pipeline']['hard_correction_manifest_sha256']=hard_info['manifest_sha256']
        cfg['stop_pipeline']['hard_correction_episodes']=hard_info['episodes']
    if hard_correction_source is not None:
        cfg['stop_pipeline']['hard_correction_source']=str(Path(hard_correction_source).resolve())
    if separate_hard_correction_pool:
        cfg['stop_pipeline']['correction_bc_scope'] = (
            'actually executed expert actions with local goal progress or successful STOP, '
            'with zero primitive collisions; actual mixed-policy transitions remain in the critic pool')
    memory_guard(max_rss_mib,min_available_gib)
    if expert_only:
        cfg['stop_pipeline']['new_data_bc_scope']='only actually executed expert actions; original pool unchanged'
        cfg['stop_pipeline']['expert_annotation_sha256']=sha(DATA/'expert_bc_annotations.json')
    if (output/'completed.json').exists():
        saved=json.loads((output/'config.json').read_text())
        if saved != cfg:
            raise ValueError('This run is already complete with different settings; change RUN.NAME for a new run')
        print(f'Run already complete with matching configuration: {output}', flush=True)
        return
    if parent is not None:
        signature=copy.deepcopy(parent['signature'])
    else:
        signature={'architecture':'independent_lstm_discrete_sac_cql_v1',
                   'feature_dim':manifest['feature_dim'],
                   'action_count':manifest['action_count']}
    if cfg['training'].get('factorized_stop_head', False):
        signature['architecture']='independent_lstm_discrete_sac_cql_stop_factorized_v2'
        cfg['stop_pipeline']['architecture_migration'] = (
            'v1 categorical actor projection copied into separate 80-way motion and center STOP heads')
        cfg['stop_pipeline']['stop_supervision'] = (
            'balanced STOP NLL; successful terminal STOP=positive, failed STOP=negative, motion actions=negative')
    signature['features']=copy.deepcopy(manifest['identity'])
    signature['stop_annotations_sha256']=sha(DATA/'training_annotations.json')
    if expert_only:signature['expert_bc_annotation_sha256']=cfg['stop_pipeline']['expert_annotation_sha256']
    device=torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested for offline RL but is unavailable')
    if device.type == 'cuda':
        if not 0 < gpu_memory_fraction <= 1:
            raise ValueError('gpu_memory_fraction must be in (0, 1]')
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction,device)
    model=RecurrentSACCQL(signature['feature_dim'],signature['action_count'],cfg['training']).to(device)
    if parent is not None:
        incompatible = initialize_parent_weights(model, parent, strict=cfg['training']['foundation_prior'])
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
    anchor_weight = float(cfg['training'].get('parent_anchor_weight', 0.))
    reference_actor = copy.deepcopy(model.actor).to(device).eval() if (is_curriculum or anchor_weight > 0 or training_method == 'actor_only_bc') else None
    if reference_actor is not None:
        for parameter in reference_actor.parameters():
            parameter.requires_grad_(False)
    with torch.no_grad():
        model.log_alpha.fill_(math.log(cfg['training']['initial_alpha']))
    initialization_checkpoint_sha256 = sha(parent_checkpoint) if parent_checkpoint is not None else None
    optimizers=model.make_optimizers();output.mkdir(exist_ok=True)
    stop_positive = sum(int(episode['stop_target'].sum()) for pool in pools for episode in pool
                        if 'stop_target' in episode)
    stop_labeled = sum(len(episode['action']) for pool in pools for episode in pool
                       if 'stop_target' in episode)
    write(output/'config.json',cfg);(folder/'config.yaml').write_text(yaml.safe_dump(cfg))
    write(output/'dataset.json',dict(base_cache=str(base_cache),new_cache=str(additional_cache),
        validation_cache=str(validation_cache),foundation=str(foundation_checkpoint),
        pool_names=cfg['stop_pipeline'].get('sampling_order',
            ['original','actual_failed_STOP_episodes','recovery_episodes']),
        pool_sizes=[len(p) for p in pools],sampling=cfg['stop_pipeline']['sampling'],
        curriculum=manifest.get('curriculum'),
        hard_correction_cache=str(hard_correction_cache) if hard_correction_cache else None,
        hard_correction_episodes=hard_info['episodes'],
        stop_supervision=dict(enabled=bool(cfg['training'].get('factorized_stop_head', False)),
                              positive_transitions=stop_positive,
                              negative_transitions=stop_labeled-stop_positive),
        reward_shaping=manifest.get('reward_shaping'),
        annotations_sha256=signature['stop_annotations_sha256']))
    epoch_start=0;best=float('inf')
    def checkpoint(epoch,metrics,path):
        save_checkpoint(path,dict(epoch=epoch,model=model.state_dict(),signature=signature,config=cfg,
            validation=metrics,best_score=best,optimizers={k:v.state_dict() for k,v in optimizers.items()},
            initialization_checkpoint_sha256=initialization_checkpoint_sha256,
            selection=cfg['stop_pipeline']['checkpoint_selection']))
    if (output/'latest.pt').exists():
        state=torch.load(output/'latest.pt',map_location=device,weights_only=False);assert state['config']==cfg
        model.load_state_dict(state['model']);epoch_start=state['epoch'];best=state['best_score']
        for k,opt in optimizers.items():opt.load_state_dict(state['optimizers'][k])
    else:
        initial=validate(model,val,device);best=initial['selection_score'];write(output/'initial_validation.json',initial)
        checkpoint(0,initial,output/'best_policy.pt');write(output/'best.json',dict(epoch=0,**initial))
    write(folder/'status.json',dict(status='training',mode=mode,started_at=now()))
    stage_ends=[]
    if is_curriculum:
        stage_end=0
        for stage in curriculum_schedule:
            stage_end += stage['epochs'];stage_ends.append((stage_end,stage))
    for epoch in range(epoch_start+1,epochs+1):
        if is_curriculum:
            stage_name,epoch_sampling=next((stage['name'],stage['sampling'])
                                           for end,stage in stage_ends if epoch <= end)
        else:
            stage_name,epoch_sampling=None,cfg['stop_pipeline']['sampling']
        model.train();rng=np.random.default_rng(TRAIN_SEED+epoch);totals={};valid_count=bc_count=0;start=time.monotonic()
        for _ in range(steps_per_epoch):
            episodes=[]
            for pool,n in zip(pools,epoch_sampling):
                if n:
                    if not pool:
                        raise ValueError('A nonzero batch quota targets an empty data pool')
                    episodes.extend(pool[int(i)] for i in rng.integers(len(pool),size=n))
            batch={k:v.to(device) for k,v in collate_episodes(episodes).items()}
            if training_method == 'actor_only_bc':
                metrics=actor_only_bc_update(model,batch,optimizers['actor'],reference_actor,
                    bc_weight=cfg['training']['bc_weight'],anchor_weight=anchor_weight)
            else:
                metrics=model.update(batch,optimizers,reference_actor=reference_actor)
            n=int(batch['valid'].sum());valid_count+=n;bc_count+=int((batch['valid']&batch['bc_mask']).sum())
            for k,v in metrics.items():totals[k]=totals.get(k,0.)+v*n
        training={k:v/valid_count for k,v in totals.items()};training.update(transitions=valid_count,bc_allowed=bc_count,
            bc_excluded=valid_count-bc_count,seconds=time.monotonic()-start)
        metrics=validate(model,val,device)
        if metrics['selection_score']<best:
            best=metrics['selection_score'];checkpoint(epoch,metrics,output/'best_policy.pt');write(output/'best.json',dict(epoch=epoch,**metrics))
        checkpoint(epoch,metrics,output/'latest.pt')
        record=dict(epoch=epoch,train=training,val=metrics)
        if is_curriculum:
            record['curriculum_stage']=dict(name=stage_name,sampling=epoch_sampling)
        with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        memory_guard(max_rss_mib,min_available_gib)
    write(output/'completed.json',dict(epochs=epochs,best_score=best,completed_at=now()))
    write(folder/'status.json',dict(status='trained',completed_at=now()))


def actor_only_bc_update(model,batch,optimizer,reference_actor,bc_weight,anchor_weight):
    """Train only the policy head from executed expert corrections and parent KL."""
    states,masks,valid=batch['features'],batch['action_mask'],batch['valid']
    logits,_=model.actor(states[:,:-1])
    probs,log_probs=masked_policy(logits,masks[:,:-1])
    bc_nll=behavior_cloning_loss(log_probs,batch['action'],valid,batch['bc_mask'])
    parent_kl=bc_nll.new_zeros(())
    anchor_mask=batch.get('anchor_mask')
    if reference_actor is not None and anchor_weight>0 and anchor_mask is not None:
        with torch.no_grad():
            ref_logits,_=reference_actor(states[:,:-1])
            ref_probs,ref_log_probs=masked_policy(ref_logits,masks[:,:-1])
        per_state=(ref_probs*(ref_log_probs-log_probs)).sum(-1)
        allowed=valid & anchor_mask.bool()
        parent_kl=per_state[allowed].sum()/allowed.sum().clamp_min(1)
    loss=bc_weight*bc_nll+anchor_weight*parent_kl
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite actor-only BC loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    parameters=[p for group in optimizer.param_groups for p in group['params']]
    torch.nn.utils.clip_grad_norm_(parameters,model.config['grad_clip'],error_if_nonfinite=True)
    optimizer.step()
    entropy=-(probs*log_probs).sum(-1)
    policy_nll=-log_probs.gather(-1,batch['action'].unsqueeze(-1)).squeeze(-1)
    return {'critic_loss':0.,'td_loss':0.,'cql_loss':0.,'actor_loss':float(loss.detach()),
        'alpha_loss':0.,'alpha':float(model.log_alpha.exp().detach()),
        'entropy':float(entropy[valid].mean().detach()),'target_entropy':0.,
        'action_accuracy':float((probs.argmax(-1)[valid]==batch['action'][valid]).float().mean().detach()),
        'policy_nll':float(policy_nll[valid].mean().detach()),'bc_nll':float(bc_nll.detach()),
        'parent_kl':float(parent_kl.detach()),'q_data':0.,'td_target':0.}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare-features',action='store_true');p.add_argument('--name');args=p.parse_args()
    if args.prepare_features:prepare_features()
    else:
        spec=next(e for e in EXPERIMENTS if e[0]==args.name);train(*spec)
