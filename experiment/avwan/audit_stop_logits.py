"""Compare STOP decisions on identical cached states, separated by data provenance."""
import json
import torch
from torch.utils.data import DataLoader
from .stop_study import OUT,DATA,EXPERIMENTS
from .evaluate_stop_study import checkpoint_for
from .run_gamma_study import write,sha
from .offline.dataset import FeatureEpisodes,collate_episodes
from .offline.sac_cql import RecurrentSACCQL,masked_policy


def main():
    torch.set_num_threads(2)
    manifest=json.loads((DATA/'frozen_ratio_features/manifest.json').read_text())
    annotations=json.loads((DATA/'training_annotations.json').read_text())
    dataset=FeatureEpisodes(DATA/'frozen_ratio_features',manifest,'train')
    val=FeatureEpisodes(DATA/'frozen_ratio_features',manifest,'val')
    results=[]
    for name,_,_ in EXPERIMENTS:
        state=torch.load(checkpoint_for(name),map_location='cpu',weights_only=False)
        sig=state['signature'];model=RecurrentSACCQL(sig['feature_dim'],sig['action_count'],state['config']['training']).eval()
        model.load_state_dict(state['model'])
        groups={}
        def record(key,prob,pred):
            group=groups.setdefault(key,dict(count=0,stop_argmax=0,mean_stop_probability=0.))
            group['count']+=int(prob.numel());group['stop_argmax']+=int((pred==40).sum())
            group['mean_stop_probability']+=float(prob.sum())
        with torch.no_grad():
            for ep,meta in zip(dataset.episodes,annotations):
                logits,_=model.actor(ep['features'][None])
                probs,_=masked_policy(logits[0,:-1],ep['action_mask'][:-1]);pred=probs.argmax(-1);ps=probs[:,40]
                bad=torch.tensor(meta['false_stop'],dtype=torch.bool)
                if bad.any():record('training_actual_failed_stop_states',ps[bad],pred[bad])
                positive=(ep['action']==40)&~bad
                if positive.any():record('training_actual_successful_stop_states',ps[positive],pred[positive])
                movement=ep['action']!=40
                if meta['kind']=='recovery':record('training_recovery_movement_states',ps[movement],pred[movement])
            for batch in DataLoader(val,batch_size=32,collate_fn=collate_episodes):
                logits,_=model.actor(batch['features'][:,:-1]);probs,_=masked_policy(logits,batch['action_mask'][:,:-1])
                ps=probs[...,40];pred=probs.argmax(-1);positive=(batch['action']==40)&batch['valid']
                movement=(batch['action']!=40)&batch['valid']
                record('validation_expert_stop_states',ps[positive],pred[positive])
                record('validation_expert_movement_states',ps[movement],pred[movement])
        for value in groups.values():
            value['mean_stop_probability']/=max(value['count'],1)
            value['stop_argmax_fraction']=value['stop_argmax']/max(value['count'],1)
        result=dict(name=name,checkpoint_sha256=sha(checkpoint_for(name)),groups=groups,
                    note='Training-state metrics measure fit only. Validation expert states are separate from closed-loop navigation.')
        write(OUT/name/'identical_state_stop_audit.json',result);results.append(result)
    baseline=results[0]['groups']['training_actual_failed_stop_states']
    assert baseline['stop_argmax']==baseline['count'], 'Cached parent policy fails to reproduce its own failed STOP labels'
    write(OUT/'identical_state_stop_audit.json',dict(parent_cached_replay_verified=True,experiments=results))


if __name__=='__main__':main()
