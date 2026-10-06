"""Compare policies on identical cached histories, not different online rollouts."""
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from .gamma_bc_ablation import OUT, ARMS
from .train_stop_rl import load_pools
from .offline.sac_cql import RecurrentSACCQL,masked_policy
from .offline.dataset import collate_episodes
from .run_gamma_study import write


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    pools,val,_=load_pools(1.,expert_only=False)
    preds={};negative={}
    for name,_ in ARMS:
        checkpoint=OUT/name/'training/latest.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=False);sig=state['signature']
        model=RecurrentSACCQL(sig['feature_dim'],sig['action_count'],state['config']['training']).eval()
        model.load_state_dict(state['model']);rows=[];bad_probs=[];bad_stop=[]
        for batch in DataLoader(val,batch_size=32,collate_fn=collate_episodes):
            logits,_=model.actor(batch['features'][:,:-1])
            p,_=masked_policy(logits,batch['action_mask'][:,:-1])
            rows.extend(p.argmax(-1)[batch['valid']].tolist())
        preds[name]=np.array(rows)
        for batch in DataLoader(pools[1],batch_size=32,collate_fn=collate_episodes):
            logits,_=model.actor(batch['features'][:,:-1])
            p,_=masked_policy(logits,batch['action_mask'][:,:-1])
            bad=batch['valid']&~batch['bc_mask']
            bad_probs.extend(p[...,40][bad].tolist());bad_stop.extend((p.argmax(-1)[bad]==40).tolist())
        negative[name]=dict(known_training_failed_stop_states=len(bad_stop),still_stop=sum(bad_stop),
                           mean_stop_probability=float(np.mean(bad_probs)))
    pairs={}
    for name,_ in ARMS:
        pairs[name]={other:dict(same_action_fraction=float(np.mean(preds[name]==preds[other])),
            different_actions=int(np.sum(preds[name]!=preds[other])),states=len(preds[name])) for other,_ in ARMS}
    write(OUT/'identical_history_policy_audit.json',dict(validation_action_agreement=pairs,
        known_training_negative_stop=negative,
        interpretation='Teacher-forced cached validation histories and known TRAINING negatives; not online navigation success or unseen STOP generalization.'))


if __name__=='__main__':main()
