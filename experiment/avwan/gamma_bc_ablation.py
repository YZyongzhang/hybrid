"""Matched continuation study: pure BC versus SAC-CQL decision discounts.

Reuses immutable STOP-study data. Every policy starts from the same parent.
"""
import argparse
import copy
import csv
import json
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .stop_study import OUT as SOURCE, DATA, PARENT, FOUNDATION, PYTHON, SPLITS, TRAIN_SEED
from .run_gamma_study import ROOT, write, sha, now, command_run
from .train_stop_rl import load_pools, validate
from .offline.dataset import collate_episodes
from .offline.sac_cql import RecurrentSACCQL, masked_policy, behavior_cloning_loss
from .offline.trainer import save_checkpoint

OUT = ROOT / 'result/2026-10-04-gamma-vs-bc-stop-recovery'
ARMS = [('exp001-pure-bc', None), ('exp002-sac-gamma0-bc5', 0.),
        ('exp003-sac-gamma0p01-bc5', .01), ('exp004-sac-gamma0p9-bc5', .9),
        ('exp005-sac-gamma0p99-bc5', .99)]


def append(path, row):
    with path.open('a') as f: f.write(json.dumps(row, allow_nan=False) + '\n')


def pure_bc_update(model, batch, optimizer):
    """No critic, target, temperature or reward computation in this arm."""
    logits, _ = model.actor(batch['features'][:, :-1])
    _, lp = masked_policy(logits, batch['action_mask'][:, :-1])
    nll = behavior_cloning_loss(lp, batch['action'], batch['valid'], batch['bc_mask'])
    loss = model.config['bc_weight'] * nll
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite BC loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = nn.utils.clip_grad_norm_(model.actor.parameters(), model.config['grad_clip'], error_if_nonfinite=True)
    optimizer.step()
    return dict(actor_loss=float(loss.detach()), bc_nll=float(nll.detach()), actor_grad_norm=float(norm))


def diagnostics(model, batch, pure=False):
    """Fixed training batch, independent autograd probes; no update or RNG use."""
    # cuDNN recurrent backward requires a training-mode forward. These heads
    # contain no dropout/batchnorm; the probe remains deterministic.
    was_training=model.training
    model.train()
    states, mask, valid = batch['features'], batch['action_mask'], batch['valid']
    logits, _ = model.actor(states)
    p, lp = masked_policy(logits, mask)
    p, lp = p[:, :-1], lp[:, :-1]
    bc = model.config['bc_weight'] * behavior_cloning_loss(lp, batch['action'], valid, batch['bc_mask'])
    pars = [x for x in model.actor.parameters() if x.requires_grad]
    def grad(loss):
        return torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss, pars, retain_graph=True)])
    gb = grad(bc)
    result = dict(weighted_bc_loss=float(bc.detach()), weighted_bc_grad_norm=float(gb.norm()),
                  bc_weight=model.config['bc_weight'], gamma=None if pure else model.config['gamma'])
    if pure:
        model.train(was_training)
        return dict(result, q_grad_norm=0., entropy_grad_norm=0., rl_grad_norm=0.,
                    rl_to_bc_grad_ratio=0., q_training_active=False)
    with torch.no_grad():
        q1, _ = model.q1(states[:, :-1]); q2, _ = model.q2(states[:, :-1])
        q = torch.minimum(q1, q2)
        t1, _ = model.target_q1(states); t2, _ = model.target_q2(states)
        pall, lall = masked_policy(logits, mask)
        alpha = model.log_alpha.exp().detach()
        nextv = (pall * (torch.minimum(t1, t2) - alpha * lall)).sum(-1)[:, 1:]
        future = model.config['gamma'] * nextv * (batch['discount'] > 0)
        target = batch['reward'] + future
        dataq = q.gather(-1, batch['action'][..., None]).squeeze(-1)
    qloss = -(p * q).sum(-1)[valid].mean()
    eloss = (p * lp * alpha).sum(-1)[valid].mean()
    gq, ge = grad(qloss), grad(eloss)
    gr = gq + ge
    cosine = lambda a,b: float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-20))
    result.update(q_training_active=True, q_actor_loss=float(qloss.detach()), entropy_actor_loss=float(eloss.detach()),
        q_grad_norm=float(gq.norm()), entropy_grad_norm=float(ge.norm()), rl_grad_norm=float(gr.norm()),
        rl_to_bc_grad_ratio=float(gr.norm()/gb.norm().clamp_min(1e-20)), rl_bc_cosine=cosine(gr,gb),
        q_bc_cosine=cosine(gq,gb), alpha=float(alpha), reward_mean=float(batch['reward'][valid].mean()),
        bootstrap_abs_mean=float(future[valid].abs().mean()), td_target_mean=float(target[valid].mean()),
        td_target_abs_max=float(target[valid].abs().max()), q_data_mean=float(dataq[valid].mean()),
        q_data_abs_max=float(dataq[valid].abs().max()),
        td_rmse=float((dataq[valid]-target[valid]).square().mean().sqrt()))
    # Compare Q with observed discounted return only on complete episodes.
    # This is a behavior-return diagnostic, NOT an unbiased Q* error estimate.
    errors=[]; mc_values=[]
    for i in range(len(valid)):
        length=int(valid[i].sum())
        if batch['discount'][i,length-1] != 0: continue
        ret=0.
        for t in reversed(range(length)):
            ret=float(batch['reward'][i,t])+model.config['gamma']*ret
            errors.append(float(dataq[i,t])-ret); mc_values.append(ret)
    result.update(behavior_return_comparison_count=len(errors),
        q_minus_behavior_return_mean=float(np.mean(errors)) if errors else None,
        behavior_return_mean=float(np.mean(mc_values)) if mc_values else None)
    if any(isinstance(v,float) and not np.isfinite(v) for v in result.values()):
        raise FloatingPointError('Nonfinite diagnostic')
    model.train(was_training)
    return result


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists(): return
    for split in SPLITS: shutil.copytree(SOURCE/'selections'/split,OUT/'selections'/split)
    shutil.copyfile(SOURCE/'evaluation.yaml',OUT/'evaluation.yaml')
    write(OUT/'protocol.json',dict(created_at=now(),arms=ARMS,parent=str(PARENT),parent_sha256=sha(PARENT),
        foundation=str(FOUNDATION),foundation_sha256=sha(FOUNDATION),data=str(DATA),
        training_annotations_sha256=sha(DATA/'training_annotations.json'),
        feature_manifest_sha256=sha(DATA/'frozen_ratio_features/manifest.json'),
        source_reference='exp005-recovery-failed-stop-penalty1',sampling=[16,8,8],
        failed_stop_penalty=1.,failed_stop_bc_mask=True,expert_only_bc=False,
        train_seed=TRAIN_SEED,epochs=40,steps_per_epoch=186,learning_rate=1e-4,bc_weight=5.,
        pure_bc='actor-only weighted BC; no critic/alpha/target updates; inherited actor initialization',
        gamma_semantics='per high-level decision, not primitive simulator step',
        checkpoint_selection='fixed epoch 40 for every arm; no surrogate checkpoint selection',
        configuration_selection='maximum independent validation navigation SPL, before NEW test evaluation',
        known_previous_test_results=True,limitation='same historical fixed100 tests, exploratory not a fresh blind test',
        diagnostics='fixed train batch each epoch: BC/Q/entropy gradients before clipping, cosine, Q, TD and observed behavior returns',
        evaluation=dict(splits=SPLITS,episodes_per_split=100,fresh_process=True,initial_input_hash_audit=True),
        reuse='prior gamma .01 evaluations only if EVERY state tensor is exactly equal after retraining; original parent baseline reused'))
    snapshot=OUT/'source_snapshot'; snapshot.mkdir()
    for path in [Path(__file__),ROOT/'experiment/avwan/evaluate_stop_study.py',ROOT/'experiment/avwan/train_stop_rl.py',
                 ROOT/'experiment/avwan/offline/sac_cql.py',ROOT/'experiment/avwan/offline/dataset.py']:
        shutil.copyfile(path,snapshot/path.name)
    write(OUT/'source_hashes.json',{p.name:sha(p) for p in snapshot.iterdir()})
    (OUT/'PROTOCOL.md').write_text('''# Gamma 与 BC 对照实验

固定 exp005 的训练数据和失败 STOP 惩罚：每批 16 原数据、8 真实失败 STOP、8 恢复轨迹；屏蔽错误 STOP 的 BC，但保留其他实际动作 BC（包含模型移动前缀）。不引入 exp006 的 expert-only BC 改动。

五组分别为纯 BC、SAC-CQL gamma=0/0.01/0.9/0.99；BC 系数固定 5、学习率 1e-4、40 epoch × 186 批。都从同一旧 parent 权重开始并重置优化器。纯 BC 仅更新 Actor，不更新 Critic、熵温度或 target；奖励不进入其优化目标。这里隔离的是继续训练目标的影响，不等于从零训练纯 BC 与 RL 的比较。

使用每组固定第 40 epoch；配置按独立 unheard 验证集 100 条导航 SPL 选择，再运行 heard/unheard 固定测试各 100 条。测试集合已经在历史实验使用过，属于探索性复用，不能称为全新盲测。每条 episode 使用独立进程，核对初始输入哈希。

每个 epoch 在同一训练 batch 检查 BC、Q、熵项裁剪前梯度范数及夹角，记录 Q 值、TD 目标、bootstrap 和行为轨迹回报差异。后者不等于真实最优 Q 误差，只用于观察数值趋势。终止状态不 bootstrap。gamma 是高层决策折扣。

已有 parent 评估直接引用；只有新训 gamma=.01 的全部模型状态与历史 exp005 逐张量完全相等，才复用其评估并保存依据。原始数据只读，所有新增过程与结果放在本目录。
''')


def train(name,gamma):
    folder=OUT/name; output=folder/'training'; output.mkdir(parents=True,exist_ok=True)
    if (output/'completed.json').exists(): return
    random.seed(TRAIN_SEED);np.random.seed(TRAIN_SEED);torch.manual_seed(TRAIN_SEED);torch.set_num_threads(4)
    parent=torch.load(PARENT,map_location='cpu',weights_only=False)
    cfg=copy.deepcopy(json.loads((SOURCE/'exp005-recovery-failed-stop-penalty1/training/config.json').read_text()))
    cfg['training'].update(output=str(output),gamma=0. if gamma is None else gamma,actor_rl_weight=0. if gamma is None else 1.)
    cfg['gamma_bc_study']=dict(pure_bc=gamma is None,checkpoint_selection='fixed final epoch 40')
    pools,val,manifest=load_pools(1.,expert_only=False)
    signature=copy.deepcopy(parent['signature']);signature['features']=copy.deepcopy(manifest['identity'])
    signature['stop_annotations_sha256']=sha(DATA/'training_annotations.json')
    device=torch.device('cuda:0')
    model=RecurrentSACCQL(signature['feature_dim'],signature['action_count'],cfg['training']).to(device)
    model.load_state_dict(parent['model']);optimizers=model.make_optimizers()
    write(output/'config.json',cfg)
    write(output/'dataset.json',dict(pool_sizes=[len(p) for p in pools],sampling=[16,8,8]))
    rng=np.random.default_rng(TRAIN_SEED+123456)
    probe=[]
    for pool,n in zip(pools,[16,8,8]): probe.extend(pool[int(i)] for i in rng.integers(len(pool),size=n))
    batch_probe={k:v.to(device) for k,v in collate_episodes(probe).items()}
    start_epoch=0
    if (output/'latest.pt').exists():
        state=torch.load(output/'latest.pt',map_location=device,weights_only=False)
        assert state['config']==cfg
        model.load_state_dict(state['model']);start_epoch=state['epoch']
        for k,opt in optimizers.items():opt.load_state_dict(state['optimizers'][k])
        # Discard records newer than the last committed checkpoint on resume.
        for filename in ['metrics.jsonl','diagnostics.jsonl']:
            path=output/filename
            if path.exists():
                rows=[json.loads(line) for line in path.read_text().splitlines()]
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['epoch']<=start_epoch))
    else:
        write(output/'initial_validation.json',validate(model,val,device))
        append(output/'diagnostics.jsonl',dict(epoch=0,**diagnostics(model,batch_probe,gamma is None)))
    for epoch in range(start_epoch+1,41):
        model.train();rng=np.random.default_rng(TRAIN_SEED+epoch);totals={};count=0;started=time.monotonic()
        for _ in range(186):
            episodes=[]
            for pool,n in zip(pools,[16,8,8]):episodes.extend(pool[int(i)] for i in rng.integers(len(pool),size=n))
            batch={k:v.to(device) for k,v in collate_episodes(episodes).items()}
            metrics=pure_bc_update(model,batch,optimizers['actor']) if gamma is None else model.update(batch,optimizers)
            n=int(batch['valid'].sum());count+=n
            for key,v in metrics.items():totals[key]=totals.get(key,0.)+v*n
        record=dict(epoch=epoch,train={k:v/count for k,v in totals.items()},val=validate(model,val,device),
                    transitions=count,seconds=time.monotonic()-started)
        diag=dict(epoch=epoch,**diagnostics(model,batch_probe,gamma is None))
        state=dict(epoch=epoch,model=model.state_dict(),signature=signature,config=cfg,validation=record['val'],
            optimizers={k:v.state_dict() for k,v in optimizers.items()},initialization_checkpoint_sha256=sha(PARENT))
        save_checkpoint(output/'latest.pt',state)
        append(output/'metrics.jsonl',record);append(output/'diagnostics.jsonl',diag)
        print(json.dumps(dict(name=name,**record)),flush=True)
    if gamma is None:
        unchanged=all(torch.equal(v.cpu(),parent['model'][k]) for k,v in model.state_dict().items() if not k.startswith('actor.'))
        assert unchanged
        write(output/'pure_bc_audit.json',dict(critic_target_alpha_unchanged=True,reward_used_in_training=False))
    write(folder/'evaluation_checkpoint.json',dict(checkpoint=str(output/'latest.pt'),sha256=sha(output/'latest.pt'),epoch=40))
    write(output/'completed.json',dict(completed_at=now(),epochs=40))


def import_evaluation(name, old, split):
    dest=OUT/name/split
    if dest.exists(): return
    shutil.copytree(SOURCE/old/split,dest)
    result=json.loads((dest/'results.json').read_text());result.update(name=name,reused_from=str(SOURCE/old/split))
    write(dest/'results.json',result)
    write(dest/'reuse_provenance.json',dict(source=str(SOURCE/old/split),reason='same policy tensors and fixed evaluation inputs',
                                         imported_at=now(),original_episode_logs_preserved=True))


def main():
    prepare()
    for name,gamma in ARMS:
        write(OUT/'status.json',dict(stage='training',experiment=name,updated_at=now()))
        command_run([PYTHON,'-m','experiment.avwan.gamma_bc_ablation','--train',name],OUT,name+'_train')
    # Verify exact reproduction before spending another 300 simulator launches.
    reference=SOURCE/'exp005-recovery-failed-stop-penalty1/training/latest.pt'
    current=OUT/ARMS[2][0]/'training/latest.pt'
    old=torch.load(reference,map_location='cpu',weights_only=False)['model']
    new=torch.load(current,map_location='cpu',weights_only=False)['model']
    equal=set(old)==set(new) and all(torch.equal(old[k],new[k]) for k in old)
    write(OUT/'gamma0p01_reproduction.json',dict(exactly_equal=equal,all_state_tensors=len(old),
        historical_checkpoint=str(reference),new_checkpoint=str(current),
        historical_sha256=sha(reference),new_sha256=sha(current)))
    if equal:
        write(OUT/ARMS[2][0]/'evaluation_checkpoint.json',dict(checkpoint=str(reference),sha256=sha(reference),epoch=40,
            reason='historical checkpoint has exactly identical model state; reuse audited rollouts'))
    import experiment.avwan.evaluate_stop_study as evaluator
    evaluator.OUT=OUT
    for split in SPLITS:
        for name,gamma in ARMS:
            write(OUT/'status.json',dict(stage='evaluation',experiment=name,split=split,updated_at=now()))
            if gamma==.01 and equal:import_evaluation(name,'exp005-recovery-failed-stop-penalty1',split)
            else:print(json.dumps(evaluator.evaluate(name,split)),flush=True)
        if split==SPLITS[0]:
            rows=[json.loads((OUT/name/split/'results.json').read_text()) for name,_ in ARMS]
            best=max(rows,key=lambda r:r['spl'])
            selection=dict(selected=best['name'],validation_spl=best['spl'],criterion='independent validation navigation SPL',
                           decided_at=now(),new_test_results_used=False,previous_test_results_known=True)
            if (OUT/'selection.json').exists():assert json.loads((OUT/'selection.json').read_text())['selected']==best['name']
            else:write(OUT/'selection.json',selection)
    for name,_ in ARMS:
        path=json.loads((OUT/name/'evaluation_checkpoint.json').read_text())['checkpoint']
        if not (OUT/name/'memory_audit.json').exists():
            command_run([PYTHON,'-m','experiment.avwan.audit_offline_memory','--checkpoint',path,
                '--cache',str(DATA/'frozen_ratio_features'),'--output',str(OUT/name/'memory_audit.json')],OUT/name,'memory_audit')
    report()


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[];audit={};paired={};rng=np.random.default_rng(20261004)
    for name,gamma in ARMS:
        row=dict(experiment=name,objective='BC' if gamma is None else 'SAC-CQL+BC',gamma=gamma)
        audit[name]={};paired[name]={}
        for split,label in zip(SPLITS,['val','heard','unheard']):
            result=json.loads((OUT/name/split/'results.json').read_text())
            for key in ['spl','success_rate','false_stop','timeout','collisions']:row[label+'_'+key]=result[key]
            records=[json.loads(line) for line in (OUT/name/split/'episodes.jsonl').read_text().splitlines()]
            base=[json.loads(line) for line in (SOURCE/'exp001-parent-baseline'/split/'episodes.jsonl').read_text().splitlines()]
            lookup={(r['scene'],r['episode_id']):r for r in base}
            assert len(records)==len(lookup)==100
            matches=sum(r['initial_observation_hashes']==lookup[(r['scene'],r['episode_id'])]['initial_observation_hashes'] for r in records)
            audit[name][split]=dict(initial_inputs_match=matches,total=100)
            assert matches==100
            delta=np.array([r['spl']-lookup[(r['scene'],r['episode_id'])]['spl'] for r in records])
            interval=np.quantile(delta[rng.integers(100,size=(20000,100))].mean(1),[.025,.975]).tolist()
            paired[name][split]=dict(delta_vs_parent=float(delta.mean()),paired_bootstrap95=interval)
        rows.append(row)
    with (OUT/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(OUT/'comparison.json',rows);write(OUT/'initial_observation_hash_audit.json',audit)
    write(OUT/'paired_comparison.json',paired)
    # Direct comparisons against pure BC and gamma .01 answer the ablation question.
    direct={}
    for split in SPLITS:
        data={name:[json.loads(line) for line in (OUT/name/split/'episodes.jsonl').read_text().splitlines()] for name,_ in ARMS}
        direct[split]={}
        for reference in [ARMS[0][0],ARMS[2][0]]:
            lookup={(r['scene'],r['episode_id']):r for r in data[reference]}
            direct[split][reference]={}
            for name,_ in ARMS:
                d=np.array([r['spl']-lookup[(r['scene'],r['episode_id'])]['spl'] for r in data[name]])
                direct[split][reference][name]=dict(delta=float(d.mean()),paired_bootstrap95=np.quantile(
                    d[rng.integers(100,size=(20000,100))].mean(1),[.025,.975]).tolist())
    write(OUT/'paired_vs_bc_and_gamma0p01.json',direct)
    diag={name:[json.loads(line) for line in (OUT/name/'training/diagnostics.jsonl').read_text().splitlines()] for name,_ in ARMS}
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    x=np.arange(len(rows));labels=['BC','g=0','g=.01','g=.9','g=.99']
    for j,(key,label) in enumerate([('val_spl','Validation'),('heard_spl','Heard'),('unheard_spl','Unheard')]):
        axes[0].bar(x+(j-1)*.25,[r[key] for r in rows],width=.25,label=label)
    axes[0].set_xticks(x,labels);axes[0].set_ylabel('SPL');axes[0].legend()
    for (name,gamma),label in zip(ARMS,labels):
        d=diag[name]
        if gamma is not None:
            axes[1].plot([r['epoch'] for r in d],[r['rl_to_bc_grad_ratio'] for r in d],label=label)
            axes[2].plot([r['epoch'] for r in d],[r['q_data_mean'] for r in d],label=label)
    axes[1].set_ylabel('RL / weighted BC gradient norm');axes[1].set_xlabel('Epoch');axes[1].legend()
    axes[2].set_ylabel('Q(data action), fixed training batch');axes[2].set_xlabel('Epoch');axes[2].legend()
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    selected=json.loads((OUT/'selection.json').read_text())['selected']
    checkpoint=Path(json.loads((OUT/selected/'evaluation_checkpoint.json').read_text())['checkpoint'])
    shutil.copyfile(checkpoint,OUT/'best_policy.pt');shutil.copyfile(FOUNDATION,OUT/'best_foundation.pt')
    lines=['# Gamma 与纯 BC 实验结果','',
        '固定 exp005 数据、失败 STOP 惩罚 1、BC 权重 5、旧 parent 初始化；每组继续训练 40×186 批。纯 BC 不更新 Critic/alpha/target。',
        'gamma 是高层决策折扣。所有候选使用固定 epoch40；按独立验证集选模型。测试沿用历史固定100，属于探索性复用。','',
        '| 实验 | 验证 SPL | heard SPL | unheard SPL | unheard 成功 | 错误 STOP | 超时 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['experiment']} | {r['val_spl']:.4f} | {r['heard_spl']:.4f} | {r['unheard_spl']:.4f} | {r['unheard_success_rate']*100:.0f}/100 | {r['unheard_false_stop']} | {r['unheard_timeout']} |")
    lines.extend(['',f'独立验证选择并导出：**{selected}**。','',
        '## 固定训练 batch 的最终梯度与 Q 诊断','',
        '| 实验 | 加权 BC 梯度范数 | RL/BC 梯度比 | Q 均值 | TD RMSE |',
        '|---|---:|---:|---:|---:|'])
    for name,_ in ARMS:
        d=diag[name][-1]
        lines.append(f"| {name} | {d['weighted_bc_grad_norm']:.4f} | {d['rl_to_bc_grad_ratio']:.4f} | {d.get('q_data_mean',0):.4f} | {d.get('td_rmse',0):.4f} |")
    lines.extend(['','纯 BC 表中的 Q/TD 为不适用（显示 0），Critic 未训练。梯度为裁剪前的固定训练 batch 诊断，不能以单个 batch 推断所有状态。',
        'Q 与已观察行为回报的差异不等于真实最优 Q 的误差：SAC 含熵项，行为动作也不等于当前策略。','',
        '## 可复现性与解释边界','',
        '- 每组每个 split 固定100条；所有初始输入哈希与 parent 的对应 episode 一致。',
        '- `gamma0p01_reproduction.json` 记录新训 .01 与历史 exp005 的逐张量比较及是否复用评估。',
        '- `paired_vs_bc_and_gamma0p01.json` 保存对纯 BC 与 gamma=.01 的配对差值和描述性 bootstrap 95% 区间；未做多重比较修正。',
        '- 所有组共享过去经过 RL 训练的 parent；纯 BC 仅代表本轮继续训练目标，不代表从零 BC。高 gamma 也继承低 gamma 的 Critic，结论限于相同步数的继续训练。',
        '- 固定 BC=5 的对照不能证明其他 BC 权重的最优值；梯度记录用于判断当前权重下各项是否主导。',
        '- 奖励为已有高层转移的奖励；本轮不重新定义 primitive-step 奖励或 SMDP 折扣。','',
        '逐 episode 轨迹、训练日志、配置、检查点、LSTM 审计均保存在各实验子目录。'])
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    annotations=json.loads((DATA/'training_annotations.json').read_text())
    unchanged=all(sha(Path(r['path']))==r['raw_sha256'] for r in annotations);assert unchanged
    memories={name:json.loads((OUT/name/'memory_audit.json').read_text())['memory_is_functional'] for name,_ in ARMS}
    assert all(memories.values())
    write(OUT/'final_audit.json',dict(raw_data_unchanged=unchanged,initial_inputs_all_match=True,
        memory_functional=memories,
        evaluated_episodes=1500,selected=selected,exported_checkpoint_sha256=sha(OUT/'best_policy.pt'),
        evaluation_checkpoint_sha256=sha(checkpoint),completed_at=now()))
    write(OUT/'status.json',dict(stage='complete',updated_at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train');p.add_argument('--prepare',action='store_true');p.add_argument('--report',action='store_true')
    args=p.parse_args()
    if args.prepare:prepare()
    elif args.train:train(*next(a for a in ARMS if a[0]==args.train))
    elif args.report:report()
    else:main()
