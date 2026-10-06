"""Audit and report the completed STOP-negative/recovery study without test tuning."""
import csv
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
from .stop_study import OUT, DATA, EXPERIMENTS, SPLITS, FOUNDATION, PARENT
from .run_gamma_study import ROOT, write, sha, now
from .evaluate_stop_study import checkpoint_for


def report():
    rows=[];episodes={};failure_mechanisms=[]
    checkpoint_hashes={name:sha(checkpoint_for(name)) for name,_,_ in EXPERIMENTS}
    foundation_hash=sha(FOUNDATION)
    for name,_,_ in EXPERIMENTS:
        for split in SPLITS:
            result=json.loads((OUT/name/split/'results.json').read_text())
            samples=[json.loads(line) for line in (OUT/name/split/'episodes.jsonl').read_text().splitlines()]
            assert len(samples)==100 and result['episodes']==100
            assert len({(r['scene'],r['episode_id']) for r in samples})==100
            assert all(r['checkpoint_sha256']==checkpoint_hashes[name] for r in samples)
            assert all(r['foundation_sha256']==foundation_hash for r in samples)
            assert all(np.isfinite(r['spl']) and 0<=r['spl']<=1 for r in samples)
            assert abs(np.mean([r['spl'] for r in samples])-result['spl'])<1e-10
            for sample in samples:
                sample['failed_within_macro_overshoot']=False
                sample['timeout_at_goal_without_stop']=bool(sample['timeout'] and sample['final_distance']<1e-6)
                if sample['visited_goal'] and not sample['success']:
                    trace=OUT/name/split/f"episode_{sample['global_index']:03d}"/'primitive_steps.jsonl'
                    groups={}
                    for line in trace.read_text().splitlines():
                        event=json.loads(line);groups.setdefault(event['macro'],[]).append(event)
                    overshot=[macro for macro,events in groups.items() if any(e['at_goal'] for e in events) and not events[-1]['at_goal']]
                    sample['failed_within_macro_overshoot']=bool(overshot)
                    failure_mechanisms.append(dict(name=name,split=split,global_index=sample['global_index'],
                        scene=sample['scene'],episode_id=sample['episode_id'],overshot_macros=overshot,
                        goal_at_decision=sample['goal_at_decision'],timeout_at_goal=sample['timeout_at_goal_without_stop'],
                        primitive_trace=str(trace)))
            result['failed_after_visiting_goal']=sum(r['visited_goal'] and not r['success'] for r in samples)
            result['failed_after_goal_decision']=sum(r['goal_at_decision'] and not r['success'] for r in samples)
            for key in ['failed_within_macro_overshoot','timeout_at_goal_without_stop']:
                result[key]=sum(r[key] for r in samples)
            episodes[name,split]=samples
            rows.append(result)
    selected=max((r for r in rows if r['split']=='val_multiple_unheard'),key=lambda r:r['spl'])['name']
    highest_test=max((r for r in rows if r['split']=='test_multiple_unheard'),key=lambda r:r['spl'])
    write(OUT/'failure_mechanism_audit.json',failure_mechanisms)
    validation_selection=json.loads((OUT/'navigation_validation_selection.json').read_text())
    assert validation_selection['selected']==selected
    for name,_,_ in EXPERIMENTS[1:]:
        for split in SPLITS[1:]:
            assert all(r['completed_at']>=validation_selection['decided_at'] for r in episodes[name,split])
    import torch
    expert_state=torch.load(checkpoint_for('exp006-penalty1-expert-only-bc'),map_location='cpu',weights_only=False)
    assert expert_state['signature']['expert_bc_annotation_sha256']==sha(DATA/'expert_bc_annotations.json')
    parent_weights=torch.load(PARENT,map_location='cpu',weights_only=False)['model']
    selected_weights=torch.load(checkpoint_for(selected),map_location='cpu',weights_only=False)['model']
    identical_to_parent=all(torch.equal(value,selected_weights[key]) for key,value in parent_weights.items())
    fields=['name','split','episodes','spl','success_rate','false_stop','correct_stop','timeout','collisions',
            'visited_goal','goal_at_decision','missed_goal_inside_macro','failed_after_visiting_goal',
            'failed_after_goal_decision','failed_within_macro_overshoot','timeout_at_goal_without_stop','checkpoint_sha256']
    with (OUT/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    baseline=EXPERIMENTS[0][0];pairs=[];hash_audit=[];effects=[]
    rng=np.random.default_rng(20261004)
    for name,_,_ in EXPERIMENTS[1:]:
        for split in SPLITS:
            before=episodes[baseline,split];after=episodes[name,split]
            assert [(r['scene'],r['episode_id']) for r in before]==[(r['scene'],r['episode_id']) for r in after]
            matching={k:sum(a['initial_observation_hashes'][k]==b['initial_observation_hashes'][k] for a,b in zip(before,after))
                      for k in before[0]['initial_observation_hashes']}
            delta=np.array([b['spl']-a['spl'] for a,b in zip(before,after)])
            bootstrap=delta[rng.integers(100,size=(10000,100))].mean(1)
            gained=[(a,b) for a,b in zip(before,after) if not a['success'] and b['success']]
            lost=[(a,b) for a,b in zip(before,after) if a['success'] and not b['success']]
            retained=[(a,b) for a,b in zip(before,after) if a['success'] and b['success']]
            effect=dict(name=name,split=split,new_successes=len(gained),lost_successes=len(lost),
                retained_successes=len(retained),new_success_spl_gain=sum(b['spl'] for a,b in gained)/100,
                lost_success_spl_loss=-sum(a['spl'] for a,b in lost)/100,
                retained_success_efficiency_delta=sum(b['spl']-a['spl'] for a,b in retained)/100)
            assert abs(sum(effect[k] for k in ['new_success_spl_gain','lost_success_spl_loss','retained_success_efficiency_delta'])-delta.mean())<1e-10
            effects.append(effect)
            hash_audit.append(dict(name=name,split=split,matching_initial_hashes=matching,episodes=100,
                spl_delta=float(delta.mean()),descriptive_paired_bootstrap_95ci=np.quantile(bootstrap,[.025,.975]).tolist(),
                note='Single rollout per policy; mismatched acoustic inputs limit attribution. CI does not cover simulator repeat variability.'))
            for a,b in zip(before,after):
                pairs.append(dict(name=name,split=split,scene=a['scene'],episode_id=a['episode_id'],
                    baseline_spl=a['spl'],spl=b['spl'],delta=b['spl']-a['spl'],
                    baseline_success=a['success'],success=b['success'],
                    baseline_false_stop=a['false_stop'],false_stop=b['false_stop'],
                    baseline_timeout=a['timeout'],timeout=b['timeout'],
                    audio_initial_match=a['initial_observation_hashes']['spectrogram']==b['initial_observation_hashes']['spectrogram']))
    write(OUT/'initial_observation_hash_audit.json',hash_audit)
    write(OUT/'paired_effect_decomposition.json',effects)
    with (OUT/'paired_episodes.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    annotations=json.loads((DATA/'training_annotations.json').read_text())
    immutable=all(sha(Path(r['path']))==r['raw_sha256'] for r in annotations)
    assert immutable
    collection=json.loads((DATA/'collection_complete.json').read_text())
    policy_meta=[json.loads(p.read_text()) for p in (DATA/'policy/metadata').glob('*.json')]
    recovery_meta=[json.loads(p.read_text()) for p in (DATA/'recovery/metadata').glob('*.json')]
    bad=[r for r in policy_meta if r['false_stop']]
    write(OUT/'collection_analysis.json',dict(policy_episodes=len(policy_meta),negative_stop_count=len(bad),
        negative_stop_distance_quantiles=np.quantile([r['final_distance'] for r in bad],[0,.25,.5,.75,1]).tolist(),
        negative_stop_within_2m=sum(r['final_distance']<=2 for r in bad),
        negative_stop_first_decision=sum(r['transitions']==1 for r in bad),
        recovery_episodes=len(recovery_meta),recovery_successes=sum(r['success'] for r in recovery_meta),
        recovery_expert_intervened=sum(r['expert_steps']>0 for r in recovery_meta),
        recovery_episodes_with_premature_stop_intervention=sum(r['interventions']>0 for r in recovery_meta),
        recovery_successes_after_premature_stop_intervention=sum(r['interventions']>0 and r['success'] for r in recovery_meta),
        recovery_expert_blocked=sum(r['blocked'] is not None for r in recovery_meta)))
    write(OUT/'selection.json',dict(selected=selected,criterion='highest independent val_multiple_unheard navigation SPL',
        highest_observed_unheard_test_name=highest_test['name'],highest_observed_unheard_test_spl=highest_test['spl'],
        selected_checkpoint_sha256=sha(checkpoint_for(selected)),parent_sha256=sha(PARENT),
        selected_parameters_identical_to_parent=identical_to_parent,
        raw_data_hashes_unchanged=immutable,completed_at=now()))
    shutil.copy2(checkpoint_for(selected),OUT/'best_policy.pt');shutil.copy2(FOUNDATION,OUT/'best_foundation.pt')
    table=['| 配置 | val unheard SPL | test heard SPL | test unheard SPL | unheard 成功数 | 提前STOP | 超时 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    def get(name,split):return next(r for r in rows if r['name']==name and r['split']==split)
    for name,_,_ in EXPERIMENTS:
        val,heard,unheard=[get(name,s) for s in SPLITS]
        table.append(f"| {name} | {val['spl']:.4f} | {heard['spl']:.4f} | {unheard['spl']:.4f} | {unheard['success_rate']*100:.0f} | {unheard['false_stop']} | {unheard['timeout']} |")
    best=get(selected,'test_multiple_unheard');base=get(baseline,'test_multiple_unheard')
    selected_effect=next((r for r in effects if r['name']==selected and r['split']=='test_multiple_unheard'),None)
    effect_text=(f"逐episode配对分解：新增成功{selected_effect['new_successes']}条、丢失原有成功{selected_effect['lost_successes']}条；新增成功贡献SPL {selected_effect['new_success_spl_gain']:+.4f}，丢失成功贡献{selected_effect['lost_success_spl_loss']:+.4f}，双方都成功的轨迹效率变化贡献{selected_effect['retained_success_efficiency_delta']:+.4f}。这是观测差值的数学分解，不是额外的oracle实验。"
                 if selected_effect else '选中模型就是基线，没有模型变化的配对分解。')
    checks=[]
    for name,_,_ in EXPERIMENTS:
        m=json.loads((OUT/name/'memory_audit.json').read_text())
        assert m['memory_is_functional'], f'Memory audit failed: {name}'
        checks.append(f"- {name}: LSTM功能检查={m['memory_is_functional']}；history改变动作比例={m['history_action_change_fraction']:.2%}；流式最大误差={m['streaming_max_abs_error']:.2g}。")
    trained=[]
    for name,_,_ in EXPERIMENTS[1:]:
        m=json.loads((OUT/name/'training/best.json').read_text())
        evaluated=json.loads((OUT/name/'evaluation_checkpoint.json').read_text())
        trained.append(f"- {name}: 原验证代理规则选中epoch {m['epoch']}；实际导航评估epoch {evaluated['epoch']}，选择说明：{evaluated['selection']}。")
    amended=(OUT/'protocol_amendment_01.json').exists()
    minimum_hash_matches=min(r['matching_initial_hashes']['spectrogram'] for r in hash_audit)
    state_audit=json.loads((OUT/'identical_state_stop_audit.json').read_text())
    assert state_audit['parent_cached_replay_verified']
    state_table=['| 配置 | 150个已采集错误STOP状态上仍选STOP | 100个独立验证正确STOP状态上选STOP |',
                 '|---|---:|---:|']
    for record in state_audit['experiments']:
        groups=record['groups']
        state_table.append(f"| {record['name']} | {groups['training_actual_failed_stop_states']['stop_argmax']} | {groups['validation_expert_stop_states']['stop_argmax']} |")
    text=f'''# 提前 STOP 负例与专家恢复实验

完成时间：{now()}。{len(EXPERIMENTS)}组模型全部重新评估，每组独立验证、heard test、unheard test各100条固定episode，共{len(EXPERIMENTS)*300}次导航。测试没有专家接管或目标坐标输入。

依据独立unheard验证SPL选择 **{selected}**。它的unheard test SPL为 **{best['spl']:.4f}**，本轮重新测量的父模型为 **{base['spl']:.4f}**，差值 **{best['spl']-base['spl']:+.4f}**。0.45目标：**{'达到' if best['spl']>=.45 else '未达到'}**。不能将减少STOP但增加超时解释为问题已解决。

全部配置中最高的unheard test SPL是 **{highest_test['spl']:.4f}（{highest_test['name']}）**。这只是测试结果汇报，不改变预先按验证集选定的模型。本轮训练负例上的STOP修正不能直接视为unheard泛化问题已经解决。

{'选中模型参数与父模型完全相同，因此观测到的分数差属于重复评估波动，不能称为训练提升。' if identical_to_parent else '选中模型参数已经改变；训练收益仍需结合STOP、超时与初始观测一致性判断。'}

{chr(10).join(table)}

![SPL与停止结果对比](comparison.png)

## 采集与训练

数据目录：`{DATA}`。

```json
{json.dumps(collection['totals'],ensure_ascii=False,indent=2)}
```

保留模型实际执行的失败STOP终止转移，Critic使用真实动作与奖励；基础负例/恢复组的Actor BC屏蔽真实错误STOP。专家恢复是在这些训练episode上的独立重跑；声学状态可能变化，不能视作同一个状态的反事实分支。恢复数据记录实际执行的模型或专家动作，不改写已经结束的STOP转移。penalty1组仅在内存加载时对失败STOP奖励减1；全部原始文件哈希复核未改变。

训练采样池为5942条原数据、150条真实失败STOP轨迹、150条恢复轨迹。其余434条新策略轨迹保留用于数据审计和相同观测回放，未加入新增训练采样池。恢复文件中同时保留接管前的真实模型动作和接管后的专家动作；exp004/005仍对未失败的模型动作做BC，exp006才把新增数据的BC限制为专家动作。

五组的原始、负例、恢复轨迹采样比例与checkpoint规则见`PROTOCOL.md`和`protocol.json`。训练组均由相同父模型出发，gamma=.01、BC=5、学习率1e-4，40epoch，每epoch186批；foundation冻结。checkpoint包含epoch0候选，按独立验证balanced STOP NLL+0.1×动作NLL选择；最终配置按验证导航SPL选择，不按test分数选择。

追加exp006（见protocol_extension_02.json）：在exp005相同数据、奖励与训练步数下，新增数据的Actor BC只使用实际执行的专家动作；模型自身的移动轨迹仍进入Critic，但不再作为BC标签。原数据池保持不变。追加依据是完整验证结果改善有限，以及恢复组预期57.4%的BC标签来自模型自身轨迹；并未根据训练组测试分数进行调整。

{'规则修订：原四组按上述代理指标均选中epoch0，实际上都是父模型。为测量真实训练干预，主导航评估统一采用固定的最终epoch40，配置仍按独立验证导航SPL选择。该修订在任何新测试结果产生前记录于protocol_amendment_01.json；原协议、epoch0 checkpoint与全部训练日志保留，不把这一修订称为事先预注册。' if amended else '本次导航评估采用原定checkpoint规则。'}

{chr(10).join(trained)}

![训练验证代理指标曲线](training_curves.png)

各训练目录的`training/best_policy.pt`保留原代理指标选出的epoch0供审计；实际评估路径由`evaluation_checkpoint.json`记录，指向epoch40的`training/latest.pt`。本报告目录顶层的`best_policy.pt`是最终按验证导航SPL选出并实际测试的权重。

## STOP、碰撞与宏动作限制

在完全相同的缓存观测与完整历史上，父模型复现了全部150次原错误STOP，验证了采集到特征再到循环策略的回放一致性：

{chr(10).join(state_table)}

前一列衡量训练负例拟合，不能当作unheard泛化性能；后一列只衡量独立专家轨迹上的正确STOP。负例屏蔽本身没有教会足够好的替代动作，专家恢复提供了关键的继续导航标签。实际泛化结论以本报告的闭环导航表为准。

选中模型的unheard提前STOP从{base['false_stop']}变为{best['false_stop']}，超时从{base['timeout']}变为{best['timeout']}，成功数从{base['success_rate']*100:.0f}变为{best['success_rate']*100:.0f}。逐episode对照见`paired_episodes.csv`；每条运行的宏动作和primitive轨迹均保存在对应配置/划分/episode目录。

{effect_text} 详细数据见`paired_effect_decomposition.json`。

本实验保留prediction interval=5。选中模型有{best['missed_goal_inside_macro']}条轨迹在primitive层面到过目标，但从未在策略决策边界观测到目标。逐primitive复核还发现：{best['failed_within_macro_overshoot']}条失败轨迹包含同一宏动作内到达后又离开目标，{best['timeout_at_goal_without_stop']}条在目标处耗尽步数而没有执行STOP；这些诊断可以重叠，不能把所有“到过目标但未成功”都叫作宏动作越过目标。详情见`failure_mechanism_audit.json`。碰撞总数为{best['collisions']}。这些诊断读数不参与策略动作选择。

选中模型另有{best['failed_after_goal_decision']}条在决策边界位于目标、最终仍失败的轨迹（基线为{base['failed_after_goal_decision']}条）。这个指标用于检查抑制提前STOP是否同时损害正确停止。

## 复现与记忆模块核查

每个episode使用新进程，初始RGB/depth/声谱图/地图/action mask都记录哈希。与新基线比较，各配置/划分的声谱图匹配数最低为{minimum_hash_matches}/100。{'仍存在声学输入差异，因此单次分数差不能完全归因于训练；配对bootstrap区间仅是描述性统计，未包含重复模拟的方差。' if minimum_hash_matches<100 else '初始声谱图全部匹配；后续轨迹不同仍会产生不同观测。'}详情见`initial_observation_hash_audit.json`。旧研究分数不充当本轮相同条件下的基线。

{chr(10).join(checks)}

回归测试日志：`regression_tests.log`与包含新增BC来源屏蔽测试的`regression_tests_extension.log`（25项通过）。源码快照、代码差异、逐次命令、采集、特征、训练及评估日志均保存在本目录。数据来源、选择清单和哈希见`protocol.json`、`data_audit.json`以及数据目录中的`training_annotations.json`、`expert_bc_annotations.json`、`frozen_ratio_features/manifest.json`。

选中权重：`best_policy.pt`与`best_foundation.pt`。重现命令见`REPRODUCE.md`。
'''
    (OUT/'RESULTS.md').write_text(text)
    (OUT/'REPRODUCE.md').write_text('''# 运行命令

在仓库根目录、ss环境执行。协调器有完成标记检查，不覆盖已经完整的原始数据和结果。

```bash
PYTHONPATH=. /home/getuanhui/anaconda3/envs/ss/bin/python -m experiment.avwan.collect_stop_data
PYTHONPATH=. /home/getuanhui/anaconda3/envs/ss/bin/python -m experiment.avwan.run_stop_study
PYTHONPATH=. /home/getuanhui/anaconda3/envs/ss/bin/python -m experiment.avwan.report_stop_study
```

固定选择清单位于selections，每条episode的确切命令位于各配置/划分/episode_XXX/evaluation_command.json。环境基础配置来自evaluation.yaml及其BASE_TASK_CONFIG_PATH引用；所有配置文件另存于source_snapshot。
''')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    x=np.arange(len(EXPERIMENTS))
    for j,split in enumerate(SPLITS):
        axes[0].bar(x+(j-1)*.24,[get(n,split)['spl'] for n,_,_ in EXPERIMENTS],.24,label=split)
    axes[0].axhline(.45,color='gray',ls='--');axes[0].set_ylabel('SPL');axes[0].legend(fontsize=8)
    for j,key in enumerate(['false_stop','timeout']):
        axes[1].bar(x+(j-.5)*.3,[get(n,'test_multiple_unheard')[key] for n,_,_ in EXPERIMENTS],.3,label=key)
    axes[1].set_ylabel('Unheard episodes / 100');axes[1].legend()
    for ax in axes:ax.set_xticks(x,[n.split('-')[0] for n,_,_ in EXPERIMENTS])
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for name,_,_ in EXPERIMENTS[1:]:
        metrics=[json.loads(l) for l in (OUT/name/'training/metrics.jsonl').read_text().splitlines()]
        ax.plot([r['epoch'] for r in metrics],[r['val']['selection_score'] for r in metrics],label=name.split('-')[0])
    ax.set(xlabel='Epoch',ylabel='Validation balanced STOP NLL + 0.1 action NLL');ax.legend();fig.tight_layout()
    fig.savefig(OUT/'training_curves.png',dpi=180);plt.close(fig)
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for path in (ROOT/'experiment/avwan').rglob('*'):
        if path.suffix not in {'.py','.yaml','.yml','.md'}:continue
        if any(part in {'runs','data_cache','__pycache__'} for part in path.parts):continue
        dest=snapshot/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    import yaml
    config=yaml.safe_load((OUT/'evaluation.yaml').read_text())
    source=Path(config['BASE_TASK_CONFIG_PATH'])
    shutil.copy2(source,snapshot/'base_task_config.yaml')
    (OUT/'git_diff.patch').write_text(subprocess.check_output(['git','diff'],cwd=ROOT,text=True))
    write(OUT/'source_hashes.json',{str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*') if p.is_file()})
    write(OUT/'final_audit.json',dict(passed=True,evaluations=len(EXPERIMENTS)*300,raw_data_hashes_unchanged=True,
        selected_unheard_spl=best['spl'],selected_meets_045=best['spl']>=.45,
        memory_checks_passed=True,parent_cached_failed_stop_replay_verified=True,
        independent_validation_selection=True,selected=selected,tests_log_sha256=sha(OUT/'regression_tests_extension.log'),completed_at=now()))
    write(OUT/'status.json',dict(stage='complete',selected=selected,updated_at=now()))
    print(text)


if __name__=='__main__':report()
