"""Paths and predeclared protocol for the premature-STOP data study."""
import gzip
import json
import random
import shutil
from pathlib import Path

from .run_gamma_study import ROOT, sha, write, now

OUT = ROOT/'result/2026-10-04-stop-negative-recovery-study'
DATA = Path('/data/Getuanhui/avwan_stop_negative_recovery_20261004')
PREVIOUS = ROOT/'result/2026-10-03-unheard-generalization-study'
PYTHON = '/home/getuanhui/anaconda3/envs/ss/bin/python'
PARENT = PREVIOUS/'exp013-ratio-sac-g0p01-prior-bc5/training/best_policy.pt'
FOUNDATION = PREVIOUS/'best_foundation.pt'
BASE_CACHE = PREVIOUS/'ratio_memory_features'
TRAIN_SEED = 20261004
EVAL_SEED = 20260930
SPLITS = ['val_multiple_unheard', 'test_multiple', 'test_multiple_unheard']
EXPERIMENTS = [
    ('exp001-parent-baseline', 'baseline', 0.),
    ('exp002-original-data-continuation', 'original', 0.),
    ('exp003-negative-stop-bc-masked', 'negative', 0.),
    ('exp004-negative-stop-and-expert-recovery', 'recovery', 0.),
    ('exp005-recovery-failed-stop-penalty1', 'recovery', 1.),
    ('exp006-penalty1-expert-only-bc', 'recovery_expert_bc', 1.),
]


def prepare():
    OUT.mkdir(parents=True, exist_ok=True);DATA.mkdir(parents=True, exist_ok=True)
    if (OUT/'protocol.json').exists():
        return json.loads((DATA/'selection.json').read_text())
    source=Path('/data/Getuanhui/avwan_multiple_greedy_train/episodes.jsonl')
    groups={}
    for line in source.read_text().splitlines():
        row=json.loads(line)
        if row.get('saved'):groups.setdefault(row['sound'],[]).append(row)
    chosen=[];rng=random.Random(TRAIN_SEED)
    for sound,pool in sorted(groups.items()):
        pool.sort(key=lambda r:(r['scene'],r['initial_distance'],r['episode_id']))
        n=min(8,len(pool));offset=rng.random()
        for j in range(n):
            r=dict(pool[int((j+offset)*len(pool)/n)])
            r['collection_stage']=j%2;chosen.append(r)
    chosen.sort(key=lambda r:(r['scene'],str(r['episode_id'])))
    for i,r in enumerate(chosen):r['global_index']=i
    keys={(r['scene'],str(r['episode_id'])) for r in chosen};assert len(keys)==len(chosen)
    audit=json.loads((PREVIOUS/'split_audit.json').read_text())
    for split in ['val_multiple_unheard','test_multiple','test_multiple_unheard']:
        assert not {r['scene'] for r in chosen}&set(audit[split]['scenes'])
    for split in ['val_multiple_unheard','test_multiple_unheard']:
        assert not set(groups)&set(audit[split]['sound_classes'])
    episodes=[]
    for scene in sorted({r['scene'] for r in chosen}):
        paths=[p for p in (ROOT/'data/datasets/audionav/mp3d/v1/train_multiple/content').glob(scene+'.json*') if p.name.endswith(('.json','.json.gz'))]
        assert len(paths)==1
        path=paths[0]
        with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:raw=json.load(f)
        episodes.extend(e for e in raw['episodes'] if (Path(e['scene_id']).stem,str(e['episode_id'])) in keys)
    assert len(episodes)==len(chosen)
    write(DATA/'dataset.json',dict(episodes=episodes))
    selection=dict(episodes=chosen,seed=TRAIN_SEED,source=str(source),source_sha256=sha(source),
                   method='up to eight systematically spaced episodes per training sound; alternating entries form two stages',
                   expand_stage_1_if_negative_stops_below=100,training_only=True,split_audit_passed=True)
    write(DATA/'selection.json',selection)
    for split in SPLITS:
        shutil.copytree(PREVIOUS/'selections'/split,OUT/'selections'/split,dirs_exist_ok=True)
    shutil.copy2(PREVIOUS/'evaluation.yaml',OUT/'evaluation.yaml')
    protocol=dict(started_at=now(),data=str(DATA),parent=str(PARENT),parent_sha256=sha(PARENT),
        foundation=str(FOUNDATION),foundation_sha256=sha(FOUNDATION),experiments=EXPERIMENTS,
        gamma=.01,bc_weight=5.,learning_rate=1e-4,epochs=40,steps_per_epoch=186,batch_size=32,
        initialization='all trained arms start from the same parent weights, fresh optimizers; epoch0 retained',
        sampling=dict(original=[32,0,0],negative=[24,8,0],recovery=[16,8,8]),
        sampling_order=['original','actual_failed_STOP_episodes','recovery_episodes'],
        checkpoint_selection='minimum independent validation balanced STOP NLL + 0.1 * full-action NLL, including epoch0',
        configuration_selection='maximum independent unheard validation navigation SPL',
        failed_stop_bc_weight=0.,raw_rewards_immutable=True,
        penalty_variant='subtract 1 only from training reward of actual failed STOP; raw reward retained',
        evaluation=dict(episodes_per_split=100,seed=EVAL_SEED,primitive_step_limit=200,prediction_interval=5,
                        fresh_process_per_episode=True,initial_observation_hash_audit=True),
        training_seed=TRAIN_SEED,expert_use='training recovery only; never validation/test policy inputs')
    write(OUT/'protocol.json',protocol);write(DATA/'protocol.json',protocol)
    (OUT/'PROTOCOL.md').write_text('''# 提前STOP负例与专家恢复实验

从上一轮独立验证选出的exp013双耳比值SAC-CQL权重出发。数据目录见protocol.json，所有新原始轨迹和特征缓存均在/data/Getuanhui/avwan_stop_negative_recovery_20261004。

五组：原模型重新评估；仅原数据继续训练；添加真实失败STOP轨迹并屏蔽错误STOP的BC；进一步添加专家恢复轨迹；在完整数据上额外减去1的失败STOP训练奖励。其他参数固定gamma=.01、BC=5、学习率1e-4。每组训练40epoch、每epoch186批，保留epoch0。

先采集每个训练声音最多4条模型轨迹；若真实失败STOP少于100条，再采集预先选好的第二阶段。针对失败STOP的episode重新采集恢复轨迹：策略首次准备在非目标处STOP时交给专家继续导航；目标处由专家提供正确STOP。两次运行不假装是完全相同的状态分支。

原始action/reward/next_state/terminated不改写。Critic使用真实行为动作；Actor只屏蔽真实错误STOP的BC。专家恢复记录实际执行动作。惩罚组只在训练加载时单独变换奖励，不修改采集文件。验证和测试没有专家动作或目标位置输入策略。

采样每批32条轨迹：原数据组32原始；负例组24原始+8失败STOP；恢复组16原始+8失败STOP+8恢复。保存采样与有效BC计数。内部checkpoint按独立验证balanced STOP NLL+0.1×动作NLL选择；配置按独立unheard验证导航SPL选择。

沿用固定heard/unheard测试各100条与独立unheard验证100条，seed=20260930、200个primitive steps、prediction interval=5。每条episode新进程，减少跨episode声学状态影响；记录初始输入哈希核对一致性。原模型重新评估，不把此前不同执行条件的分数当本轮基线。

指标包括SPL、成功率、错误STOP、超时、到达却未停止、碰撞。提前STOP负例不能自动解决宏动作途中错过目标的问题，本轮不改变动作间隔以隔离数据处理效果。
''')
    return selection
