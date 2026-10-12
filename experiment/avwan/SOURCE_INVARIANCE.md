# AVWAN：声音种类不敏感的 foundation 与 offline RL 复现

目标是**同一份** foundation 和 SAC-CQL actor 在完整的 `test_multiple` 与
`test_multiple_unheard` 各 1000 条 episode 上均达到 SPL > 0.4。任何验证轨迹上的
动作准确率、100 条闭环验证或声音扰动稳定性都不能替代这个判定。

`unheard` 是评测划分：完整测试含 18 种训练中未出现的声音，训练集有 73 种声音；
它本身不是一种实现“不敏感”的模型方法。当前音频前端先计算逐频点双耳比
`(L-R)/(L+R)`，双耳共同的音量或频谱缩放会在非静音频点抵消。双视图训练再要求
同一导航观测在合成音色扰动前后给出一致动作。两者减弱某些音色影响，但声源
频带缺失、房间反射和噪声仍可能改变可用方向信息，因此必须用闭环 SPL 判断。

## 方法

本轮测试两种音频前端。较保守的候选保留已训练的 `binaural_ratio`，从已有 ratio
foundation 初始化，在原始声音和随机共享双耳频谱染色/频带缺失的两种视图上训练。
损失包含两种视图的专家动作和声源角度交叉熵、动作分布双向 KL、融合特征余弦
一致性。视觉和地图编码器冻结；音频编码器、融合层及动作头可训练。

另一个前端 `binaural_robust_ratio` 用相对于整段能量的静音下限抑制近静音频点的
噪声，并提供局部平滑后的双耳差值。它的声音扰动稳定性提高，但搭配 8 epoch
actor 的 100 条闭环验证只有 heard 0.3192、unheard 0.2802，因此不是下面推荐
复现的候选，也没有完整测试成绩。

这是一种针对声音表征的模型和损失修改。测试时不使用专家接管、重访选择器或
依据声音类别写的动作规则；AVWAN 原有的动作 mask、mapper 和 waypoint 局部
planner 仍在使用。`reproduce_heard_unheard audit` 与完整测试报告负责核对权重、
训练/测试交集、评价配置和 episode 覆盖。

## 从冻结数据重训

在 `/home/getuanhui/project/hybrid_offline_RL` 运行。下列输出目录须是新目录，
不要复用已有特征缓存。训练使用 2026-10-11 固定的 40,275 条 train episode
快照；采集进程继续写入其原目录不会改变本次训练样本。
如果固定快照不在本机，先按 [OFFLINE_RL_HEARD_UNHEARD.md](OFFLINE_RL_HEARD_UNHEARD.md)
的“固定持续收集数据后训练”一节生成新快照，并为该快照重建 foundation 原始观测
缓存；下面的默认 `--cache` 指向本轮已构建的固定观测缓存。

```bash
PY=/home/getuanhui/anaconda3/envs/ss/bin/python
FOUND_RUN=result/my_source_consistency_ratio_foundation
FOUND=$FOUND_RUN/training/best.pt

$PY -m experiment.avwan.train_source_invariant_foundation \
  --output "$FOUND_RUN" --epochs 8 --seed 20261013 \
  --audio-normalization binaural_ratio --augmentation-strength 1.0 \
  --action-consistency-weight 0.25 --representation-consistency-weight 0.01

$PY -m experiment.avwan.eval_source_invariance \
  --checkpoint "$FOUND" --output "$FOUND_RUN/source_sensitivity.json"
```

`training/best.pt` 按 heard、unheard 独立专家验证的**较差动作 NLL**选择 epoch。
`eval_source_invariance` 测双耳共享频谱扰动前后的动作变化率与 KL；它只验证
声音稳定性，不报告导航 SPL。

```bash
BASE=result/my_source_invariant_base_features
CORRECTION=result/my_source_invariant_correction_features
VALIDATION=result/my_source_invariant_validation_features

$PY -m experiment.avwan.train_offline_rl --stage extract \
  --source result/2026-10-11-large-train-snapshot \
  --checkpoint "$FOUND" --cache "$BASE" --device cuda:0
$PY -m experiment.avwan.train_offline_rl --stage extract \
  --source /data/Getuanhui/avwan_stop_negative_recovery_20261004/feature_source \
  --checkpoint "$FOUND" --cache "$CORRECTION" --device cuda:0
$PY -m experiment.avwan.train_offline_rl --stage extract \
  --source result/2026-10-03-unheard-generalization-study/validation_demonstrations \
  --checkpoint "$FOUND" --cache "$VALIDATION" --device cuda:0

OUT=result/my_source_consistency_ratio_actor
NAME=ratio_consistency_8ep
$PY -m experiment.avwan.reproduce_heard_unheard train \
  --output "$OUT" --name "$NAME" --seed 20261011 \
  --foundation "$FOUND" --base-cache "$BASE" \
  --additional-cache "$CORRECTION" --validation-cache "$VALIDATION" \
  --steps-per-epoch 500 --epochs 8 --sampling 16 8 8
$PY -m experiment.avwan.reproduce_heard_unheard audit \
  --output "$OUT" --name "$NAME" --foundation "$FOUND"
$PY -m experiment.avwan.reproduce_heard_unheard validate \
  --output "$OUT" --name "$NAME" --foundation "$FOUND" \
  --validation-split val_multiple --validation-episodes 100
$PY -m experiment.avwan.reproduce_heard_unheard validate \
  --output "$OUT" --name "$NAME" --foundation "$FOUND" \
  --validation-split val_multiple_unheard --validation-episodes 100
```

完整测试对两组各执行两段 500 条分片。每条命令可以中断后原样继续执行。

```bash
for split in test_multiple test_multiple_unheard; do
  for shard in 0 1; do
    $PY -m experiment.avwan.reproduce_heard_unheard evaluate \
      --output "$OUT" --name "$NAME" --foundation "$FOUND" \
      --splits "$split" --shards 2 --shard-index "$shard" \
      --episodes-per-process 20 --max-rss-mb 7000
  done
done
$PY -m experiment.avwan.reproduce_heard_unheard evaluate \
  --output "$OUT" --name "$NAME" --foundation "$FOUND" \
  --splits test_multiple test_multiple_unheard --shards 2
```

只有 `full_test/report.json` 确认每组 1000 条唯一 episode、零错误、同一组权重
哈希，且两组平均 SPL 均严格大于 0.4，才算达到目标。

## 2026-10-12 实际验证

按上面保守方法训练的 foundation 选中第 2 epoch，SHA-256 为
`7788c43ee9c356d7d800eac6a89613d956e6113476907c7632dcb5b8e80b05d5`。
相同合成声音扰动下，旧 foundation 的动作变化率为 unheard 20.0%、heard 20.5%；
新 foundation 分别为 10.6%、11.8%。这些只是独立专家轨迹上的声音稳定性指标。

8 epoch actor（SHA-256
`71b0310816a9deaea9645f0c6bd3790ebb1df7d9818234a18ea4cd2fa694f565`）
100 条闭环验证的 heard/unheard SPL 为 0.4526/0.4022。20 epoch actor 的结果为
0.4435/0.3647；按较低 SPL 选中 8 epoch，证据见
`result/2026-10-12-source-consistency-ratio-actors/validation_selection.json`。
选中模型的两组完整 1000 条测试已完成，零错误、权重哈希一致：heard SPL
**0.4048**，unheard SPL **0.3277**，因此**没有达到双任务目标**。正式报告在
`result/2026-10-12-source-consistency-ratio-actors/full_test/report.json`。

查看该候选部分测试结果后，另以种子 `20261012` 训练了 8 epoch actor：100 条验证
heard/unheard 为 0.4036/0.4197，两组较低值比预先选中的 actor 只高 0.0014。
这是探索性追加候选，**没有替换预先选中的全量测试模型**，也没有它自己的完整测试
成绩。时间线和哈希见
`result/2026-10-12-source-consistency-ratio-seed12-actor/validation_extension.json`。

## 第三版：声音类别对抗损失

`build_sound_labels.py` 从冻结轨迹索引生成训练样本的声音类别旁路缓存，并检查
173,572 个训练样本与 episode 边界对齐。训练时将音频空间表征送入声音类别判别器；
反向梯度让空间表征难以预测已见声音类别，同时保留专家动作和角度监督。判别器不
进入部署 checkpoint 的模型，也不参与测试动作。梯度反转系数为 0.15，分类
交叉熵权重为 0.3；视图动作 KL 权重降为 0.1，频谱增强强度为 0.8。
声音标签来自冻结训练轨迹，覆盖 40,275 条 episode、173,572 个样本、73 个
训练声音类别；标签变化经过 episode 边界检查。

本轮 foundation 第 1 epoch 按较差动作 NLL 选中，SHA-256 为
`3bfdcd142b01dca9530f16701f5147c99949ca7863359e6d9d688ea5e0ab4186`。
独立专家轨迹上的合成音色扰动动作变化率为 unheard 10.25%、heard 12.08%，
与第二版相近。这不能证明真实 unheard 导航性能提高。8 epoch actor 的
SHA-256 为 `671dbf82917644debbcefac7e13e340ba7d2925c00d2bbb8196e2ffd50912bb5`。
审计文件在 `result/2026-10-12-sound-adversarial-actor/audit.json`。
同一组权重在 heard/unheard 各 100 条闭环验证上的 SPL 为 **0.3946/0.3557**，
两组均零错误；对应汇总位于
`result/2026-10-12-sound-adversarial-actor/validation/`。这低于第二版同规模验证的
0.4526/0.4022，因此不将该分支推进到完整测试。100 条结果也不能用来推断它的
完整测试 SPL。声音类别对抗损失在当前配置下没有证明能抵消 heard/unheard 差异。

从同一冻结缓存复现第三版：

```bash
PY=/home/getuanhui/anaconda3/envs/ss/bin/python
LABELS=result/my_foundation_sound_labels
FOUND_RUN=result/my_sound_adversarial_foundation
FOUND=$FOUND_RUN/training/best.pt

$PY -m experiment.avwan.build_sound_labels \
  --cache result/2026-10-11-large-foundation-study/composite_cache \
  --index /data/Getuanhui/avwan_multiple_greedy_every_1000/episodes.jsonl \
  --output "$LABELS"
$PY -m experiment.avwan.train_source_invariant_foundation \
  --output "$FOUND_RUN" --epochs 8 --seed 20261014 \
  --audio-normalization binaural_ratio --augmentation-strength 0.8 \
  --action-consistency-weight 0.1 --representation-consistency-weight 0 \
  --sound-labels "$LABELS" --sound-adversary-weight 0.3 \
  --sound-reversal-scale 0.15
$PY -m experiment.avwan.eval_source_invariance \
  --checkpoint "$FOUND" --output "$FOUND_RUN/source_sensitivity.json"
```

随后用前一节的三条 `train_offline_rl --stage extract` 命令重新提取特征，
将 `FOUND` 与三个缓存目录替换为本次新路径。再用同一节的
`reproduce_heard_unheard train` 命令，以 `--seed 20261011 --epochs 8`
训练 actor，并执行 audit、validate、evaluate。不要沿用第二版的特征缓存：
每份特征必须由本版 foundation 重新提取。

该分支没有完整测试成绩；只有两组各 1000 条测试、零错误、同权重的正式报告
才可判定是否达到 SPL > 0.4。目前已完成完整测试的第二版也没有达到双任务目标。

## 第四版：用真实声音频谱决定可用频带

直接交换双耳共同的频谱缩放会被 `binaural_ratio` 完全消去，对模型输入没有实际
作用，因此没有把这种无效增强当作实验结果。第四版从同一训练批次中选择**不同
声音类别**的样本，计算其双耳和时间平均频谱，按该真实声音较弱的频带对原样本
做双耳同步遮蔽。未遮蔽频带的左右耳比、地图、目标动作和角度标签保持不变。
遮蔽后的双耳比确实与原输入不同，相关单元测试覆盖这一点。损失仍包括两视图
专家动作、角度监督和动作 KL 一致性。该方案试图让动作减少依赖某些训练声音
特有的可用频带；其泛化收益需要重新训练 actor 和闭环评测。

本轮 foundation 选中第 3 epoch，SHA-256 为
`96599e54164c00ff311c0daf42853a47493632c4e4b5e0a8010121883b5b4183`；
8 epoch actor 的 SHA-256 为
`83bc77632f7fe35847cd4b6901202b35cf274b05f9b93331ee3ab0ac77c7680a`。
三组特征缓存的 foundation 哈希一致，actor 审计确认训练与测试 episode 无交集，
测试没有额外重访选择器或专家接管。合成音色扰动动作变化率为 unheard 15.13%、
heard 14.63%，高于第二版约 10%–12%；这只是声音敏感性代理指标。
两组各 100 条闭环验证均零错误：heard SPL **0.4393**、unheard SPL **0.4255**。
相同验证集上第二版分别为 0.4526、0.4022。新版提高了这 100 条 unheard
验证的 SPL，但不能由此推断完整 1000 条测试达标。完整 unheard 1000 条已完成，
SPL **0.3286**、成功率 0.503、零错误，几乎等于第二版完整 unheard 的 0.3277；
100 条验证没有预测到这一结果。heard 完整测试仍在运行，最终同权重报告尚未生成。

```bash
PY=/home/getuanhui/anaconda3/envs/ss/bin/python
$PY -m experiment.avwan.train_source_invariant_foundation \
  --output result/my_real_source_profile_foundation --epochs 8 --seed 20261015 \
  --audio-normalization binaural_ratio \
  --augmentation-type source_profile --source-profile-strength 0.7 \
  --augmentation-strength 0.8 --action-consistency-weight 0.25 \
  --representation-consistency-weight 0.01 \
  --sound-labels result/2026-10-12-foundation-sound-labels
```

随后按第二版流程，用本版 `training/best.pt` **重新提取全部三组特征**，训练
8 epoch actor，执行审计及两组闭环评测。不能复用其他 foundation 的特征缓存。
