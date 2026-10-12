# AVWAN offline RL：heard / unheard 完整测试复现

## 目标和判定

同一组 `foundation checkpoint + SAC-CQL actor checkpoint` 分别在
`test_multiple`（heard）和 `test_multiple_unheard`（unheard）的全部 1000 条
episode 上取得平均 SPL **严格大于 0.4**。评测脚本检查每组恰好 1000 个唯一
episode、没有运行错误、checkpoint 哈希一致，才写 `full_test/report.json`。
此前各 100 条的结果只是探索性结果，不能代替完整测试。

## 2026-10-11 的固定候选

本轮按两组独立、各 100 条验证 SPL 的较低值选中同一套权重：
foundation `result/2026-10-11-large-foundation-study/ratio_big_8ep/training/best.pt`
（SHA-256 `560051e798777238c172e53a0e1c85fd4ada68b9121f6c532a202aad06eecf4b`），
actor `result/2026-10-11-refined-actor/seed_20261011_refined/training/latest.pt`
（SHA-256 `f88ac345d94cb923834eec597219cb6c8189e4c603949d88c04544173875d1c4`）。
验证结果是 heard 0.4295216332、unheard 0.4032419297；它们不是完整测试成绩。
训练流程见下文“固定持续收集数据”和“微调 foundation”两节，固定快照是
`result/2026-10-11-large-train-snapshot`，actor 用 `--sampling 16 8 8`、
`--steps-per-epoch 500 --epochs 20 --seed 20261011`。
`result/2026-10-11-refined-actor/validation_selection.json` 记录选择；
`result/2026-10-11-selection-extension-03.json` 记录该分支是在看到旧模型部分测试
结果后提出，属于探索性研究。

旧 foundation 大数据候选的完整测试已完成，每组恰好 1000 条、零错误：
heard SPL 0.3294740689、unheard SPL 0.3022979562，**未达标**。
审计和结果在 `result/2026-10-11-large-heavy-recovery/audit.json`、
`result/2026-10-11-large-heavy-recovery/full_test/report.json`。

上述 20 epoch 新 foundation 候选的完整测试也已完成，每组 1000 条、零错误：
heard SPL 0.4303404737、unheard SPL 0.3372000389，**未达双任务目标**。
报告在 `result/2026-10-11-refined-actor/full_test/report.json`。后续 8 epoch
actor 的双验证分别为 heard 0.3990725440、unheard 0.4445870096；它未通过原定
“两组较低验证 SPL 超过 0.4032419297”的规则。由于差距只有约 0.0042，
另行作为探索性候选测试，详见 `result/2026-10-11-selection-extension-06.json`。
不能把它写成预先选中的盲测模型。

8 epoch 探索性候选也已完成两组各 1000 条、零错误测试：heard SPL
**0.4096224103**、unheard SPL **0.3400147024**，仍**未达双任务目标**。
报告在 `result/2026-10-11-refined-actor-8ep/full_test/report.json`，actor
SHA-256 为 `24a030c1bac8ee6db80a441fded55cb4be83331da4a6800f58eedf405be3255a`。
它是目前已完成全量测试的模型中两组 SPL 较低值最高的一组。4 epoch 候选
验证 heard 0.4114052324、unheard 0.4256137883，但 unheard 固定分片前
103 条测试仅有 SPL 0.2557387469；这条探索分支已停止，**没有完整测试成绩**，
见 `result/2026-10-11-refined-actor-4ep/full_test/stopped_unselected.json`。

从已提取好的三组新 foundation 特征缓存重训目前最好全量候选的 actor：

```bash
cd /home/getuanhui/project/hybrid_offline_RL
PY=/home/getuanhui/anaconda3/envs/ss/bin/python
FOUND=result/2026-10-11-large-foundation-study/ratio_big_8ep/training/best.pt
OUT=result/my_refined_8epoch_reproduction
$PY -m experiment.avwan.reproduce_heard_unheard train \
  --output "$OUT" --name seed_20261011_refined_8ep --seed 20261011 \
  --foundation "$FOUND" \
  --base-cache result/2026-10-11-refined-base-features \
  --additional-cache result/2026-10-11-refined-correction-features \
  --validation-cache result/2026-10-11-refined-validation-features \
  --steps-per-epoch 500 --epochs 8 --sampling 16 8 8
$PY -m experiment.avwan.reproduce_heard_unheard audit \
  --output "$OUT" --name seed_20261011_refined_8ep --foundation "$FOUND"
$PY -m experiment.avwan.reproduce_heard_unheard validate \
  --output "$OUT" --name seed_20261011_refined_8ep --foundation "$FOUND" \
  --validation-split val_multiple_unheard --validation-episodes 100
$PY -m experiment.avwan.reproduce_heard_unheard validate \
  --output "$OUT" --name seed_20261011_refined_8ep --foundation "$FOUND" \
  --validation-split val_multiple --validation-episodes 100
```

随后在下方完整测试命令中把 `OUT` 和 `name` 换为自己的目录与
`seed_20261011_refined_8ep`。如要从原始轨迹开始，先按下文两个小节生成固定
快照、foundation 与三组特征缓存。没有任何一条已核实的 pipeline 可以宣称
在这两个完整测试集上都超过 0.4。

从同一套权重跑两组完整测试的命令如下。`--shards 2` 把每组固定分成两个
500 条分片；在内存允许的终端分别运行四种 `split/shard` 组合，最后的汇总命令
检查每组 1000 条唯一 episode、零错误和权重哈希。中断后重跑同一命令会续跑。

```bash
cd /home/getuanhui/project/hybrid_offline_RL
PY=/home/getuanhui/anaconda3/envs/ss/bin/python
OUT=result/2026-10-11-refined-actor
FOUND=result/2026-10-11-large-foundation-study/ratio_big_8ep/training/best.pt
for split in test_multiple test_multiple_unheard; do
  for shard in 0 1; do
    $PY -m experiment.avwan.reproduce_heard_unheard evaluate \
      --output "$OUT" --name seed_20261011_refined --foundation "$FOUND" \
      --splits "$split" --shards 2 --shard-index "$shard" \
      --episodes-per-process 20 --max-rss-mb 7000 &
  done
done
wait
$PY -m experiment.avwan.reproduce_heard_unheard evaluate \
  --output "$OUT" --name seed_20261011_refined --foundation "$FOUND" \
  --splits test_multiple test_multiple_unheard --shards 2
```

## 固定数据与环境

在 `/home/getuanhui/project/hybrid_offline_RL` 运行，使用
`/home/getuanhui/anaconda3/envs/ss/bin/python`。实验只使用 AVWAN；训练轨迹
来自 `/data/Getuanhui/avwan_multiple_greedy_train` 和
`/data/Getuanhui/avwan_stop_negative_recovery_20261004`。
冻结特征缓存是 `result/2026-10-03-unheard-generalization-study/ratio_memory_features`
及 `/data/Getuanhui/avwan_stop_negative_recovery_20261004/frozen_ratio_features`。
缓存身份绑定 foundation 权重 SHA-256，训练脚本会校验。已有 foundation 和
parent actor 分别是 `result/2026-10-03-unheard-generalization-study/best_foundation.pt`
与 `result/2026-10-03-unheard-generalization-study/exp013-ratio-sac-g0p01-prior-bc5/training/best_policy.pt`。
这条流程从已训练好的视觉/听觉 foundation 和 SAC parent 继续训练 **新的一版
offline RL actor**；如果需要重训 foundation/parent，先按照
`result/2026-10-03-unheard-generalization-study/PROTOCOL.md` 的原始实验生成这两份权重。

基础旧数据分支有 5650 个原始 train episode。两组测试各 1000 条；train/test 的
episode key 无重叠。unheard 的 18 类声音与训练声音无交集。审计命令会再次检查。
基础旧数据分支不直接读取持续增长的目录；使用大数据时须先按下一节冻结快照。

## 可选：固定持续收集数据后训练

采集器还在写入 `/data/getuanhui/avwan_multiple_greedy_every_1000`。先固定已保存
episode 的文件清单，再提取特征；不要直接对仍在增长的源目录提取。快照只创建
符号链接，原始 `.pkl.gz` 文件必须保留。程序会检查 train/test 场景及 episode
无交集，并在特征提取结束前后检查文件集合与大小、修改时间未变。

```bash
cd /home/getuanhui/project/hybrid_offline_RL
PY=/home/getuanhui/anaconda3/envs/ss/bin/python
SNAP=result/my_avwan_large_snapshot
CACHE=result/my_avwan_large_features
OUT=result/my_avwan_large_run

$PY -m experiment.avwan.snapshot_large_train --output "$SNAP"
$PY -m experiment.avwan.train_offline_rl --stage extract \
  --source "$SNAP" \
  --checkpoint result/2026-10-03-unheard-generalization-study/best_foundation.pt \
  --cache "$CACHE" --device cuda:0
$PY -m experiment.avwan.reproduce_heard_unheard train \
  --output "$OUT" --name seed_20261011_large --seed 20261011 \
  --base-cache "$CACHE" --steps-per-epoch 500 --epochs 20 \
  --sampling 24 4 4
$PY -m experiment.avwan.reproduce_heard_unheard audit \
  --output "$OUT" --name seed_20261011_large
$PY -m experiment.avwan.reproduce_heard_unheard validate \
  --output "$OUT" --name seed_20261011_large --validation-episodes 100
$PY -m experiment.avwan.reproduce_heard_unheard validate \
  --output "$OUT" --name seed_20261011_large \
  --validation-split val_multiple --validation-episodes 100
```

这条分支从同一 parent actor 初始化，使用大数据专家轨迹、原失败 STOP 和专家恢复
轨迹每批 24:4:4；固定最终 epoch 20。它是单独候选，必须先在独立验证集决定是否
采用。多个候选按两组验证 SPL 中较低者选最高，平局按 unheard 验证 SPL。再对
heard/unheard 两个完整测试集评测。`--checkpoint` 可用于在同一套
完整评测命令中指定该分支的 `training/latest.pt`。

## 可选：使用同一大快照微调 foundation

如果现有 foundation 的闭环验证仍不足，可以在固定快照上微调双耳比值模型。
`train_large_foundation` 只用大快照训练，复用现有独立 unheard/ heard 专家轨迹
做验证。选中的 foundation 权重确定后，**训练、恢复和验证三组冻结特征都必须
用同一份新权重重新提取**，不能混用旧缓存。

```bash
cd /home/getuanhui/project/hybrid_offline_RL
PY=/home/getuanhui/anaconda3/envs/ss/bin/python
SNAP=result/my_avwan_large_snapshot
FOUND_CACHE="$PWD/result/my_large_foundation_cache"
FOUND_OUT=result/my_large_foundation_run
$PY -m experiment.avwan.train --stage prepare \
  --source "$PWD/$SNAP" --cache "$FOUND_CACHE"
$PY -m experiment.avwan.train_large_foundation \
  --train-cache "$FOUND_CACHE" --output "$FOUND_OUT" \
  --epochs 8 --learning-rate 0.00005
FOUND_BEST="$FOUND_OUT/ratio_big_8ep/training/best.pt"

$PY -m experiment.avwan.train_offline_rl --stage extract \
  --source "$SNAP" --checkpoint "$FOUND_BEST" \
  --cache result/my_refined_base_features --device cuda:0
$PY -m experiment.avwan.train_offline_rl --stage extract \
  --source /data/Getuanhui/avwan_stop_negative_recovery_20261004/feature_source \
  --checkpoint "$FOUND_BEST" \
  --cache result/my_refined_correction_features --device cuda:0
$PY -m experiment.avwan.train_offline_rl --stage extract \
  --source result/2026-10-03-unheard-generalization-study/validation_demonstrations \
  --checkpoint "$FOUND_BEST" \
  --cache result/my_refined_validation_features --device cuda:0

$PY -m experiment.avwan.reproduce_heard_unheard train \
  --output result/my_refined_actor --name seed_20261011 --seed 20261011 \
  --foundation "$FOUND_BEST" --base-cache result/my_refined_base_features \
  --additional-cache result/my_refined_correction_features \
  --validation-cache result/my_refined_validation_features \
  --steps-per-epoch 500 --epochs 20 --sampling 16 8 8
$PY -m experiment.avwan.reproduce_heard_unheard audit \
  --output result/my_refined_actor --name seed_20261011 \
  --foundation "$FOUND_BEST"
```

其后用相同 `--foundation "$FOUND_BEST"` 在 heard 与 unheard 的独立验证集
选模，最后才进行完整测试。`train_large_foundation.py` 对现有 ratio foundation
微调音频和动作头，冻结视觉/地图主干；这一步不是从随机初始化重训整个系统。

## 从现有数据训练一版新的 actor

```bash
cd /home/getuanhui/project/hybrid_offline_RL
PY=/home/getuanhui/anaconda3/envs/ss/bin/python
OUT=result/my_avwan_heard_unheard

$PY -m experiment.avwan.reproduce_heard_unheard train \
  --output "$OUT" --name seed_20261010 --seed 20261010
$PY -m experiment.avwan.reproduce_heard_unheard audit \
  --output "$OUT" --name seed_20261010
$PY -m experiment.avwan.reproduce_heard_unheard validate \
  --output "$OUT" --name seed_20261010 \
  --validation-split val_multiple --validation-episodes 100
$PY -m experiment.avwan.reproduce_heard_unheard validate \
  --output "$OUT" --name seed_20261010 \
  --validation-split val_multiple_unheard --validation-episodes 100
```

策略从 parent 权重初始化，40 epoch、每 epoch 186 批。每批从原始轨迹、
真实失败 STOP 轨迹和专家实际执行的恢复轨迹按 16:8:8 采样。Critic 使用
真实执行动作和相应转移；BC 仅使用允许的真实动作。训练时对失败 STOP 的
奖励减 1，错误 STOP 不参加 BC。参数为 gamma=0.01、BC 权重 5、CQL 权重 1、
learning rate=1e-4、冻结 foundation、保留 recurrent actor。评估使用固定
**第 40 epoch `training/latest.pt`**。验证集的 STOP 代理损失曾偏向 epoch 0，
因此不把 `best_policy.pt` 当作这轮训练的产物。

同名 run 已完成时训练命令会跳过；换 `--name` 与 `--seed` 可训练新 run。
如果特征缓存缺失，可先按 `experiment/avwan/train_stop_rl.py --prepare-features`
重建新增轨迹的冻结特征；原始 `ratio_memory_features` 需按前述 Oct 3 实验生成。

## 完整闭环评测

下面两个命令可在两个终端同时运行。它们各自顺序评测一个 split，分批释放
SoundSpaces 仿真进程内存。中断后原命令重跑会续跑；每条结果保存在
`full_test/<split>/shard_0/episodes.jsonl`，进度在相邻的 `summary.json`。

```bash
$PY -m experiment.avwan.reproduce_heard_unheard evaluate \
  --output "$OUT" --name seed_20261010 --splits test_multiple \
  --episodes-per-process 20
$PY -m experiment.avwan.reproduce_heard_unheard evaluate \
  --output "$OUT" --name seed_20261010 --splits test_multiple_unheard \
  --episodes-per-process 20
```

最后任意一条命令完成时，如果另一组也已完成，会写
`full_test/report.json`，其中 `passed` 只在两组 SPL 都大于 0.4 时为 true。
两组都完成后也可显式运行
`$PY -m experiment.avwan.reproduce_heard_unheard report --output "$OUT" --name seed_20261010`
重建汇总。
两条命令同一时间使用同一个 checkpoint；勿让两进程同时评测同一 split。
默认 seed=20260930、200 primitive steps、prediction interval=5。

内存允许时，可在四个终端分别运行下面的命令，利用固定的两个分片并行；四条
命令必须使用同一个 `OUT`、`name`、checkpoint 和 `--shards 2`。各分片完成后，
最后一条不带 `--shard-index` 的命令核对并汇总两组各 1000 条结果。

```bash
for split in test_multiple test_multiple_unheard; do
  for shard in 0 1; do
    $PY -m experiment.avwan.reproduce_heard_unheard evaluate \
      --output "$OUT" --name seed_20261010 \
      --splits "$split" --shards 2 --shard-index "$shard" \
      --episodes-per-process 20 --max-rss-mb 7000 &
  done
done
wait
$PY -m experiment.avwan.reproduce_heard_unheard evaluate \
  --output "$OUT" --name seed_20261010 --shards 2 \
  --splits test_multiple test_multiple_unheard
```

## 无高层 rule-based 提升的边界

`evaluate.py` 默认 `revisit_penalty=0`，未设置 `--reset-memory`，也不传
`--selection-source` 或 `--max-episodes`。`offline/policy.py` 从冻结观测特征
得到 actor logits，经合法 waypoint mask 后取 argmax；每个 episode 重置 LSTM。
评测循环不调用 `get_expert_map_action()`。数据采集中的专家行为只作为训练标签。

AVWAN 本身的 mapper、合法动作 mask 和局部 waypoint planner 是既有的低层
执行部件；planner 在无路径时还有随机回退。结论只能表述为
“没有专家接管或额外高层 rule-based 动作重排”，
不能表述为“完全没有规则代码”。尤其不能把加入 `--revisit-penalty` 的分数
算作纯 offline RL actor 的提升。

SoundSpaces 当前实时声学渲染使用 `temporalCoherence=True`。既往相同 seed、不同
进程访问顺序的复测出现过初始声谱图差异，见
`result/2026-10-03-unheard-generalization-study/REPRODUCIBILITY.md`。因此一次完整
测试只证明该次执行的成绩；若接近阈值，应在保持相同推理配置下重复整套测试，
并报告重复运行的波动。
