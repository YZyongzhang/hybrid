# AV-WaN v2 训练与测试复现

本文给出从 foundation 训练、temporal audio 微调、offline RL 到完整 heard/unheard 测试的命令。所有 Python 调用都直接指向 `.py` 文件。

## 复现目标

`2026-10-17-unheard-generalization-v2` 本身是推理设置实验：它固定 temporal foundation 与 SAC-CQL actor，只把 waypoint 的最大执行间隔从 5 个 primitive steps 改成 3 个。每 3 步重新观测并由策略选择 waypoint。

历史首轮完整测试结果如下；每个 split 均为 1,000 个 episode，0 个评测错误。

| Split | SPL | 成功率 |
|---|---:|---:|
| `test_multiple`（heard） | 0.4727 | 0.613 |
| `test_multiple_unheard`（unheard） | 0.4085 | 0.547 |

第二次完整测试的 SPL 分别为 0.4738 和 0.4078。两轮 unseen SPL 都超过 0.4，但相对 interval 5 的 paired bootstrap 95% 区间跨过 0；不能据此声称统计显著提升。两轮成功率也略有下降。详细数值见 [实验报告](../../result/2026-10-17-unheard-generalization-v2/EXPERIMENT_REPORT.md)。

从头训练会重新生成权重，受软件版本、硬件和非确定性算子影响，不保证逐位得到历史 checkpoint 或相同 SPL。若要复跑历史模型的指标，请使用下文列出的归档 checkpoint 哈希。

这两个固定 test split 已被历史实验多次使用，因此复跑属于历史结果复现，不是新的盲测。

## 运行环境与数据

从仓库根目录开始。当前配置使用 conda 环境 `ss`、CUDA `cuda:0`、Habitat/SoundSpaces、MP3D 场景资源和 AudioNav 数据集：

```bash
conda activate ss
cd /home/getuanhui/project/hybrid_offline_RL
```

训练前还要确保系统 `MemAvailable` 大于 12 GiB。两阶段 foundation 配置将单进程 RSS 上限设为 12,000 MiB，并在 RAM 不足时主动退出；等待其他任务释放内存后再运行，不建议为了绕过保护而降低 `MIN_AVAILABLE_GIB`。

开始前确认下列资产在本机可访问：

- 训练 episode：`/data/Getuanhui/avwan_multiple_greedy_every_1000`
- heard validation episode：`/data/Getuanhui/avwan_multiple_greedy_val_heard`
- STOP-negative 和 recovery 训练 episode：`/data/Getuanhui/avwan_stop_negative_recovery_20261004/policy`、`/data/Getuanhui/avwan_stop_negative_recovery_20261004/recovery`
- temporal foundation 使用的预处理缓存：`result/10-11/total/2026-10-03-unheard-generalization-study/domain_cache`
- offline RL validation demonstrations：`result/10-11/total/2026-10-03-unheard-generalization-study/validation_demonstrations`
- AV-WaN 的 `test_multiple`、`test_multiple_unheard` 数据及对应 MP3D 场景、音频资源

foundation 训练命令会从 raw training episodes 重建缓存。若这些本地资产缺失，需要先按项目数据准备流程恢复；不要用 validation 或 test episode 补齐训练数据。

## 1. 训练前的数据隔离审计

在训练前先冻结主训练集 episode 清单，并核对它与完整 test manifest 的场景和 episode ID 不重叠。随后运行归档数据审计核对 STOP-negative/recovery 训练来源、heard/unheard 声音隔离及已有 validation 清单。审计只读取 test episode ID 和声音类别等元数据，不读取 test 观测、动作、奖励、轨迹或指标。

```bash
python experiment/avwan/snapshot_large_train.py \
  --source /data/Getuanhui/avwan_multiple_greedy_every_1000 \
  --output result/2026-10-17-unheard-generalization-v2-reproduction/train-snapshot
```

该命令会在 `train-snapshot/snapshot.json` 记录 episode 数、来源索引哈希，以及与两个 test split 的 overlap 检查。当前数据源应得到 46,403 episodes、57 个场景、73 个声音类别，`test_scene_overlap` 和 `test_episode_overlap` 都为 0。快照使用指向原始训练文件的符号链接；数据文件应保持不变。

继续检查其余冻结数据来源：

```bash
python experiment/avwan/reproduce_heard_unheard.py audit \
  --output result/2026-10-17-unheard-generalization-v2-reproduction/preflight \
  --name seed_20261017_temporal \
  --checkpoint result/2026-10-17-temporal-audio-foundation/offline/seed_20261017_temporal/training/best_policy.pt \
  --foundation result/2026-10-17-temporal-audio-foundation/training/best.pt
```

确认 `train-snapshot/snapshot.json` 中 `test_scene_overlap`、`test_episode_overlap` 均为 0，并确认新生成的 `preflight/audit.json` 中两个 split 的 `training_episode_overlap` 为 0、`test_multiple_unheard.sounds_seen_in_training` 为 0。归档审计的训练 episode key 数为 46,832。独立 validation 与 test 的划分记录位于 `result/10-11/total/2026-10-03-unheard-generalization-study/split_audit.json`；若来源、manifest 或数据内容与审计不一致，先解决差异，不要开始训练。

若发现 train/validation/test 重叠或违反动作协议，受影响 run 的指标必须标记为 non-compliant，不能用来声称模型提升。

## 2. 训练两阶段 foundation

这个阶段从随机初始化训练 GRL audio stage，再冻结 audio encoder 训练 action stage。训练 episode 来自主训练 snapshot；heard validation 来自独立 validation source；伪 unseen 声音类别只在训练数据内部划分。validation 只用于 checkpoint 选择。输出使用新的 reproduction 目录，不覆盖历史 checkpoint。

必须先完成第 1 步并生成 `result/2026-10-17-unheard-generalization-v2-reproduction/train-snapshot`。如果跳过快照命令，foundation 脚本会报 `Raw episode source directory does not exist` 并在训练开始前退出。

```bash
python run_avwan.py --config experiment/avwan/config/foundation_v2_reproduction.yaml
```

选中的 action foundation 应写入：

```text
result/2026-10-17-unheard-generalization-v2-reproduction/staged-foundation/foundation/action_stage/best.pt
```

如果该输出目录已经有内容，不要直接重跑覆盖；检查上次运行状态，或给新实验配置另一个全新的输出目录。

## 3. 微调 temporal audio foundation

下一步载入刚训练出的 action foundation，使用同一个 `train-snapshot` 中的 train split 做 temporal、圆周方向和空间动作微调；同时使用独立 heard/unheard validation 选择 checkpoint。测试集不参与训练或选择。

```bash
python experiment/avwan/train_temporal_audio_foundation.py \
  --foundation result/2026-10-17-unheard-generalization-v2-reproduction/staged-foundation/foundation/action_stage/best.pt \
  --train-cache result/2026-10-17-unheard-generalization-v2-reproduction/staged-foundation/prepared_data/dataset \
  --unheard-cache result/10-11/total/2026-10-03-unheard-generalization-study/domain_cache \
  --output result/2026-10-17-unheard-generalization-v2-reproduction/temporal-foundation \
  --epochs 5 \
  --batch-size 16 \
  --workers 4 \
  --audio-learning-rate 0.00001 \
  --fusion-learning-rate 0.000005 \
  --weight-decay 0.0005 \
  --seed 20261017 \
  --cpu-threads 4 \
  --device cuda:0
```

按 heard/unheard validation 中较差 split 的 action cross-entropy 选择最优 epoch。checkpoint 路径为 `result/2026-10-17-unheard-generalization-v2-reproduction/temporal-foundation/training/best.pt`。

## 4. 提取离线特征并训练 SAC-CQL actor

配置 [offline_rl_temporal_v2_reproduction.yaml](config/offline_rl_temporal_v2_reproduction.yaml) 固定训练数据来源、验证数据来源、随机种子和超参数；`PARENT: null` 表示 actor/critic 从随机初始化开始。`FRESH_FEATURES` 会在新的 reproduction 输出目录下重新提取特征。

```bash
python experiment/avwan/train_offline_rl_reproduction.py \
  --config experiment/avwan/config/offline_rl_temporal_v2_reproduction.yaml \
  --stage all \
  --fresh-features
```

配置为 20 epochs、每个 epoch 500 updates、batch size 32、pool sampling `[16, 8, 8]`、gamma 0.01、BC weight 5、CQL weight 1、learning rate 0.0001、foundation prior 开启。脚本依次提取 train/additional/validation 特征、训练 actor、写入数据审计并运行 100 集 validation。actor checkpoint 按独立 validation demonstrations 上的 `balanced STOP NLL + 0.1 × action NLL` 选择，策略 checkpoint 为：

```text
result/2026-10-17-unheard-generalization-v2-reproduction/offline/seed_20261017_temporal/training/best_policy.pt
```

训练完成后、full test 前检查 `result/2026-10-17-unheard-generalization-v2-reproduction/offline/audit.json`。它必须报告零 training/test episode overlap，并记录所用 foundation 和源 manifest。测试 wrapper 会核对 checkpoint/foundation hash、测试集覆盖数和设置。

## 5. 只用 validation 选择 waypoint 间隔

在完整测试前，用两个 validation split 各 100 个 episode 对比间隔 5 和 3。不同间隔使用不同输出目录，避免续跑时复用另一设置的结果。下面四条命令都使用同一个冻结 actor/foundation；每条仅运行 validation：

```bash
python experiment/avwan/reproduce_heard_unheard.py validate \
  --output result/2026-10-17-unheard-generalization-v2-reproduction/validation-i3 \
  --name seed_20261017_temporal \
  --checkpoint result/2026-10-17-unheard-generalization-v2-reproduction/offline/seed_20261017_temporal/training/best_policy.pt \
  --foundation result/2026-10-17-unheard-generalization-v2-reproduction/temporal-foundation/training/best.pt \
  --evaluation-config experiment/avwan/config/audiogoal.yaml \
  --validation-split val_multiple \
  --validation-episodes 100 \
  --prediction-interval 3 \
  --eval-seed 20260930 \
  --device cuda:0 \
  --gpu-memory-fraction 0.12 \
  --max-rss-mb 7000

python experiment/avwan/reproduce_heard_unheard.py validate \
  --output result/2026-10-17-unheard-generalization-v2-reproduction/validation-i3 \
  --name seed_20261017_temporal \
  --checkpoint result/2026-10-17-unheard-generalization-v2-reproduction/offline/seed_20261017_temporal/training/best_policy.pt \
  --foundation result/2026-10-17-unheard-generalization-v2-reproduction/temporal-foundation/training/best.pt \
  --evaluation-config experiment/avwan/config/audiogoal.yaml \
  --validation-split val_multiple_unheard \
  --validation-episodes 100 \
  --prediction-interval 3 \
  --eval-seed 20260930 \
  --device cuda:0 \
  --gpu-memory-fraction 0.12 \
  --max-rss-mb 7000

python experiment/avwan/reproduce_heard_unheard.py validate \
  --output result/2026-10-17-unheard-generalization-v2-reproduction/validation-i5 \
  --name seed_20261017_temporal \
  --checkpoint result/2026-10-17-unheard-generalization-v2-reproduction/offline/seed_20261017_temporal/training/best_policy.pt \
  --foundation result/2026-10-17-unheard-generalization-v2-reproduction/temporal-foundation/training/best.pt \
  --evaluation-config experiment/avwan/config/audiogoal.yaml \
  --validation-split val_multiple \
  --validation-episodes 100 \
  --prediction-interval 5 \
  --eval-seed 20260930 \
  --device cuda:0 \
  --gpu-memory-fraction 0.12 \
  --max-rss-mb 7000

python experiment/avwan/reproduce_heard_unheard.py validate \
  --output result/2026-10-17-unheard-generalization-v2-reproduction/validation-i5 \
  --name seed_20261017_temporal \
  --checkpoint result/2026-10-17-unheard-generalization-v2-reproduction/offline/seed_20261017_temporal/training/best_policy.pt \
  --foundation result/2026-10-17-unheard-generalization-v2-reproduction/temporal-foundation/training/best.pt \
  --evaluation-config experiment/avwan/config/audiogoal.yaml \
  --validation-split val_multiple_unheard \
  --validation-episodes 100 \
  --prediction-interval 5 \
  --eval-seed 20260930 \
  --device cuda:0 \
  --gpu-memory-fraction 0.12 \
  --max-rss-mb 7000
```

历史 validation screen 的 heard SPL 为 interval 5 的 0.4407、interval 3 的 0.4178；unheard SPL 分别为 0.3919 和 0.4089。按“两组 validation SPL 均须超过 0.4”的目标选 interval 3。paired bootstrap 区间跨 0，因此这是按既定验证目标作的设置选择，不是显著性结论。记录两组 validation episode 数、选择哈希和错误数；有错误时先处理环境问题，不要用 test 调参。

## 6. 冻结设置后跑完整测试

确认 policy、foundation、`prediction_interval=3`、seed、配置和数据审计均已固定后，运行两个完整 test split。评估使用策略原始 masked argmax 输出；`revisit_penalty=0`，不重置 recurrent memory，不使用 expert action，也不加碰撞恢复、深度启发式方向选择或 STOP 阈值。共同 waypoint planner 只执行所选动作，设置对两个 split 和所有模型一致。

```bash
python experiment/avwan/reproduce_heard_unheard.py evaluate \
  --output result/2026-10-17-unheard-generalization-v2-reproduction/test-i3 \
  --name seed_20261017_temporal \
  --checkpoint result/2026-10-17-unheard-generalization-v2-reproduction/offline/seed_20261017_temporal/training/best_policy.pt \
  --foundation result/2026-10-17-unheard-generalization-v2-reproduction/temporal-foundation/training/best.pt \
  --evaluation-config experiment/avwan/config/audiogoal.yaml \
  --splits test_multiple test_multiple_unheard \
  --prediction-interval 3 \
  --eval-seed 20260930 \
  --device cuda:0 \
  --shards 4 \
  --episodes-per-process 20 \
  --max-rss-mb 7000 \
  --gpu-memory-fraction 0.12
```

```bash
python experiment/avwan/reproduce_heard_unheard.py evaluate \
  --output result/2026-10-17-unheard-generalization-v2-reproduction/test-i4 \
  --name seed_20261017_temporal \
  --checkpoint result/2026-10-17-temporal-audio-foundation/offline/seed_20261017_temporal/training/best_policy.pt \
  --foundation result/2026-10-17-temporal-audio-foundation/training/best.pt \
  --evaluation-config experiment/avwan/config/audiogoal.yaml \
  --splits test_multiple test_multiple_unheard \
  --prediction-interval 3 \
  --eval-seed 20260930 \
  --device cuda:0 \
  --shards 4 \
  --episodes-per-process 20 \
  --max-rss-mb 7000 \
  --gpu-memory-fraction 0.12
```

评测 wrapper 会续跑未完成 shard，最后核对每个 split 恰好包含 1,000 个唯一 episode，checkpoint/foundation hash 与设置相符且 errors 为 0。每个 shard 会保存 `run.json`、`selection.json`、`episodes.jsonl`、`summary.json` 和 `environment.yaml`。结果写到：

```text
result/2026-10-17-unheard-generalization-v2-reproduction/test-i3/full_test/test_multiple.json
result/2026-10-17-unheard-generalization-v2-reproduction/test-i3/full_test/test_multiple_unheard.json
result/2026-10-17-unheard-generalization-v2-reproduction/test-i3/full_test/report.json
```

報告至少保留每個 split 的 episode 數、SPL、成功率、errors、checkpoint 與 foundation SHA-256、prediction interval 和 policy 標識。不要根據 test 結果改 checkpoint、間隔或動作處理後再報同一輪結果；任何新候選都應回到 train/validation 階段重新選定，再做一次完整 test，並標明固定 test 集已被歷史實驗使用。

## 僅重跑歷史 checkpoint 的測試

如果目標是確認歸檔模型的歷史指標，而不是重新訓練，可略過第 2–5 步，直接使用保存的 temporal checkpoint。歷史模型 SHA-256 為：

- actor：`d4b038bbebff2d8d9683e93173426a71d0d72d752a2549ac121ce66d673807dc`
- foundation：`5d8668da3581a330163702066f9871e957e5ab6b2af21f4f69af6a9e18cc32c8`

将上一节完整测试命令中的 `--output` 改为一个新的输出目录，并将 `--checkpoint` 与 `--foundation` 改为：

```text
result/2026-10-17-temporal-audio-foundation/offline/seed_20261017_temporal/training/best_policy.pt
result/2026-10-17-temporal-audio-foundation/training/best.pt
```

运行配置与归档 hash 都匹配时，首轮结果应接近 Heard SPL 0.4727、Unheard SPL 0.4085；Habitat/SoundSpaces 或硬件差异可能带来少量变化。

## 相关记录

- [v2 实验报告](../../result/2026-10-17-unheard-generalization-v2/EXPERIMENT_REPORT.md)
- [interval validation 配对汇总](../../result/2026-10-17-unheard-generalization-v2/interval_validation/paired_summary.json)
- [首轮完整测试汇总](../../result/2026-10-17-unheard-generalization-v2/full_test/report.json)
- [interval 5/3 逐 episode 对比](../../result/2026-10-17-unheard-generalization-v2/paired_full_test_comparison.json)
- [temporal foundation 与 offline RL 报告](../../result/2026-10-17-temporal-audio-foundation/EXPERIMENT_REPORT.md)
