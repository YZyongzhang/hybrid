# 2026-10-12 Staged GRL Offline RL 复现指南

本文从原始 episode 重新整理训练数据、重新训练 foundation 与 SAC-CQL，并评估 `2026-10-12-staged-grl-offline-rl` 实验。流程不会复用之前生成的观测张量、声音标签或 foundation 特征。数据收集、场景资产和声音资产属于前置条件；本流程不重新采集轨迹。

原实验记录位于：

- 协议与结果：`result/2026-10-12-staged-grl-offline-rl/EXPERIMENT.md`
- 完整配置：`result/2026-10-12-staged-grl-offline-rl/experiment_config.json` 和 `offline_rl.yaml`
- 最终评估：`result/2026-10-12-staged-grl-offline-rl/full_test/report.json`

先激活已安装项目依赖和 Habitat / SoundSpaces 的 Python 环境，再从项目根目录运行以下命令。新结果写入 `result/2026-10-13-staged-grl-reproduction/`，不覆盖归档实验。

## 1. 准备原始数据

确认以下原始 episode 文件和仿真资源可访问：

| 用途 | 来源 |
|---|---|
| 73 类 foundation 训练 episode 的固定成员快照 | `result/10-11/total/2026-10-11-large-train-snapshot/episodes/`；快照链接到 `/data/Getuanhui/avwan_multiple_greedy_every_1000` 中的原始 episode |
| Heard 验证 episode | `/data/Getuanhui/avwan_multiple_greedy_val_heard/episodes/` |
| 离线 RL 独立验证演示 | `result/10-11/total/2026-10-03-unheard-generalization-study/validation_demonstrations/episodes/` |
| 离线 RL 基础轨迹 | `/data/Getuanhui/avwan_multiple_greedy_every_1000/episodes/` |
| 失败 STOP 与恢复轨迹 | `/data/Getuanhui/avwan_stop_negative_recovery_20261004/policy/episodes/` 和 `/data/Getuanhui/avwan_stop_negative_recovery_20261004/recovery/episodes/` |
| STOP / 专家动作标注 | `/data/Getuanhui/avwan_stop_negative_recovery_20261004/training_annotations.json`；若配置启用专家动作 BC，还需同目录的 `expert_bc_annotations.json` |
| 仿真评估 | Habitat / SoundSpaces MP3D 场景、RIR、声音资产，以及 `test_multiple`、`test_multiple_unheard` 数据集 |

foundation 命令每次都会从训练集和 heard 验证集重新整理观测，并从训练 episode 元数据读取声音类别。离线 RL 命令会从轨迹、动作标注和本轮 foundation 权重重新计算训练特征，也会使用表中的独立验证演示。原始轨迹及其固定成员快照必须保留；派生数据会写入本次新结果目录。

若数据存放位置不同，可用 `--train-source` 和 `--heard-validation-source` 指定 foundation 的 episode 根目录。离线数据源位置在新建的 `offline_rl.yaml` 中配置。

## 2. 生成 pseudo-heard / pseudo-unheard 划分并训练 foundation

类别划分取自 73 个基础训练集类别。seed `20261012` 确定性打乱类别后，将前 36 类标为 pseudo-unheard，其余 37 类标为 pseudo-heard。真实 unheard 数据不参与类别划分、梯度训练或 foundation 检查点选择。

Foundation 网络从随机初始化开始。音频归一化为 `binaural_ratio`，地图归一化为 `relative`，并启用 spatial-detail 分支。数据源、训练轮数和设备等参数已写入 `experiment/avwan/config/foundation_two_stage.yaml`，直接运行：

```bash
python run_avwan.py --config experiment/avwan/config/foundation_two_stage.yaml
```

第一阶段用方向分类交叉熵和 GRL 域分类损失训练 5 轮，按 heard 验证集角度损失选择检查点。第二阶段加载该阶段最佳权重，冻结完整音频编码器，再训练动作头 5 轮，按 heard 验证集动作损失选择最终 foundation。真实 unheard 泛化留到离线 RL 验证和第 4 步完整测试中评估。

检查以下结果：

- `pseudo_unheard_split.json`：seed、36/37 类清单和类别计数。
- `foundation/audio_stage/best.pt`：音频阶段最佳权重。
- `foundation/action_stage/best.pt`：给离线 RL 使用的 foundation 权重。
- `foundation/action_stage/frozen_audio_audit.json`：`"passed": true`。
- `status.json`：`"status": "foundation_trained"`。

## 3. 用 run_avwan.py 训练离线 SAC-CQL 并审计

第 2 步完成后，foundation 权重位于 `result/2026-10-13-staged-grl-reproduction/foundation/action_stage/best.pt`。离线 RL 的数据源、输出目录和超参数已写入 `experiment/avwan/config/offline_rl_reproduction.yaml`。配置中的 `PARENT` 为 `null`，因此 actor、双 critic 和目标 critic 都从随机权重初始化；actor 的冻结 prior 使用第 2 步训练出的 foundation 动作头。

直接从项目根目录启动完整流程：

```bash
python run_avwan.py --config experiment/avwan/config/offline_rl_reproduction.yaml
```

配置的 `STAGE: all` 会重新提取基础轨迹、失败 STOP / 恢复轨迹和独立验证数据的特征，然后训练、审计数据并运行独立导航验证。`FRESH_FEATURES: true` 会清理并重建本轮 `RUN.OUTPUT` 下的派生特征目录。训练设置为 20 epochs × 500 updates、batch size 32、轨迹采样 `[16, 8, 8]`（基础 / 失败 STOP / 恢复）、`gamma=0.01`、`BC_WEIGHT=5`、`CQL_WEIGHT=1`、`FOUNDATION_PRIOR=true`。检查点按独立验证上的 `balanced STOP NLL + 0.1 × action NLL` 选择。

若训练中断，使用同一命令会重新生成特征，并从训练 `latest.pt` 恢复。派生特征路径必须位于该配置的 `RUN.OUTPUT` 目录内。

运行完成后检查：

- `result/2026-10-13-staged-grl-reproduction/offline/seed_20261012_grl_staged_foundation/training/completed.json`
- 同一 `training/` 目录下的 `best.json` 和 `best_policy.pt`
- `result/2026-10-13-staged-grl-reproduction/offline/audit.json`：训练 episode 与两个测试集无 episode 重叠；unheard 测试中的声音类别没有出现在训练集中。

## 4. 对两个完整测试集做闭环评估

用最佳 offline policy 和对应 foundation 各评估完整 1,000 集。每个仿真进程最多新跑 20 集；`--resume` 允许进程退出后从已保存的 episode 清单继续。

```bash
python -m experiment.avwan.reproduce_heard_unheard evaluate \
  --output result/2026-10-13-staged-grl-reproduction \
  --name seed_20261012_grl_staged_foundation \
  --checkpoint result/2026-10-13-staged-grl-reproduction/offline/seed_20261012_grl_staged_foundation/training/best_policy.pt \
  --foundation result/2026-10-13-staged-grl-reproduction/foundation/action_stage/best.pt \
  --device cuda:0 \
  --splits test_multiple test_multiple_unheard \
  --shards 1 \
  --episodes-per-process 20 \
  --eval-seed 20260930 \
  --prediction-interval 5 \
  --max-rss-mb 8000
```

结果写入 `result/2026-10-13-staged-grl-reproduction/full_test/report.json`。完整复现要求两个 split 都有 1,000 集且 rollout errors 为 0。对比基线时使用相同的测试 episode 清单；同一机器、同一数据集和 seed `20260930` 应生成相同的 selection 清单。

## 5. 归档运行结果

下面是 2026-10-12 归档运行的历史结果。该次运行加载了预训练 foundation 检查点和离线 RL `PARENT`；不能作为本指南随机初始化流程的预期值：

| 阶段 / split | 原实验记录 |
|---|---|
| pseudo 类别划分 | 36 pseudo-unheard / 37 pseudo-heard，seed `20261012` |
| 音频阶段最佳轮次 | epoch 3，worst-domain angle loss `1.42735` |
| 动作阶段最佳轮次 | epoch 3，worst-domain action loss `2.25893` |
| 离线 RL 最佳轮次 | epoch 13，validation selection score `0.23950` |
| `test_multiple` | 1,000 集，SPL `0.3953`，成功率 `0.596` |
| `test_multiple_unheard` | 1,000 集，SPL `0.3545`，成功率 `0.538` |

该实验设定的 SPL 目标为 0.4，归档报告 `passed: false`。本指南会产生新的权重和指标，写入 `result/2026-10-13-staged-grl-reproduction/`。原始训练协议、审计和 episode 结果见 `result/2026-10-12-staged-grl-offline-rl/`。其中 `EXPERIMENT.md` 的基线 unheard 数值是当时的部分结果；完整基线结果见 `result/2026-10-12-current-foundation-offline-rl/full_test/report.json`。
