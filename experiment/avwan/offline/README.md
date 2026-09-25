# Frozen foundation → LSTM SAC-CQL

在仓库根目录运行（不需要启动 Habitat 仿真器）：

```bash
/home/getuanhui/anaconda3/envs/ss/bin/python experiment/avwan/train_offline_rl.py
```

默认配置为 `experiment/avwan/config/offline_rl.yaml`，依次执行全量特征提取和 30 epoch 离线训练。YAML 相对路径以 `experiment/avwan` 为根；命令行路径以当前目录为根。

## 分阶段与恢复

```bash
# 冻结 foundation，提取全部训练及独立验证 episode
python experiment/avwan/train_offline_rl.py --stage extract

# 只读取已完成的特征缓存训练，不再读取原始观测或运行 foundation
python experiment/avwan/train_offline_rl.py --stage train

# epochs 是目标总轮数；恢复模型、目标网络、全部优化器及随机状态
python experiment/avwan/train_offline_rl.py --stage train --epochs 60 \
  --resume experiment/avwan/runs/telephone_epoch010_offline_lstm_saccql_v1/latest.pt
```

可以通过 `--checkpoint`、`--source`、`--val-source`、`--cache`、`--output`、`--device`、`--batch-size`、`--extract-batch-size` 覆盖设置。`--max-episodes N` 仅用于小规模诊断，每个 split 最多 N 个 episode，必须指定不同缓存目录。已有训练目录须显式恢复，防止覆盖结果。

提取中断后重新运行相同命令即可续跑。缓存身份绑定 checkpoint SHA-256、源数据快照和预处理版本；改变权重或数据后需要新缓存目录。训练阶段以缓存内的 checkpoint 身份为准，不要求原始 checkpoint 仍在线。

## 特征格式

默认权重：`runs/telephone_train_val_20260925/epoch_010.pt`。支持纯 `state_dict` 或带 `model` 的 foundation 训练 checkpoint。整个 foundation 使用 `requires_grad_(False)`、`eval()` 和 `inference_mode()`，以 float32 提取融合层的 256 维特征。

默认缓存：`data_cache/telephone_epoch010_frozen_rl_v1/`。`manifest.json` 是完成标记，`sources.json` 保存源文件清单，每个 `train/*.npz`、`val/*.npz` 文件保存一个 episode：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| features | `[T+1, 256]` | `s0 … sT` 的冻结特征 |
| action_mask | `[T+1, 81]` | 每个状态的合法 waypoint |
| action | `[T]` | 数据中的实际动作 |
| reward | `[T]` | collector 已折扣的多步回报 |
| discount | `[T]` | 终止为 0，其余为 `gamma ** num_steps` |
| terminated / truncated | `[T]` | 真终止 / 时间上限截断 |
| num_steps / gamma | `[T]` | 原始 SMDP 元数据 |

输入预处理与 foundation 训练一致：前一决策帧和当前 RGB/depth 拼接，episode 开始补零；RGB 除以 255；几何图无损解包；声音与声学图沿用模型内预处理。下一状态使用当前帧作为它的前一帧。检查相邻 transition 的观测与 mask 连续性，不混接 episode，不输入 angle/目标坐标标签。

## 训练方式

- Actor、Q1、Q2 各自有可训练 LSTM；两个目标 critic 包含独立的目标 LSTM，并整体 Polyak 更新。
- 每个 batch 包含完整 episode，右侧 padding 并屏蔽 padding 的全部 loss。每个训练 epoch 遍历全部训练 transition，验证遍历全部独立验证 transition。
- LSTM 从真实 episode 起点零状态开始。下一状态价值从同一次 `[s0, …, sT]` 前向的后移位置取得，保留正确历史。
- 离散 SAC 的目标为 `reward + discount * sum(pi * (min(target_Q1, target_Q2) - alpha * log(pi)))`。不再次乘 gamma；超时截断仍 bootstrap。
- CQL 对每个 critic 使用 `temperature * logsumexp(Q / temperature) - Q(data_action)`，仅对合法动作求和。
- Actor 使用完整合法动作分布及停止梯度的双 Q 最小值；熵温度自动学习，目标熵为 `entropy_ratio * log(合法动作数)`。默认比例 0.5、初始 alpha 0.1、CQL 权重 1.0。训练目标不含 BC loss。

算法参考本机 AVNav/SoundSpaces 的 `yz/net/sac_cql/trainer/SAC_CQL.py`，实现位于本目录，不依赖外部项目。针对 waypoint 的动作掩码、多步折扣、递归状态和目标网络独立实现。

## 输出与使用

默认输出：`runs/telephone_epoch010_offline_lstm_saccql_v1/`。

- `latest.pt`：最后完成轮次的全部训练状态。
- `best.pt`：独立验证集 critic loss 最低的 checkpoint。
- `best_policy.pt`：验证策略 NLL 最低的 checkpoint。
- `top_checkpoints.json`：上述两份模型和最终轮次组成的闭环验证候选。
- `best_spl.pt`：运行下述闭环选模命令后生成，按验证 SPL 选定。
- `steps.jsonl`（每次更新）、`metrics.jsonl`（每轮）、`tensorboard/`：TD/CQL/actor loss、熵、alpha、数据动作 Q 值及动作一致率。
- `config.json`、`dataset.json`：运行配置及数据来源。
- `completed.json`：目标总轮数完成标记。

离线 critic loss 和动作一致率不能代替闭环 Success/SPL；`best.pt` 仅表示上述离线选择标准。

完成下述闭环选模后，载入选定的 policy（示例导入不触发顶层 `experiment` 的仿真环境导入）：

```python
import sys
import torch
sys.path.insert(0, 'experiment')
from avwan.offline.sac_cql import RecurrentSACCQL

state = torch.load('experiment/avwan/runs/telephone_epoch010_offline_lstm_saccql_v1/best_spl.pt',
                   map_location='cpu', weights_only=False)
signature = state['signature']
policy = RecurrentSACCQL(signature['feature_dim'], signature['action_count'],
                         state['config']['training'])
policy.load_state_dict(state['model'])
policy.eval()
hidden = None  # 每个 episode 开始时重置
# feature: [B,256]，mask: [B,81]，来自相同冻结 foundation / 预处理
action, hidden = policy.act(feature, mask, hidden)
```

回归验证：

```bash
python experiment/avwan/tests/test_offline_rl.py
```

覆盖冻结参数/BN 不变、批量特征一致性、前后帧对齐、提取恢复、训练恢复逐参数一致、动作 mask、SMDP 目标、熵梯度方向、目标网络更新和 padding 不影响 loss。

## 闭环选模与随机 100 条 test 评估

```bash
python -m experiment.avwan.evaluate_candidates \
  --run experiment/avwan/runs/telephone_epoch010_offline_lstm_saccql_v1 \
  --foundation-checkpoint experiment/avwan/runs/telephone_train_val_20260925/epoch_010.pt \
  --validation-concurrency 2 --test-workers 2 --test-concurrency 2 \
  --test-max-episodes 100 --seed 20260925
```

先对候选 checkpoint 在独立的 100 条验证轨迹上比较 SPL。按验证 SPL、验证成功率、离线 critic loss 依次打破平局，选择完成后再对固定随机种子抽出的 100 条 `test_telephone` 轨迹评估。选择清单、逐 episode SPL、checkpoint/foundation SHA-256 均保存在 `rollouts/`；测试结果不参与选模。每 25 个 episode 或达到内存阈值重启仿真进程，依据已保存清单续跑。

SAC actor 每个 episode 重置 LSTM hidden state 和双帧视觉历史，foundation 始终冻结；部署时校验 foundation 与训练缓存的 SHA-256 完全一致。

导出 actor/critic/TD/CQL loss、熵和 alpha 的 CSV 和曲线：

```bash
python experiment/avwan/offline/report.py \
  --run experiment/avwan/runs/telephone_epoch010_offline_lstm_saccql_v1
```
