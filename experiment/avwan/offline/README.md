# Frozen foundation → LSTM SAC-CQL

## 批量训练参数

修改默认配置 `experiment/avwan/config/offline_rl_batch.example.yaml`。文件里显式列出随机种子、设备、特征提取设置和全部训练参数。所有 `training.*` 参数都在 `grid` 里：单值列表表示固定该参数，多值列表表示扫描候选值。运行：

```bash
python -m experiment.avwan.train_offline_rl_batch
```

默认读取 `experiment/avwan/config/offline_rl_batch.example.yaml`；需要切换批量配置时再传 `--config PATH`。

`grid` 按笛卡尔积组合，而不是按位置配对。当前 `gamma` 有 2 个值、`cql_weight` 有 2 个值，其余训练参数各有 1 个值，因此会运行 `2 × 2 × 1 × … = 4` 组：每个 gamma 都分别搭配每个 cql_weight。再给某参数增加 3 个候选值，组数就变成 `4 × 3 = 12`。自动生成的 run 目录名使用发生变化的参数名和值，例如 `gamma-0p7__cql_weight-0p5`。脚本先提取一次冻结特征，再逐组串行训练；每组有独立输出目录、配置和日志。TensorBoard event 文件集中写入 `output_root/tensorboard/<run-name>/`，可用一个 `--logdir output_root/tensorboard` 查看整批实验。若要手动指定几组一一对应的参数组合，使用 `experiments` 命名列出。中断后重跑会复用特征缓存、跳过已完成实验，并从 `latest.pt` 恢复未完成训练。

用 `--dry-run` 查看展开后的参数组合，或用 `--skip-extract` 直接使用已有缓存。批量 YAML 中未填写的设置才从 `base_config` 补齐；其中相对数据、权重和缓存路径以 `experiment/avwan` 为根，`base_config` 相对批量 YAML 文件，`output_root` 相对 `experiment/avwan`。

2026-09-30 的固定 heard/unheard 100 episode 实验见
`result/2026-09-30-multiple-gamma-study/PROTOCOL.md` 和 `RESULTS.md`。
入口为 `python -m experiment.avwan.run_gamma_study`，汇总为
`python -m experiment.avwan.report_gamma_study`。新增可选训练参数：

- `terminal_mask: true`：真正终止不 bootstrap，timeout 保留 bootstrap；`gamma` 仍按决策步使用。
- `bc_weight: 1.0`：在 SAC actor loss 中加入数据动作的负对数似然正则。
- `foundation_prior: true`：冻结已有 foundation 动作头，初始化为零的 LSTM 残差修正其 logits。

上述选项默认关闭，原始 checkpoint 的部署行为不变。历史 checkpoint 没有
`training.gamma` 时，loss 使用其原始缓存 SMDP discount。
`evaluate --reset-memory` 可仅清空 actor 的历史状态，保留视觉双帧，用于记忆消融。
`python -m experiment.avwan.audit_offline_memory --help` 提供真实缓存轨迹上的数值检查。

在仓库根目录运行（不需要启动 Habitat 仿真器）：

```bash
/home/getuanhui/anaconda3/envs/ss/bin/python experiment/avwan/train_offline_rl.py
```

默认配置为 `experiment/avwan/config/offline_rl.yaml`，依次执行全量特征提取和 30 epoch 离线训练。YAML 相对路径以 `experiment/avwan` 为根；命令行路径以当前目录为根。

## 分阶段与恢复

```bash
# 冻结 foundation，提取全部训练 episode
python experiment/avwan/train_offline_rl.py --stage extract

# 只读取已完成的特征缓存训练，不再读取原始观测或运行 foundation
python experiment/avwan/train_offline_rl.py --stage train

# epochs 是目标总轮数；恢复模型、目标网络、全部优化器及随机状态
python experiment/avwan/train_offline_rl.py --stage train --epochs 60 \
  --resume experiment/avwan/runs/telephone_epoch010_offline_lstm_saccql_v1/latest.pt
```

可以通过 `--checkpoint`、`--source`、`--cache`、`--output`、`--device`、`--batch-size`、`--extract-batch-size` 覆盖设置。`--max-episodes N` 仅用于小规模诊断。已有训练目录须显式恢复，防止覆盖结果。

提取中断后重新运行相同命令即可续跑。缓存身份绑定 checkpoint SHA-256、源数据快照和预处理版本；改变权重或数据后需要新缓存目录。训练阶段以缓存内的 checkpoint 身份为准，不要求原始 checkpoint 仍在线。

## 特征格式

默认权重：`runs/telephone_train_val_20260925/epoch_010.pt`。支持纯 `state_dict` 或带 `model` 的 foundation 训练 checkpoint。整个 foundation 使用 `requires_grad_(False)`、`eval()` 和 `inference_mode()`，以 float32 提取融合层的 256 维特征。

默认缓存：`data_cache/multiple_train_only_rl_v1/`。`manifest.json` 是完成标记，`sources.json` 保存源文件清单；每个 `train/*.npz` 文件保存一个 episode：

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
- 每个 batch 包含完整 episode，右侧 padding 并屏蔽 padding 的全部 loss。每个训练 epoch 遍历全部训练 transition，不单独运行离线验证集。
- LSTM 从真实 episode 起点零状态开始。下一状态价值从同一次 `[s0, …, sT]` 前向的后移位置取得，保留正确历史。
- 离散 SAC 的目标为 `reward + discount * sum(pi * (min(target_Q1, target_Q2) - alpha * log(pi)))`。不再次乘 gamma；超时截断仍 bootstrap。
- CQL 对每个 critic 使用 `temperature * logsumexp(Q / temperature) - Q(data_action)`，仅对合法动作求和。
- Actor 使用完整合法动作分布及停止梯度的双 Q 最小值；熵温度自动学习，目标熵为 `entropy_ratio * log(合法动作数)`。默认比例 0.5、初始 alpha 0.1、CQL 权重 1.0。训练目标不含 BC loss。

算法参考本机 AVNav/SoundSpaces 的 `yz/net/sac_cql/trainer/SAC_CQL.py`，实现位于本目录，不依赖外部项目。针对 waypoint 的动作掩码、多步折扣、递归状态和目标网络独立实现。

## 输出与使用

默认输出：`runs/multiple_offline_lstm_saccql/`。

- `latest.pt`：最后完成轮次的全部训练状态。
- `epoch_001.pt` 等：每轮一个部署用 checkpoint，不含 optimizer/RNG 状态。
- `top_checkpoints.json`：列出已保存的 epoch checkpoint，供可选的闭环评估命令读取。
- `best_spl.pt`：仅在运行下述闭环评估后生成，按导航验证集 SPL 选定；训练本身不使用验证数据选 checkpoint。
- `steps.jsonl`（每次更新）、`metrics.jsonl`（每轮）、`tensorboard/`：TD/CQL/actor loss、熵、alpha、数据动作 Q 值及动作一致率。
- `config.json`、`dataset.json`：运行配置及数据来源。
- `completed.json`：目标总轮数完成标记。

训练日志中的 critic loss 和动作一致率不能代替闭环 Success/SPL。训练结束时 `latest.pt` 就是最终 checkpoint。

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

训练不需要离线验证数据。可选地，下面的闭环评估会在独立的导航验证 split 上测量各 epoch checkpoint 的 SPL/成功率，选定后再评估固定随机种子抽取的 `test_telephone` 轨迹；测试结果不参与选模。选择清单、逐 episode SPL、checkpoint/foundation SHA-256 均保存在 `rollouts/`；每 25 个 episode 或达到内存阈值重启仿真进程，依据已保存清单续跑。

SAC actor 每个 episode 重置 LSTM hidden state 和双帧视觉历史，foundation 始终冻结；部署时校验 foundation 与训练缓存的 SHA-256 完全一致。

导出 actor/critic/TD/CQL loss、熵和 alpha 的 CSV 和曲线：

```bash
python experiment/avwan/offline/report.py \
  --run experiment/avwan/runs/telephone_epoch010_offline_lstm_saccql_v1
```
