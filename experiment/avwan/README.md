# AVWAN Foundation Model 训练

Foundation 完成后，可通过 `run_avwan.py` 配置并启动 refined offline RL。当前
`audiogoal.yaml` 默认 `TRAIN.MODE: offline_rl`；foundation、parent、数据源、缓存和
训练参数都在 `TRAIN.OFFLINE_RL` 下。

```bash
python run_avwan.py
```

此流程会提取三组冻结特征，按配置训练 SAC-CQL actor，然后执行数据审计和 heard/unheard
验证。若要恢复 foundation 监督训练，把 `TRAIN.MODE` 改为 `foundation`。普通 SAC-CQL
基线仍可直接运行 `python experiment/avwan/train_offline_rl.py`，参数见 `config/offline_rl.yaml`。
refined 配方的 `GAMMA`、`BC_WEIGHT`、`CQL_WEIGHT`、熵参数、学习率、target 更新率、
梯度裁剪、终止 mask、foundation prior、batch 和失败 STOP penalty 都可在
`TRAIN.OFFLINE_RL.TRAINING` 中调整。`HIDDEN_DIM`、`LSTM_LAYERS` 必须与 `PARENT` actor 一致。

训练使用 `model/hybrid.py` 中的 `Network`，在用户更新的视听 Transformer 架构上，
加入 AVWAN 几何地图、声学地图和 waypoint 可行性信息。
Foundation 训练配置统一在 `config/audiogoal.yaml` 的 **TRAIN** 节点。

## 从 run_avwan.py 开始训练

训练模式和参数配置在 `experiment/avwan/config/audiogoal.yaml`：

```yaml
TRAIN:
  OPEN: true
  MODE: offline_rl
  FOUNDATION:
    STAGE: all
    # foundation 专属 DATA、OUTPUT、EPOCHS 等配置
  OFFLINE_RL:
    STAGE: all
    FOUNDATION: experiment/avwan/runs/every_scene_1000_10_12/best.pt
    # SOURCES、CACHES、RUN、EXTRACTION、TRAINING、VALIDATION 也在此节点下
```

在项目根目录执行：

```bash
conda activate ss
python run_avwan.py
```

- `TEST.OPEN: true`：进入独立测试流程，优先于 TRAIN 和 COLLECT。测试 checkpoint、foundation、数据集及运行参数统一配置在根级 `TEST` 节点。
- `TRAIN.OPEN: true`：当 `TEST.OPEN` 为 false 时进入训练流程，不创建仿真环境或加载仿真 episode。
- `TRAIN.OPEN: false` 且 `TEST.OPEN: false`：按照 `COLLECT.OPEN` 执行采集流程。
- `TRAIN.MODE: offline_rl`：运行 `TRAIN.OFFLINE_RL` 中的特征提取、offline RL、审计和验证。
- `TRAIN.MODE: foundation`：运行 `TRAIN.FOUNDATION` 中的监督式 foundation 训练；此模式的 `TRAIN.FOUNDATION.STAGE` 可设为 `all`、`prepare` 或 `train`。
- `TRAIN.OFFLINE_RL.STAGE`：可设为 `all`、`extract`、`train`、`audit` 或 `validate`，便于中断后单独续跑某一阶段。

完整测试集评估由根级 `TEST` 配置控制，默认评估 `test_multiple` 和 `test_multiple_unheard`，每个 split 覆盖其全部 1,000 个 episode。测试环境直接读取 `BASE_TASK_CONFIG_PATH` 指向的 `audiogoal.yaml`；测试专属的 `PREDICTION_INTERVAL` 等参数在根级 `TEST` 中配置。将 `TEST.OPEN` 设为 `true` 并执行 `python run_avwan.py`，会只启动测试，不会进入训练或采集：

```bash
python run_avwan.py
```

结果写入 `TEST.OUTPUT/full_test/`，包括每个 split 的 SPL、成功率和汇总报告。

`TRAIN.OFFLINE_RL` 中的相对路径以仓库根目录为基准。提取完成的缓存会自动复用；训练目标目录已有未完成 checkpoint 时会从中续训。

若需要通过单独的实验覆盖文件启动，使用 `TASK_CONFIG.TRAIN` 层级：

```yaml
TASK_CONFIG:
  TRAIN:
    OPEN: true
    MODE: offline_rl
    OFFLINE_RL:
      STAGE: all
      # 可在这里覆盖 OFFLINE_RL 配置
```

```bash
python run_avwan.py --config experiment/avwan/my_overrides.yaml
```

## 模型结构

保留用户模型的 VisualEncoder、AudioCRNN、visual Transformer、融合 Transformer
及动作/角度分类结构，动作头由原来的 4 类适配为采集数据中的 81 个 waypoint 动作。

| 输入 | 形状（不含 batch） | 编码后 |
|---|---|---|
| 前一决策帧 + 当前 RGB/depth | RGB 128×128×6，depth 128×128×2 | 16 个视觉 token |
| 当前 spectrogram | 65×26×2 | 1 个音频 token |
| 当前 gm（障碍、已探索区域） | 400×400×2 | 新增 CNN → 16 个 token |
| 当前 am（声学地图） | 20×20×1 | 新增 CNN → 4 个 token |
| 当前 action_mask | 81 | 新增 MLP → 1 个 token |

38 个 token 加入模态标识和位置编码，由融合 Transformer 输出 **256 维特征**，
用于 waypoint 动作预测。action_mask 同时屏蔽非法动作。
地图是采集器提供的以智能体为中心的地图，不输入声源真值坐标。
几何/声学地图来自当前决策时刻，不能使用 next_observation。

音频八方向分类使用同一次音频前向的 token。默认联合目标：

```text
loss = ACTION_LOSS_WEIGHT × waypoint_cross_entropy
     + ANGLE_LOSS_WEIGHT  × audio_direction_cross_entropy
```

两项权重默认均为 1。`angle[1]`（弧度）只生成八方向标签，不作为模型输入。
角度标签使用 `floor(((degrees(angle[1]) + 180) % 360) / 45)`。
保留的 polar/融合角度 probe 不参与这两个训练损失。
音频按样本归一化；视觉空间位置通过正确的维度转换成为 token。
本阶段进行监督预训练，不训练 Q/V 或使用 reward/discount。

## 数据与 I/O

`data_pipeline/prepare.py` 用有界线程池将可信 `.pkl.gz` 转换成 `.npy` 分片：

- RGB 保持 uint8，depth、spectrogram、am 保持 float32。
- gm 必须是二值图，无损位打包后每份从约 1.22 MiB 缩为约 39 KiB；在 GPU 上展开。
- RGB/depth 每帧保存一次，用 `previous_index.npy` 引用前一决策帧。
  episode 首帧历史补零，跨 shard 时只补存必要的一帧，不跨 episode 使用历史。
- 训练用 mmap、按分片分配 worker、片内打乱、常驻 worker、有限 batch 预取、
  pinned memory 与非阻塞 GPU 传输。每轮每个样本使用一次，保留尾批。
- 不在训练时反复解压原始文件，不加载整个数据集到 Python 内存。

默认 `SHARD_SIZE: 512`、`WORKERS: 4`、`PREFETCH_FACTOR: 2`、`BATCH_SIZE: 64`。
缓存建议放本地 SSD。日志记录 `samples_per_second` 和 `data_wait_seconds`，
应以初始化后的完整 epoch 评估吞吐。
设置 `TRAIN.FOUNDATION.DATA.VAL_SOURCE` 时，全部 SOURCE 用于训练，VAL_SOURCE 独立用于验证，检查 episode 不重叠。
VAL_SOURCE 为空时才按场景留出 10% 验证组。

缓存 v3 包含 gm/am，默认目录为 `data_cache/telephone_foundation_spatial_v3`。
不能直接复用旧 v2 缓存或旧模型 checkpoint，旧输出文件保留。
源文件集合、大小、mtime 和数据配置参与缓存身份校验；变化后选择新缓存目录。
转换完成才写 manifest，中断留下的未完成缓存需换目录重新转换。

## 配置、结果与续训

`TRAIN.FOUNDATION.DATA` 管理 SOURCE、VAL_SOURCE、CACHE、INPUTS、SHARD_SIZE、CONVERT_WORKERS、
VAL_FRACTION、SPLIT_BY、MAX_EPISODES；其他训练参数直接位于 TRAIN 中。
其中 INPUTS 为 `[rgb, depth, spectrogram, gm, am]`。
所有相对 SOURCE/CACHE/OUTPUT/RESUME 路径均基于 `experiment/avwan`。

默认输出 `runs/telephone_hybrid/`：

- `config.json`、`dataset.json`：训练配置与缓存来源。
- `metrics.jsonl`：每轮动作/角度 loss、准确率及吞吐。
- `latest.pt`：最近完成 epoch 的模型、optimizer、AMP scaler 和随机数状态。
- `best.pt`：验证联合损失最低的完整状态。
- `foundation_weights.pt`：对应 best.pt 的纯模型权重。

续训时设置：

```yaml
TRAIN:
  OPEN: true
  STAGE: train
  RESUME: runs/telephone_hybrid/latest.pt
  EPOCHS: 60
```

`EPOCHS` 是总目标轮数，从最后完成的 epoch 继续。已有输出目录需要 RESUME 或新 OUTPUT。
离线分类准确率不代表导航成功率，后续仍需环境 rollout 评估。

特征接口（在 `experiment/avwan` 下运行 Python）：

```python
import torch
from model.hybrid import Network

model = Network(action_count=81).eval()
model.load_state_dict(torch.load('runs/telephone_hybrid/foundation_weights.pt',
                                map_location='cpu', weights_only=True))
# audio/rgb/depth/gm/am 为 float32，action_mask 为 bool，均含 batch 维。
# RGB 已除以 255；RGB/depth 双帧沿通道拼接，历史在前、当前在后。
with torch.no_grad():
    features = model.embedding_forward(audio, rgb, depth, gm, am, action_mask)  # [B,256]
    actions, angles = model.forward_joint(audio, rgb, depth, gm, am, action_mask)
```

## 验证与单独入口

```bash
python experiment/avwan/tests/test_training_pipeline.py
python experiment/avwan/train.py --max-episodes 256 \
  --cache data_cache/my_spatial_smoke --output runs/my_spatial_smoke --epochs 2
```

独立 `train.py` 也读取 audiogoal.yaml 的 TRAIN 配置，用于诊断时显式启动训练，
不受 OPEN 开关限制。`MAX_EPISODES` / `MAX_BATCHES` 为 0 时使用完整数据。
根入口使用现有 ss 环境；独立训练入口仅需 PyTorch、NumPy、PyYAML。

## TensorBoard 与闭环导航评估

单独评估时，先在 `config/evaluate.yaml` 设置 checkpoint、数据 split 和输出目录，然后直接运行：

```bash
python experiment/avwan/evaluate.py
```

离线 recurrent SAC 策略还需在该 YAML 中设置 `foundation_checkpoint`。默认参数见
`experiment/avwan/config/evaluate.yaml`；必要时也可以用命令行参数临时覆盖。

`TRAIN.FOUNDATION.TENSORBOARD: true` 将训练/验证的总 loss、动作 loss、方向 loss、准确率和吞吐
记录到 `TRAIN.FOUNDATION.OUTPUT/tensorboard/`。例如：

```bash
tensorboard --logdir experiment/avwan/runs/telephone_train_val_20260925/tensorboard
```

`TRAIN.SAVE_TOP_K: 3` 维护验证 loss 最低的三个候选 checkpoint，记录在
`top_checkpoints.json`。训练结束后运行：

```bash
python -m experiment.avwan.evaluate_candidates \
  --run experiment/avwan/runs/telephone_train_val_20260925 --test-workers 4 --test-concurrency 2
```

该流程在采集 val 的相同 episode 上比较三个候选模型的实际导航 SPL，以最高验证 SPL
选择一个 checkpoint，再对完整 test_telephone split 评估一次。测试结果不参与选模。
策略仅根据观测和 action mask 选择 waypoint，底层 planner 负责执行，不调用专家选点。
评估脚本逐条保存轨迹、SPL、成功率、错误和选择清单；汇总输出到 `rollouts/`。
SoundSpaces 原 reader 每次截取 100 条，评估器分批解析，从而覆盖完整选定集合。
`--test-workers` 指定 episode 分片数量，`--test-concurrency` 限制同时运行的进程数
（默认 2，大型声学场景占用较多内存）。续跑时保持分片数量不变，可调低并发。
协调器默认每 25 条新轨迹或进程峰值 RSS 达到 8192 MiB 后，在 episode 边界重启
仿真进程，释放原生内存；分别通过 `--process-episodes` 和 `--process-rss-mb` 调整。
逐条固定随机种子，汇总时校验重复和遗漏；支持从已有输出续跑。
这里的闭环环境是 Habitat/SoundSpaces 仿真。

本次实验目录为 `runs/telephone_train_val_20260925/`。`best.pt` 和
`foundation_weights.pt` 按离线验证 loss 选择；`best_spl.pt` 按验证闭环 SPL 选择。
完整评估结束后的指标见该目录 `RESULTS.md` 与 `rollouts/final_results.json`，
训练曲线见 `training_curves.png`。
