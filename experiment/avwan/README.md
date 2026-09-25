# AVWAN Foundation Model 训练

训练使用 `model/hybrid.py` 中的 `Network`，在用户更新的视听 Transformer 架构上，
加入 AVWAN 几何地图、声学地图和 waypoint 可行性信息。
所有训练配置统一在 `config/audiogoal.yaml` 的 **TRAIN** 节点。

## 从 run_avwan.py 开始训练

修改 `experiment/avwan/config/audiogoal.yaml`：

```yaml
TRAIN:
  OPEN: true
  STAGE: all
  RESUME: ''
  # 其余 DATA、EPOCHS、BATCH_SIZE 等配置保留文件中已有值
```

在项目根目录执行：

```bash
conda activate ss
python run_avwan.py
```

- `TRAIN.OPEN: true`：直接进入训练流程，优先于 COLLECT，不创建仿真环境或加载仿真 episode。
- `TRAIN.OPEN: false`：按照 `COLLECT.OPEN` 执行原有采集流程。
- `TRAIN.STAGE: all`：首次转换原始采集数据，然后训练；完整缓存可直接复用。
- `TRAIN.STAGE: prepare`：只转换缓存。
- `TRAIN.STAGE: train`：要求缓存已完成，再训练。

仓库默认 `TRAIN.OPEN` 保持 false，开启后上述命令即可训练。
`TRAIN.DATA.SOURCE` 明确指定训练采集目录，与用于采集的 `DATASET.SPLIT`、
`COLLECT.TO_PATH` 独立。例如采集 val_telephone 时，训练源仍可保持 telephone train。
原来的独立 `config/train.yaml` 已移除。

若需要通过单独的实验覆盖文件启动，使用 `TASK_CONFIG.TRAIN` 层级：

```yaml
TASK_CONFIG:
  TRAIN:
    OPEN: true
    EPOCHS: 10
    OUTPUT: runs/my_hybrid
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
默认按场景留出 10% 验证组，保证场景不跨 train/val；这不是官方 val/test split。

缓存 v3 包含 gm/am，默认目录为 `data_cache/telephone_foundation_spatial_v3`。
不能直接复用旧 v2 缓存或旧模型 checkpoint，旧输出文件保留。
源文件集合、大小、mtime 和数据配置参与缓存身份校验；变化后选择新缓存目录。
转换完成才写 manifest，中断留下的未完成缓存需换目录重新转换。

## 配置、结果与续训

`TRAIN.DATA` 管理 SOURCE、CACHE、INPUTS、SHARD_SIZE、CONVERT_WORKERS、
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
