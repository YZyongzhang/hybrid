# AVNav 数据转换与训练

根目录入口 `run_avnav.py` 调用 `experiment.avnav.run.main`，项目级
`configs.default.get_config` 加载本目录的 `config/default.yaml`。
依赖：PyTorch、NumPy、PyYAML、tqdm；启用 TensorBoard 时还需 tensorboard。
转换和训练不依赖 Habitat/SoundSpaces 环境。

先修改 `config/default.yaml`，也可通过 `--config my_avnav.yaml` 提供局部覆盖：

```yaml
TASK_CONFIG:
  PT:
    FROM_PATH: /your/pickle/train
    TO_PATH: data/avnav/train
    VAL_FROM_PATH: /your/pickle/val
    VAL_TO_PATH: data/avnav/val
    SHARD_SIZE: 256
  TRAIN:
    DATA_PATH: data/avnav/train
    VAL_DATA_PATH: data/avnav/val
    EPOCHS: 10
    BATCH_SIZE: 32
    DEVICE: auto
```

从项目根目录启动：

```bash
python run_avnav.py --config my_avnav.yaml --stage convert
python run_avnav.py --config my_avnav.yaml --stage train
python run_avnav.py --config my_avnav.yaml --stage all
```

不指定 `--stage` 时按 `PT.OPEN` 和 `TRAIN.OPEN` 执行，默认仅训练。
`all` 先转换再训练；请使 `PT.TO_PATH` 与 `TRAIN.DATA_PATH` 一致，独立验证集的
两个路径也应一致。数据和输出的相对路径均相对于项目根目录；`--config` 相对于当前目录。
默认原始数据路径是占位符，实际运行前需要填写。

转换递归读取 `.pkl` / `.pickle`。每个文件是一段完整 episode，包含：

- `obs`：N+1 个观测，最后一个为终态；每帧有 `rgb`、`depth`、`spectrogram`、`angle`。
- `action_id`：N 个动作，允许嵌套列表，动作 ID 为 0–3。
- `sound_id`：可选的 episode 标签或逐步标签。

RGB 为 H×W×3 uint8，depth 为 H×W×1 或 H×W，spectrogram 为 F×T×2
或 1×F×T×2，也支持采集器返回的 `(spectrogram, waveform)` 元组（仅取频谱）；
`angle[1]` 是弧度方位角。转换后 RGB 除以 255、角度换算为度；
RGB/depth 前后两帧沿通道拼接，每个 episode 第一帧的历史输入补零。
当前 Network 需要 128×128 的图像以及 65×26×2 的音频频谱。
仅使用可信来源的 pickle 数据。

输出 `foundation_model_shard_*.pt` 和包含标签映射、样本数的 `manifest.json`。
每个 PT 包含 `rgb`、`depth`、`audios`、`actions`、`angles`、`sound_ids`；
训练返回音频、RGB、depth、角度、动作五项。已有同字段的 PT 数据也可直接训练。
分片大小是目标值，为保留 episode 完整性可能超出；输出目录已有 PT 时会报错，
请使用新的输出目录。

未设置 `VAL_DATA_PATH` 时按 `VAL_RATIO` 随机划分整个分片，至少需要两个分片。
新转换的分片不会跨越 episode 边界；旧 PT 的 episode 边界需要由数据提供方保证。
`CACHE_SIZE` 控制每个 worker 缓存的分片数量。
训练每轮随机排列分片，并在每个分片内随机排列样本，避免全局随机采样反复读取大型分片；
所有样本每轮均使用一次，分片末尾不足一个 batch 的样本也保留。
存在 manifest 时从中读取样本数量，并在实际加载分片时核对，避免启动时扫描所有张量。

训练使用 `model/hybrid.py` 的 `Network` 和 `train_class/hybrid_train.py` 的 `Train`，
优化动作分类及八方向角度分类的交叉熵损失。每轮保存
`SAVE_DIR/model_epoch_<n>.pth`，并按验证集总损失保存 `best.pth`。
`metrics.json` 保存每轮训练/验证的样本加权损失、准确率、样本数和耗时。
`MODEL_PATH` 可加载本实现保存的模型权重进行初始化，
不恢复优化器状态或 epoch。音频注意力使用项目内实现，旧外部实现的检查点不保证兼容。

本机 multiple 数据实验使用 `config/multiple.yaml`：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -u -m experiment.avnav.run_multiple
```

该入口将 `/data/Getuanhui/avnav-hybrid-mutiple/pickle` 中的
`hybird(heard_train_multiple)`、`hybird(val_multiple)`、`hybird(val_multiple_unheard)`
分别转换到同级 `pt/train`、`pt/val`、`pt/val_unheard`，随后训练 10 轮。
训练仅使用 train，以 val 选择最佳模型，再在 val_unheard 上评估最佳模型。
权重、指标、配置副本及运行环境保存在 `experiment/avnav/result`。
已完成转换时可用 `--stage train`，只重新评估可用 `--stage evaluate`；
入口不会覆盖已有 PT 分片。

验证完整流程：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -v
```
