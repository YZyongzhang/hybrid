# AV-WaN 离线 RL 环境

`environment/avwan_env.py` 注册 `AVWANRLEnv`。它复用 SoundSpaces 的真实
`Planner` / `Mapper`，支持中间点策略的数据采集与环境评估。
离线训练通过 `offline.AVWANOfflineDataset` 读取 PT，不创建模拟器。
本实现没有添加 PPO、IQL 或 CQL 训练算法。

所有实现、配置与输出均在当前项目；SoundSpaces / Habitat 作为已安装的只读依赖使用。

## 中间点动作与终态

- 动作空间为 `Discrete(ACTION_MAP.MAP_SIZE ** 2)`，默认 9×9 共 81 个动作。
- 动作 ID 按 agent 相对地图的行优先顺序排列。中心格 `40` 是 STOP，**不是底层动作 0**。
- Planner 最多执行 `PREDICTION_INTERVAL` 个底层动作，默认 10 步。
  提前到达中间点、发现无法到达或 episode 结束时退出。
- 不可达目标保留官方第一步 fallback 机制，保证每条 transition 至少执行一步。
- `step(action)` 支持整数、`step(action=整数)` 和 `step({'action': 整数})`。
  返回 `(observation, cumulative_reward, done, info)`；奖励是与官方一致的底层奖励之和。
- **step 内不自动 reset**。done 后返回该 episode 的真实最终观测和最终地图；
  再次 step 会报错，必须显式 `reset()`。reset 同时清空 Planner 地图和图状态。
- 成功 STOP、失败 STOP、任务自身结束标记为 `terminated`；仅时间/步数上限结束标记为
  `truncated`。到达中间点不会单独结束 episode。
- `CONTINUOUS=False` 使用官方离散图的 `sim.reaching_goal`；连续模式使用成功距离阈值。
- `action_mask()` 返回局部占据地图的可选动作，STOP 始终可选；局部可通行不保证全局图可达。
- 兼容本机缓存的 `H×W×1×1` depth（统一为 `H×W×1`）和
  `(spectrogram, waveform)` 音频元组（策略输入只取频谱）。

## 离线数据接口

`env.step_transition(action)` 执行一次中间点动作，返回独立快照字典。
如果使用 `env.step(action)`，则可以从 `env.last_transition` 取出同样的数据。
两者都复制观测，后续 step/reset 或调用方改动数组不会污染之前的数据。

| 字段 | 含义 |
|---|---|
| `observation`, `next_observation` | 中间点动作开始/结束的观测，含地图 |
| `action` | 中间点 ID |
| `num_steps` | 实际执行的底层动作数 k |
| `low_level_actions`, `low_level_rewards` | 长度 k 的底层动作和奖励序列 |
| `reward` | `r0 + gamma*r1 + ... + gamma**(k-1)*r[k-1]` |
| `undiscounted_reward` | 与 step 返回值相同的底层奖励之和 |
| `discount` | terminated 时为 0，否则为 `gamma**k` |
| `terminated`, `truncated`, `done` | done = terminated or truncated |
| `gamma` | 每个底层时间步的折扣，默认 0.99 |
| `action_mask`, `next_action_mask` | 动作前后的局部可通行掩码 |
| `episode_id`, `scene_id` | 数据归属 |

SMDP（变长动作）目标示例：

```python
target = batch['reward'] + batch['discount'] * next_value
```

不要再乘一次 gamma 或 `(1 - done)`：真正终止的折扣已归零，时间截断保留 bootstrap。
如果需要复现官方 PPO 按每个中间点一步折扣的目标，则应使用
`undiscounted_reward` 和每个中间点的 gamma；两种目标不要混用。

## 数据采集与读取示例

在 `ss` 环境、项目根目录中运行。音源与 RIR 路径需要指向真实可用的本地数据。
`get_config()` 将资源相对路径解析到 SoundSpaces 根目录，不会切换工作目录。
也可以把已有完整的 AV-WaN 配置直接传给环境。

```python
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from experiment.avwan.config import get_config
from experiment.avwan.environment.avwan_env import AVWANRLEnv
from experiment.avwan.offline import save_transitions, AVWANOfflineDataset

cfg = get_config(split='train_telephone')
cfg.defrost()
cfg.TASK_CONFIG.SIMULATOR.AUDIO.SOURCE_SOUND_DIR = '/actual/source/sounds'
cfg.TASK_CONFIG.SIMULATOR.AUDIO.BINAURAL_RIR_DIR = '/actual/binaural_rirs'
cfg.freeze()

with AVWANRLEnv(cfg) as env:
    env.reset()
    episode = []
    while True:
        # 示例行为策略；正式离线数据应换成你的专家/混合策略。
        action = int(np.random.choice(np.flatnonzero(env.action_mask())))
        transition = env.step_transition(action)
        episode.append(transition)
        if transition['done']:
            break
    save_transitions(episode, Path('outputs/avwan_offline/episode_0001.pt'))

# 以下代码不依赖 Habitat 或 SoundSpaces，训练期间不启动环境。
dataset = AVWANOfflineDataset('outputs/avwan_offline')
batch = next(iter(DataLoader(dataset, batch_size=16, shuffle=True)))
```

建议每个 episode 保存一个 PT。保存函数拒绝覆盖已有文件，读取使用 `weights_only=True`。
Dataset 返回固定训练字段，变长底层动作序列通过 `load_transitions()` 读取。
默认策略观测仅包含已有的 `rgb/depth/spectrogram/gm/am/action_map`，不包含真实目标坐标或角度；
可通过 `observation_keys` 指定输入字段。RGB 保留 uint8，归一化由策略预处理完成。

之前的 AVNav PT 只有四类底层动作和监督标签，不是中间点 RL transition，不能直接用于此接口。
需要重新采集或依据完整轨迹、地图和动作分段重新构造，不能仅修改 action ID。

## 验证

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -p 'test_avwan_env.py' -v
python -m experiment.avwan.smoke --source-sound-dir /actual/sounds --rir-dir /actual/binaural_rirs
```

本机 SoundSpaces 的 `data/binaural_rirs` 指向不存在的位置，默认 `data/sounds/1s_all` 也不存在。
因此完整音频模拟器检查需要补充可用的资源路径；这里不修改外部软链接。
可以使用已有缓存的视觉观测检查真实 Habitat、SoundSpaces 与 Planner 的导航链路：

```bash
python -m experiment.avwan.smoke --visual-only \
  --source-sound-dir /home/getuanhui/project/sound-spaces/data/sounds_old/1s_all \
  --output outputs/avwan_smoke/visual_episode.pt
```

`--visual-only` 明确禁用音频传感器，只用于检查地图、动作、终态和存取接口，不代表音频导航性能。

如需将依赖缓存也限制在当前项目内，可在运行前设置：

```bash
export PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR="$PWD/.cache/numba"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
```

已验证：13 项 AV-WaN 测试通过，包括真实 Planner 的前向坐标、碰撞位姿和声学地图更新，
以及禁止导入 Habitat/SoundSpaces 时的离线数据读取。
真实视觉场景 smoke check 执行 4 个中间点动作并保存、读回 transition，显式 reset 成功。
完整音频联调因缺失的 RIR 资源未运行。

执行全项目测试时还发现一项已有 AVNav 测试失败：
`hybrid_train.py` 中逐轮 `torch.save` 在本次修改前已被注释，导致找不到 `model_epoch_1.pth`。
本次未更改该 AVNav 训练逻辑。
