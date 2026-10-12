# AVWAN 数据采集

配置使用 `experiment/avwan/config/default.py` 的 `get_config()`。
`run_avwan.py` 创建 dataset 和环境，再调用 `COLLECTER(config, env).collect()`，
与 AVNAV 一样由 `COLLECT.TYPE` 选择采集器。

```bash
# 在项目根目录、ss 环境运行，需要 Habitat GPU/EGL
python run_avwan.py
python run_avwan.py --config experiment/avwan/config/default.yaml
```

## 数据加载与遍历

`GreedyCollect` 遍历 `env.habitat_env.episodes`，每个 episode 尝试一次，
复用同一个环境并在下一条数据开始前 reset。episode 出错会记录错误并继续后续数据。
旧配置中的 `COLLECT.NUM_EPISODES` 仅为兼容而保留，不额外限制环境中已有的 episode 数量。

与 AVNAV 一样直接使用原始 SoundSpaces `AudioNavDataset`，不继承或重写 `from_json()`。
当前原始实现先随机打乱 episode，再通过 `[:100]` 每个场景最多加载 100 条。
启动入口不覆盖环境的 `NUM_EPISODE_SAMPLE`、`SHUFFLE` 或 `CYCLE` 配置。

采集范围由原 `DATASET.SPLIT`、`DATASET.DATA_PATH`、`DATASET.CONTENT_SCENES` 决定。
采集器遍历的是环境实际加载的 episode，不代表源文件中的全部数据，也不会跨越到其他 split。
如显式设置 `COLLECT.EPISODE_ID` 或 `SELECTION_PATH`，则只收集指定案例；
正常采集应让它们保持为空。

## 配置与输出

- `COLLECT.TYPE`：`greedy` 或 `avwan_greedy` 使用 BFS waypoint 专家；`diverse`、`deverse` 或 `avwan_diverse` 使用 `DeverseCollect`，按 episode 混合 BFS 专家、轻度探索和较强探索。
- `COLLECT.TO_PATH`：保存目录；为空时使用 `OUTPUT_DIR`。
- `COLLECT.OPEN`：是否采集。
- `COLLECT.SUCCESS_ONLY`：默认 false，保留失败或超时的有效轨迹。
- `COLLECT.VISUALIZE`：保存现有 waypoint TopDownMap 图片。
- `COLLECT.RESUME`：按场景续采。为 true 时，扫描当前输出目录，跳过已有采集数据的整个场景；
  即使该场景只采了一部分也跳过。空目录、仅有 `.tmp` 文件或仅有错误记录不算已采集。
  为 false 时，从头采集，要求输出目录为空。`audiogoal.yaml` 已开启续采。
  场景集合只在启动时扫描一次，新场景仍会采完本次加载的全部 episode。
  不要求随机抽样结果或代码哈希与旧任务相同，可直接接着原来的全量采集目录使用。
  输出目录应继续用于同一数据集和任务；旧文件与 `run.json` 保留，索引追加新记录，
  新任务配置保存到 `resume_runs/`。manifest 累计历史数量，planned 为历史已完成数加本次待采数。

底层执行步数上限仍由 `TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS` 控制，
waypoint 动作周期由 `PREDICTION_INTERVAL` 控制，折扣由 `OFFLINE_RL.GAMMA` 控制。
采集器已不再使用逐 episode 子进程，旧 `TIMEOUT_SECONDS` 不再控制运行时间；
数据集路径使用 `DATASET.DATA_PATH`，旧 `CONTENT_DIR` 不再单独加载其他文件。

```text
output/
  run.json                # 完整配置、代码哈希、数据集标识
  manifest.json           # 总数、已尝试数、成功/错误数和保存数
  episodes.jsonl          # 每行一个 episode 结果，避免全量结果驻留内存
  episodes/场景名/*.pkl.gz  # 按场景分目录，文件名为 episode 编号
  topdownmaps/*.png        # waypoint 图片
  logs/*.json             # 每个 episode 的状态、指标或错误
```

```python
from experiment.avwan.collector.collect import load_episode

data = load_episode('保存目录/episodes/场景名/episode编号.pkl.gz')
for t in data['transitions']:
    obs, next_obs = t['observation'], t['next_observation']
    action, reward, discount = t['action'], t['reward'], t['discount']
    # TD target = reward + discount * V(next_obs)
```

文件保留环境 `last_transition` 全部字段：前后完整观测、地图、action mask、
waypoint 动作、SMDP 奖励、底层动作和终止标记，另存专家决策、waypoint 到达情况等。
默认动作空间为 81 个 waypoint，40 为 STOP。完整观测中的目标角度及元数据中的
声源坐标是目标真值，训练时应明确选择模型输入。只加载可信来源的 pickle。

`DeverseCollect` 的探索动作始终从当前 `action_mask` 的有效 waypoint 中选取，
并按历史动作频数降低重复动作的采样权重；每个 episode 记录 `collection_mode`、
`exploration_probability` 和逐步的 `behavior_policy` / `expert_action`。随机 STOP
以低概率提供提前停止样本，正常专家 STOP 仍保留。输出继续包含 offline RL 所需的
完整 SMDP transition，不会合成 reward、next observation 或 terminal/timeout 标记。

任务终止时折扣为 0；超时或专家不可达的采集截断保留 bootstrap 折扣。
遍历环境已加载的 episode，不保证每个 episode 都能成功到达声源。
