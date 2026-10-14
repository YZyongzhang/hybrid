# AVWAN 模型训练

## 训练方式

AVWAN 将双帧 RGB/depth、双耳 spectrogram、几何图、声学图和 81 维 waypoint action mask 编码后预测 waypoint。推荐复现仓库中已完整跑通的两阶段 foundation + 离线 RL 配方：先训练声学方向表示，再冻结音频编码器训练导航动作，最后用 SAC-CQL 学习时序策略。训练和验证只用 train/validation 数据；测试集只在选定 checkpoint 后评估。

## 复现步骤

在项目根目录使用已有的 ss 环境。配置中的数据路径位于 /data，需确保这些目录可访问；新实验请为输出和缓存指定新的目录，避免覆盖已有结果。

1. **训练 foundation**（固定随机种子 20261012）：

   ~~~bash
   conda activate ss
   python run_avwan.py --config experiment/avwan/config/foundation_two_stage.yaml
   ~~~

   配置为 batch 8、workers 0、音频和动作阶段各 5 epoch。音频阶段用梯度反转（GRL）降低声源类别信息，同时学习 8 向声学方向；按 heard 验证集方向 loss 选 checkpoint。动作阶段加载该音频权重并冻结音频编码器，训练 waypoint 动作头，按验证动作 loss 选出 foundation。默认输出：result/2026-10-13-staged-grl-reproduction/foundation/。

2. **训练离线策略**：

   ~~~bash
   python run_avwan.py --config experiment/avwan/config/offline_rl_reproduction.yaml
   ~~~

   该步骤用 foundation 提取并缓存特征，再训练一层 LSTM 的 SAC-CQL。固定设置为 seed 20261012、20 epoch、每轮 500 steps、batch 32；样本按 [16, 8, 8] 混合采样。BC 权重 5、CQL 权重 1、gamma 0.01、学习率 1e-4、梯度裁剪 10，并启用 foundation prior 和终止 mask。策略 checkpoint 按独立验证集选择。配置中的 FRESH_FEATURES: true 会在本次 run 目录下重建特征缓存。

3. **完整测试**（heard 与 unheard 各 1,000 episodes）：

   ~~~bash
   python run_avwan.py --config experiment/avwan/config/test_staged_grl.yaml
   ~~~

   汇总报告写入 result/2026-10-13-staged-grl-reproduction/full_test/report.json。若为新实验改了输出目录，要同步修改 foundation_two_stage.yaml 的 OUTPUT、offline_rl_reproduction.yaml 的 CACHES.* 和 RUN.OUTPUT，以及 test_staged_grl.yaml 的 OUTPUT、CHECKPOINT 和 FOUNDATION，指向新 run 的产物。

## 稳定性与结果

这套配方使用小 batch、单进程数据加载和资源上限，适合先复现再扩展：foundation 配置限制 GPU 显存占用为 22%、CPU 线程为 2、进程 RSS 不超过 12,000 MiB，并要求至少 12 GiB 可用内存；离线特征提取也采用小 batch 和低并发。保持配置中的 seed 和超参数，先跑完这套基线，再单独调整参数。

已记录的完整测试没有运行错误：test_multiple SPL 0.340、成功率 51.6%；test_multiple_unheard SPL 0.319、成功率 48.4%。这次结果未达到报告中的 SPL 0.4 目标；这些指标用于核对复现，不应当作性能保证。训练日志、验证选择和审计文件均在上述 run 目录下。
