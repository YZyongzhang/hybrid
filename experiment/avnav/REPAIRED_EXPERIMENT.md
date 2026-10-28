# AVNav 修复实验

实验目录：`result/2026-10-27-avnav-repaired`。配置与候选在 `run_protocol.json` 中预先声明；旧实验与随机 smoke 模型不参与选模。

已修复样本间耦合的音频归一化、视觉 token 排列、位置编码公式及设备迁移、单样本 embedding 维度、joint forward 解包、SAC 更新的序列对齐与目标时序编码器。碰撞 BC 权重和历史输入作为训练消融项。碰撞、进展与超时统计仅用于诊断；环境始终执行原始 actor argmax。

1. `prepare_repaired_run` 重新核对 train 来源、验证/测试清单哈希、episode/scene 零交集及训练轨迹覆盖。
2. `train_codex_foundation` 从头训练音频 8 轮、融合模型 15 轮；验证 NLL 选择表征，融合阶段冻结音频。
3. `extract_codex_features` 使用同一特征处理读取 train/validation，并审计训练碰撞后 16 步内取得进展的覆盖。测试数据不用于提取训练特征。
4. `run_repaired_pipeline` 训练 4 个逐项消融候选，每个保留第 10/20/30 轮 checkpoint。所有候选在完整 heard/unheard validation 各 500 集闭环评估。按两者平均 SPL、平均 SR、固定候选顺序依次选模。
5. 写入 `frozen_selection.json`，冻结模型、配置、代码哈希和验证结果后，仅对所选 checkpoint 运行完整 heard/unheard test 各 1,000 集。test 不用于训练、调参、选择或调试候选。

运行入口：

```bash
/home/getuanhui/anaconda3/envs/ss/bin/python -m unittest tests.test_avnav_repaired -v
/home/getuanhui/anaconda3/envs/ss/bin/python -m experiment.avnav.run_repaired_pipeline --run-dir result/2026-10-27-avnav-repaired
```

运行中的阶段和进程号写入 `pipeline_status.json`；日志在 `logs/`。`validation_candidates.json` 保存所有候选闭环分数及结果哈希；最终 `final_results.json` 报告各 test 的集数、错误数、SR、SPL、碰撞率与超时率。任何缺集、重复集或评估错误均不能作为通过的选模或最终结果。

默认基准上限 200 步，种子 20261027，动作编号为 STOP=0、前进=1、左转=2、右转=3。RGB/depth 传感器仅声明一次，防止已安装模拟器重复追加传感器；未改变动作空间、成功定义或动作处理。音频、RGB、depth 和模型自身上一动作是策略输入，地理距离和碰撞指标不进入策略。

token 上限为 1,000,000 输出 token；任务计量采用更保守的总 token 预算，并限制返回日志长度。计量器处于 UI 暂停状态时，其读数不是实时值，必须注明最后有效读数，不能虚报实时用量。
