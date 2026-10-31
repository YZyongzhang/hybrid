！禁止阅读trash/ 和result下的内容，除非明确给出了可以阅读的指示
python 环境为 ss
1. 数据位置：
    1.1 train数据集：/data/getuanhui/avnav-hybrid-mutiple/pickle
    1.2 val数据集： /data2/getuanhui/avnav-codex-val
2. 实验所有内容存放位置: /home/getuanhui/project/hybrid_offline_RL/result/2026-10-28-avnav-codex_attemp_sr_3
3. foundaiton model 位置：experiment/avnav/hybrid , 不可更改模型
4. offlineRL 位置：experiment/avnav/offline_rl_v1_5_lstm ， 不可更改模型

实验pipeline：充分利用GPU和CPU加快实验运行速度。
1. 在不改变模型结构的情况下训练avnav的模型。offlineRL最终test spl要超过0.25.unheard超过0.15就可以。sr的话，需要heard达到50%以上，unheard也越高越好。
2. 在测试的时候不能使用rule base的方式。
3. 该轮次训练主要解决碰撞问题，尝试方向有两个 1. 通过大规模增加碰撞探索离开碰撞这种状态的数据，然后让foundation能够很好的提取到碰撞特征，并且用这种解决碰撞的轨迹，微调reward,并且微调之前已经训练好的offlineRL,让offlineRL进行重新学习，从而解决碰撞的能力。 2. 朝着lstm记忆方向入手，让lstm能够学到多步碰撞之后，主动的执行探索的能力。
3.1 offlineRL ：/home/getuanhui/project/hybrid_offline_RL/result/2026-10-28-avnav-codex_attemp/checkpoints/signed_return_seq32_weight3_20261030_epoch_012/best_policy.pt
foundation：/home/getuanhui/project/hybrid_offline_RL/result/2026-10-28-avnav-codex_attemp/checkpoints/foundation_native_spectral_20261030/epoch_006.pth

4. 没有达到目标之前，必须持续工作。
5. 必须要充分利用gpu和cpu资源，尽量跑满，加快实验速度。
7. 监控输出token，如果达到百万输出就终止实验，或者是1000万输入就终止实验。最后终止实验的时候，必须给出明确的可以执行的方向。
8. 允许在一次测试集上发现效果不好，然后重新调整model。只要测试集不参与训练，并且选择model的时候不依赖于测试集，都是允许的。
9. 如果需要重新收集数据，收集的数据放到/data2下面，并在本次的result下面保持引用。