！禁止阅读trash/
python 环境为 ss
1. 数据位置：
    1.1 train数据集：/data/getuanhui/avnav-hybrid-mutiple/pickle
    1.2 val数据集： /data2/getuanhui/avnav-codex-val
2. 实验所有内容存放位置: /home/getuanhui/project/hybrid_offline_RL/result/2026-10-28-avnav-codex_attemp
3. foundaiton model 位置：experiment/avnav/hybrid , 不可更改模型
4. offlineRL 位置：experiment/avnav/offline_rl_v1_5_lstm ， 不可更改模型

实验pipeline：充分利用GPU和CPU加快实验运行速度。
1. 在不改变模型结构的情况下训练avnav的模型。offlineRL最终test spl要超过0.25.unheard超过0.15就可以
2. 在测试的时候不能使用rule base的方式。
3. 你可以尝试任何方式进行训练，增加数据，调整训练方式，修改训练loss等一系列内容。如果训练出来的model没有达到要求的spl，就必须要再次寻找其他方式更高的model。
4. 没有达到目标之前，必须持续工作。
5. 必须要充分利用gpu和cpu资源，尽量跑满，加快实验速度。
7. 监控输出token，如果达到百万输出就终止实验