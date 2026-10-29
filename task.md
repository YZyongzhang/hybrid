！禁止阅读trash/ result。除非指令明确给出可以阅读的路径
python 环境为 ss
充分利用GPU和CPU加快实验运行速度。
1. 数据位置：
    1.1 train数据集：/data/getuanhui/avnav-hybrid-mutiple/pickle
    1.2 val数据集： /data2/getuanhui/avnav-codex-val
2. 实验所有内容存放位置: /home/getuanhui/project/hybrid_offline_RL/result/2026-10-28-avnav-codex_attemp_rgb
3. foundaiton model 位置：experiment/avnav/hybrid , 不可更改模型
4. offlineRL 位置：experiment/avnav/offline_rl_v1_5_lstm ， 不可更改模型

实验任务：
1. 阅读result中的2026-10-28-avnav-codex_attemp。找出最好模型训练的那一版本的方式。然后在experiment/avnav/下创建一个rgb-only。里面的foundation和offline代码几乎和另外两个一致。唯一区别就是输入改成了rgb。
2. 训练rgb的avnav模型。如果效果不好，请不断迭代进行实验。一直可以让在完整的test集合上heardspl到0.25以上。unheard0.15以上。
3. 训练不能使用val或者是test数据集进行训练，造成数据污染。
4. inputtoken不超过一千万，输出token不超过一百万。如果超过了，请停止实验。