# task
## avwan和avnav数据收集
### avwan数据收集
----
1. 收集offlineRL在训练过程中需要的diverse的数据。包含各种失败的例子，恢复的例子。这些数据将用于提高对offlineRL的探索能力和最终测试能力。
2. 收集数据可以通过之前测试时表现不好的场景进行数据加强。由于不能直接使用test数据集进行收集数据。因此可以分析test上面为什么表现不好，找到这个原因，然后再trian数据集上进行重收集，并加强数据。例如：复杂场景。声源距离过长。碰撞问题。错误stop这些问题。
3. 在数据收集过程中，你需要进行以下的代码改写。1. 配置该次数据收集的config。在experiment下面的avwan的config里面建立collectconfig文件夹。在里面配置每一个收集数据的config。在外层目录的collect.yaml中配置收集数据的名称然后引用相应的config。最终可以通过run_avwan.py直接进行运行。
4. 收集的数据首先默认保存在/data/Getuanhui/下面
5. 将数据收集相关的实验到result里面。

### avnav的数据收集
---
1. avnav的数据收集和avwan数据收集的目的一样。
2. avnav中model最做不好的问题，然后针对这个问题进行数据的重收集。
3. 在收集数据的代码方面，和avwan规则的3,4，5是一致的。

### 任务优先级
1. 首先完成avwan的数据收集。尽量覆盖所有的model做不好的问题。数据体量由你自己决定。
2. 完成avwan数据收集之后，进行model的重训练。训练方式阅读experiment/avwan/train.md
3. 训练和测试相关内容保存根目录result下面。
4. 完成上述实验之后，开始进行avnav的数据收集任务。