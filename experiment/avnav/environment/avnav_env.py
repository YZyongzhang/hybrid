import sys

# from ss_baselines.common.env_utils import construct_envs
from ss_baselines.common.environments import AudioNavRLEnv
# from configs.default import get_config
# from habitat.datasets import make_dataset
from soundspaces.datasets.audionav_dataset import AudioNavDataset

# config = get_config()

"""  
从soundspaces中获取到env并暴漏出来
"""

# dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
# ENV = AudioNavRLEnv(config=config)