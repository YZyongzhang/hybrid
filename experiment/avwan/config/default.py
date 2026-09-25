#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union
import os
import logging
import shutil
from pathlib import Path
import yaml

import numpy as np

from habitat import get_config as get_task_config
from habitat.config import Config as CN
from habitat.config.default import SIMULATOR_SENSOR
import habitat

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","
# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 0
_C.BASE_TASK_CONFIG_PATH = "experiment/avwan/config/audiogoal.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.TRAINER_NAME = "AVNavTrainer"
_C.ENV_NAME = "AudioNavRLEnv"
_C.SIMULATOR_GPU_ID = 0
_C.TORCH_GPU_ID = 0
_C.VIDEO_OPTION = ["disk", "tensorboard"]
_C.VISUALIZATION_OPTION = ["top_down_map"]
_C.TENSORBOARD_DIR = "tb"
_C.VIDEO_DIR = "video_dir"
_C.TEST_EPISODE_COUNT = 2
_C.EVAL_CKPT_PATH_DIR = "data/checkpoints"  # path to ckpt or path to ckpts dir
_C.NUM_PROCESSES = 16
_C.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
_C.CHECKPOINT_FOLDER = "data/checkpoints"
_C.NUM_UPDATES = 10000
_C.LOG_INTERVAL = 10
_C.LOG_FILE = "train.log"
_C.CHECKPOINT_INTERVAL = 50
_C.USE_VECENV = True
_C.USE_SYNC_VECENV = False
_C.EXTRA_RGB = False
_C.DEBUG = False
_C.USE_LAST_CKPT = False
_C.DISPLAY_RESOLUTION = 128
_C.CONTINUOUS = False
_C.FOLLOW_SHORTEST_PATH = False
# Maximum primitive steps per AV-WaN waypoint action.
_C.PREDICTION_INTERVAL = 10
_C.OFFLINE_RL = CN()
_C.OFFLINE_RL.GAMMA = 0.99
# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
# The split to evaluate on
_C.EVAL.SPLIT = "val"
_C.EVAL.USE_CKPT_CONFIG = True
# -----------------------------------------------------------------------------
# REINFORCEMENT LEARNING (RL) ENVIRONMENT CONFIG
# -----------------------------------------------------------------------------
_C.RL = CN()
_C.RL.SUCCESS_REWARD = 10.0
_C.RL.SLACK_REWARD = -0.01
_C.RL.WITH_TIME_PENALTY = True
_C.RL.WITH_DISTANCE_REWARD = True
_C.RL.DISTANCE_REWARD_SCALE = 1.0
_C.RL.TIME_DIFF = False
# -----------------------------------------------------------------------------
# PROXIMAL POLICY OPTIMIZATION (PPO)
# -----------------------------------------------------------------------------
_C.RL.PPO = CN()
_C.RL.PPO.clip_param = 0.2
_C.RL.PPO.ppo_epoch = 4
_C.RL.PPO.num_mini_batch = 16
_C.RL.PPO.value_loss_coef = 0.5
_C.RL.PPO.entropy_coef = 0.01
_C.RL.PPO.lr = 7e-4
_C.RL.PPO.eps = 1e-5
_C.RL.PPO.max_grad_norm = 0.5
_C.RL.PPO.num_steps = 5
_C.RL.PPO.hidden_size = 512
_C.RL.PPO.use_gae = True
_C.RL.PPO.use_linear_lr_decay = False
_C.RL.PPO.use_linear_clip_decay = False
_C.RL.PPO.gamma = 0.99
_C.RL.PPO.tau = 0.95
_C.RL.PPO.reward_window_size = 50
# -----------------------------------------------------------------------------
# TASK CONFIG
# -----------------------------------------------------------------------------
_TC = habitat.get_config()
_TC.defrost()
_TC.COLLECT = CN()
_TC.COLLECT.OPEN = True
_TC.COLLECT.TYPE = "greedy"
_TC.COLLECT.TO_PATH = ""
_TC.COLLECT.OUTPUT_DIR = "data/avwan/expert"
_TC.COLLECT.NUM_EPISODES = -1  # Legacy YAML compatibility; collection uses env.episodes.
_TC.COLLECT.EPISODE_ID = None
_TC.COLLECT.SELECTION_PATH = ""
_TC.COLLECT.CONTENT_DIR = ""
_TC.COLLECT.SUCCESS_ONLY = False
_TC.COLLECT.VISUALIZE = True
_TC.COLLECT.RESUME = False
_TC.COLLECT.TIMEOUT_SECONDS = 300
# Register the training schema with YACS; audiogoal.yaml owns its defaults.
_TC.TRAIN = CN(yaml.safe_load(Path(__file__).with_name('audiogoal.yaml').read_text())['TRAIN'])
# -----------------------------------------------------------------------------
# AUDIOGOAL_SENSOR
# -----------------------------------------------------------------------------
_TC.TASK.AUDIOGOAL_SENSOR = CN()
_TC.TASK.AUDIOGOAL_SENSOR.TYPE = "AudioGoalSensor"
# -----------------------------------------------------------------------------
_TC.TASK.EMANTICOBJECT_SENSOR = CN()
_TC.TASK.EMANTICOBJECT_SENSOR.TYPE = "SemanticObjectSensor"
# SPECTROGRAM_SENSOR
# -----------------------------------------------------------------------------
_TC.TASK.SPECTROGRAM_SENSOR = CN()
_TC.TASK.SPECTROGRAM_SENSOR.TYPE = "SpectrogramSensor"
# -----------------------------------------------------------------------------
# ANGLE_SENSOR
_TC.TASK.AngleSensor = CN()
_TC.TASK.AngleSensor.TYPE = "AngleSensor"

#Collision_SENSOR
_TC.TASK.Collision = CN()
_TC.TASK.Collision.TYPE = "Collision"
#POSE_SENSOR
_TC.TASK.PoseSensor = CN()
_TC.TASK.PoseSensor.TYPE = "PoseSensor"
#egomap
_TC.TASK.EgoMap = SIMULATOR_SENSOR.clone()
_TC.TASK.EgoMap.TYPE = "EgoMap"
_TC.TASK.EgoMap.MAP_SIZE = 31
_TC.TASK.EgoMap.MAP_RESOLUTION = 0.1
_TC.TASK.EgoMap.HEIGHT_THRESH = (0.5, 2.0)
# soundspaces
# -----------------------------------------------------------------------------
_TC.SIMULATOR.GRID_SIZE = 0.5
_TC.SIMULATOR.CONTINUOUS_VIEW_CHANGE = False
_TC.SIMULATOR.VIEW_CHANGE_FPS = 10
_TC.SIMULATOR.SCENE_DATASET = 'replica'
_TC.SIMULATOR.USE_RENDERED_OBSERVATIONS = False
_TC.SIMULATOR.SCENE_OBSERVATION_DIR = 'data/scene_observations'
_TC.SIMULATOR.STEP_TIME = 1.0
_TC.SIMULATOR.AUDIO = CN()
_TC.SIMULATOR.AUDIO.SCENE = ""
_TC.SIMULATOR.AUDIO.BINAURAL_RIR_DIR = "data/binaural_rirs"
_TC.SIMULATOR.AUDIO.RIR_SAMPLING_RATE = 44100
_TC.SIMULATOR.AUDIO.SOURCE_SOUND_DIR = "data/sounds/1s_all"
_TC.SIMULATOR.AUDIO.METADATA_DIR = "data/metadata"
_TC.SIMULATOR.AUDIO.POINTS_FILE = 'points.txt'
_TC.SIMULATOR.AUDIO.GRAPH_FILE = 'graph.pkl'
_TC.SIMULATOR.AUDIO.HAS_DISTRACTOR_SOUND = False
_TC.SIMULATOR.AUDIO.EVERLASTING = True
_TC.SIMULATOR.AUDIO.CROSSFADE = False
# -----------------------------------------------------------------------------
# DistanceToGoal Measure
# -----------------------------------------------------------------------------
_TC.TASK.DISTANCE_TO_GOAL = CN()
_TC.TASK.DISTANCE_TO_GOAL.TYPE = "DistanceToGoal"
_TC.TASK.DISTANCE_TO_GOAL.DISTANCE_TO = "POINT"
_TC.TASK.NORMALIZED_DISTANCE_TO_GOAL = CN()
_TC.TASK.NORMALIZED_DISTANCE_TO_GOAL.TYPE = "NormalizedDistanceToGoal"
_TC.TASK.SUCCESS_WHEN_SILENT = CN()
_TC.TASK.SUCCESS_WHEN_SILENT.TYPE = "SWS"
# -----------------------------------------------------------------------------
# Dataset extension
# -----------------------------------------------------------------------------
_TC.DATASET.VERSION = 'v1'
_TC.DATASET.CONTINUOUS = False
# -----------------------------------------------------------------------------
# NumberOfAction Measure
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
_TC.TASK.NUM_ACTION = CN()
_TC.TASK.NUM_ACTION.TYPE = "NA"
_TC.TASK.SUCCESS_WEIGHTED_BY_NUM_ACTION = CN()
_TC.TASK.SUCCESS_WEIGHTED_BY_NUM_ACTION.TYPE = "SNA"
_TC.TASK.ORACLE_ACTION_SENSOR = CN()
_TC.TASK.ORACLE_ACTION_SENSOR.TYPE = "OracleActionSensor"


def merge_from_path(config, config_paths):
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)
    return config


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
    model_dir: Optional[str] = None,
    run_type: Optional[str] = None,
    overwrite: bool = False
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
        model_dir: suffix for output dirs
        run_type: either train or eval
    """
    config = _C.clone()
    config.TASK_CONFIG = _TC.clone()
    config = merge_from_path(config, config_paths)
    config.TASK_CONFIG = get_task_config(config_paths=config.BASE_TASK_CONFIG_PATH)
    # Apply experiment TASK_CONFIG overrides after loading the base task YAML.
    config.TASK_CONFIG.defrost()
    config = merge_from_path(config, config_paths)

    # config_name = os.path.basename(config_paths).split('.')[0]
    if model_dir is None:
        model_dir = 'data/models/output'
    config.TENSORBOARD_DIR = os.path.join(model_dir, 'tb')
    config.CHECKPOINT_FOLDER = os.path.join(model_dir, 'data')
    config.VIDEO_DIR = os.path.join(model_dir, 'video_dir')
    config.LOG_FILE = os.path.join(model_dir, 'train.log')
    config.EVAL_CKPT_PATH_DIR = os.path.join(model_dir, 'data')

    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    dirs = [config.VIDEO_DIR, config.TENSORBOARD_DIR, config.CHECKPOINT_FOLDER]

    config.TASK_CONFIG.defrost()
    config.TASK_CONFIG.SIMULATOR.USE_SYNC_VECENV = config.USE_SYNC_VECENV
    if config.CONTINUOUS:
        config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE = 0.25
        config.TASK_CONFIG.SIMULATOR.TYPE = "ContinuousSoundSpacesSim"
        config.TASK_CONFIG.SIMULATOR.USE_RENDERED_OBSERVATIONS = False
        config.TASK_CONFIG.SIMULATOR.STEP_TIME = 0.25
        config.TASK_CONFIG.SIMULATOR.AUDIO.CROSSFADE = True
        config.TASK_CONFIG.DATASET.CONTINUOUS = True
        config.RL.DISTANCE_REWARD_SCALE = 1.0

        # config.TASK_CONFIG.SIMULATOR.STEP_TIME = 1.0
        # config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE = 1.0
        # config.TASK_CONFIG.SIMULATOR.TURN_ANGLE = 90
    else:
        config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE = config.TASK_CONFIG.SIMULATOR.GRID_SIZE
    config.TASK_CONFIG.freeze()
    config.freeze()
    return config


def get_task_config(
        config_paths: Optional[Union[List[str], str]] = None,
        opts: Optional[list] = None
) -> habitat.Config:
    config = _TC.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config
