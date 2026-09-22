"""Local AV-WaN configuration; external SoundSpaces assets are read-only."""
from pathlib import Path

from habitat.config import Config
from ss_baselines.av_wan.config.default import get_task_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_config(soundspaces_root="/home/getuanhui/project/sound-spaces", *,
               task_config_path=None, split="train_telephone", scenes=None):
    root = Path(soundspaces_root).expanduser().resolve()
    task_file = Path(task_config_path) if task_config_path else root / "configs/audionav/av_wan/mp3d/audiogoal.yaml"
    task = get_task_config(str(task_file))
    task.defrost()
    task.DATASET.SPLIT = split
    # The local SoundSpaces fork appends SIMULATOR.SENSORS to AGENT_0.SENSORS.
    # Keep that extension empty to avoid duplicates on repeated reconfigure().
    if not hasattr(task.SIMULATOR, "SENSORS"):
        task.SIMULATOR.SENSORS = []
    task.SIMULATOR.AGENT_0.SENSORS = list(dict.fromkeys(
        list(task.SIMULATOR.AGENT_0.SENSORS) + ["RGB_SENSOR", "DEPTH_SENSOR"]))
    if scenes is not None:
        task.DATASET.CONTENT_SCENES = list(scenes)
    # Resolve assets against SoundSpaces without chdir or writing into that repo.
    for node, keys in ((task.DATASET, ("SCENES_DIR", "DATA_PATH")),
                       (task.SIMULATOR, ("SCENE_OBSERVATION_DIR",)),
                       (task.SIMULATOR.AUDIO, ("BINAURAL_RIR_DIR", "SOURCE_SOUND_DIR", "METADATA_DIR"))):
        for key in keys:
            value = Path(getattr(node, key))
            if not value.is_absolute():
                setattr(node, key, str(root / value))
    task.freeze()
    config = Config()
    config.ENV_NAME = "AVWANRLEnv"
    config.SEED = 42
    config.CONTINUOUS = False
    config.MASKING = True
    config.PREDICTION_INTERVAL = 10
    config.MODEL_DIR = str(PROJECT_ROOT / "experiment/avwan/result")
    config.TASK_CONFIG = task
    config.RL = Config()
    config.RL.WITH_TIME_PENALTY = True
    config.RL.SLACK_REWARD = -0.01
    config.RL.WITH_DISTANCE_REWARD = True
    config.RL.DISTANCE_REWARD_SCALE = 1.0
    config.RL.SUCCESS_REWARD = 10.0
    config.OFFLINE_RL = Config()
    config.OFFLINE_RL.GAMMA = 0.99
    config.freeze()
    return config
