"""Load the installed benchmark explicitly, without deleted project config shims."""
import importlib.util
from pathlib import Path


BENCHMARK_ROOT = Path('/home/getuanhui/project/finnal_exp')
DATA_ROOT = Path('/hdd-data3/yongzhang/finnal_exp/data')


def get_benchmark_config(config_path, data_root, split, manifest, max_steps, seed):
    import soundspaces  # Register AudioNav dataset, simulator, task, and sensors.
    spec = importlib.util.spec_from_file_location('avnav_fixed_benchmark_config', BENCHMARK_ROOT / 'configs/default.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._C.BASE_TASK_CONFIG_PATH = str(Path(config_path).resolve())
    config = module.get_config()
    config.defrost()
    data_root = Path(data_root).resolve()
    config.SEED = seed
    config.TASK_CONFIG.SEED = seed
    config.TASK_CONFIG.DATASET.SPLIT = split
    config.TASK_CONFIG.DATASET.CONTENT_SCENES = ['*']
    config.TASK_CONFIG.DATASET.DATA_PATH = str(Path(manifest).resolve())
    config.TASK_CONFIG.DATASET.SCENES_DIR = str(data_root / 'scene_datasets/mp3d')
    config.TASK_CONFIG.SIMULATOR.AUDIO.BINAURAL_RIR_DIR = str(data_root / 'binaural_rirs')
    config.TASK_CONFIG.SIMULATOR.AUDIO.SOURCE_SOUND_DIR = str(data_root / 'sounds/1s_all')
    config.TASK_CONFIG.SIMULATOR.AUDIO.METADATA_DIR = str(data_root / 'metadata')
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR', 'DEPTH_SENSOR']
    # This installed simulator appends SENSORS on every agent-config lookup.
    # Declare sensors once on the agent to avoid duplicate UUIDs and list growth.
    config.TASK_CONFIG.SIMULATOR.SENSORS = []
    config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = max_steps
    config.freeze()
    return config
