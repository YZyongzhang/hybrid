import torch
import os
import random
import re
from pathlib import Path
import torch.nn.functional as F

from configs.default import get_config
from env import Env
from utils.log import append_experiment_journal, logger, setup_run_logger


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_onlinerl_setup(online_cfg):
    model_name = str(getattr(online_cfg, "model", "v1")).lower().strip()
    foundation_ckpt = str(getattr(online_cfg, "FOUNDATION_CKPT", "")).strip()
    freeze_backbone = bool(getattr(online_cfg, "freeze_backbone", False))
    experiment_mode = str(getattr(online_cfg, "experiment", "custom")).lower().strip()

    if experiment_mode in {"v1_freeze", "freeze_foundation", "freezefoundationmodel"}:
        model_name = "v1"
        freeze_backbone = True
        if not foundation_ckpt:
            raise ValueError(
                "TRAIN.ONLINE.FOUNDATION_CKPT is required when TRAIN.ONLINE.experiment=v1_freeze."
            )
    elif experiment_mode in {"v2_scratch", "scratch", "from_scratch"}:
        model_name = "v2"
        foundation_ckpt = ""
        freeze_backbone = False
    elif experiment_mode in {"custom", ""}:
        pass
    else:
        raise ValueError(
            f"Unsupported TRAIN.ONLINE.experiment: {experiment_mode}. "
            "Use one of [custom, v1_freeze, v2_scratch]."
        )

    return model_name, foundation_ckpt, freeze_backbone, experiment_mode


def _load_hybrid_model(ckpt_path):
    from network import HybridNetwork
    # from network.av_wan_network import Network
    from network import HybridNetworkUnheard

    device = _get_device()
    logger.info("loading hybrid model ckpt: %s", ckpt_path)
    # model = HybridNetwork().to(device)
    model = HybridNetworkUnheard().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    logger.info("hybrid model loaded on device=%s", device)
    return model


def _load_offline_v1_5_model(config, ckpt_path):
    from network import SAC_LSTM_CQL_v1_5

    offline_cfg = config.TASK_CONFIG.TRAIN.OFFLINE
    device = _get_device()
    logger.info("loading offline v1_5 model ckpt: %s", ckpt_path)
    model = SAC_LSTM_CQL_v1_5(
        state_dim=int(offline_cfg.state_dim),
        hidden_dim=int(offline_cfg.hidden_dim),
        action_dim=int(offline_cfg.action_dim),
        actor_lr=float(offline_cfg.actor_lr),
        critic_lr=float(offline_cfg.critic_lr),
        alpha_lr=float(getattr(offline_cfg, "alpha_lr", 0.0001)),
        target_entropy=float(offline_cfg.target_entropy),
        tau=float(offline_cfg.tau),
        gamma=float(offline_cfg.gamma),
        beta=float(offline_cfg.beta),
        device=device,
        lstm_hidden_dim=int(getattr(offline_cfg, "lstm_hidden_dim", offline_cfg.state_dim)),
        lstm_num_layers=int(getattr(offline_cfg, "lstm_num_layers", 1)),
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    model.eval()
    logger.info("offline v1_5 model loaded on device=%s", device)
    return model


def _run_collect(config, collect_config):
    from col import COLLECTER
    from run import Collect

    logger.info("task=COLLECT type=%s", collect_config.TYPE)
    logger.info("collect config: %s", collect_config)

    collect_type = str(getattr(collect_config, "TYPE", "")).strip()

    # AVWAN greedy collection runs its own WhcMapNavEnv internally
    if collect_type == "avwan_greedy":
        from tools.collect_avwan_greedy_data import collect_avwan_greedy_main

        logger.info("AVWAN greedy data collection start")
        collect_avwan_greedy_main(config=config, collect_config=collect_config)
        logger.info("AVWAN greedy data collection finished")
        return

    env = Env(config=config)
    if collect_type == "offlineRL_v1_5":
        hybrid_model = _load_hybrid_model(collect_config.COLLECT_CKPT)
        offline_ckpt = str(getattr(collect_config, "OFFLINE_CKPT", "")).strip()
        if not offline_ckpt:
            raise ValueError("COLLECT.OFFLINE_CKPT is required when COLLECT.TYPE=offlineRL_v1_5")
        offline_model = _load_offline_v1_5_model(config, offline_ckpt)
        collecter = COLLECTER(
            config,
            env,
            hybrid_model=hybrid_model,
            offline_model=offline_model,
        )
    else:
        collecter = COLLECTER(config, env , collect_ckpt = collect_config.COLLECT_CKPT )
    logger.info("collect begin")
    Collect(collecter=collecter)
    logger.info("collect finished")


def _run_pt(pt_config):
    from train.VADE import LoadLmdb

    logger.info("task=PT type=%s", pt_config.TYPE)
    logger.info("pt config: %s", pt_config)
    if pt_config.TYPE == "HybridNetwork":
        LoadLmdb.load_pt(pt_config.RAW_DATA_PATH, pt_config)
        return
    if pt_config.TYPE == "WaypointPolarPT":
        LoadLmdb.load_pt_waypoint_polar(pt_config.RAW_DATA_PATH, pt_config)
        return
    if pt_config.TYPE == "OfflineWithHybridPT":
        LoadLmdb.load_offline_with_hybrid(pt_config.RAW_DATA_PATH, config=pt_config)
        return
    if pt_config.TYPE == "HybridNetworkTwoFrame":
        LoadLmdb.load_two_frame_pt(pt_config.RAW_DATA_PATH, pt_config)
        return

    # Types below require foundation ckpt.
    ckpt_required_types = {
        "OfflineTwoFrameWithHybridPT": "load_offline_two_frame",
        "OfflineRLFeatures": "load_offline_rl_features",
        "offline": "load_offline",
        "offlinetwoframe": "load_offline_two_frame",
        "av_wan_offline_rl_features": "load_offline_rl_features",
        "offlineoneframe": "load_offline_one_frame",
        "offlinelstm": "load_offline_lstm",
        "offlinelstm_v15": "load_offline_lstm_v15",
        "offlinetransformer_v16": "load_offline_lstm_v15",
        "offlinetransformer": "load_offline_lstm_v15",
        "offlinesequence":"load_offline_lstm_v15",
        "avwanoffline": "load_av_wan_offline",
        "offlinewaypointreward":"load_offline_way_point_reward",
        "offlinelstm_by_level": "load_offline_lstm_level",
        "offlinelstm_by_level_audio_visual": "load_offline_lstm_level_audio_visual",
        "load_way_point_offline_lstm":"load_way_point_offline_lstm",
        "load_hybrid_action_sequence":"load_hybrid_action_sequence",
        "load_pt_hybrid":"load_pt_hybrid"
    }
    if pt_config.TYPE not in ckpt_required_types:
        raise ValueError(f"Unsupported PT.TYPE: {pt_config.TYPE}")

    method_name = ckpt_required_types[pt_config.TYPE]

    # AVWAN offline PT uses AVWANNetwork backbone
    if pt_config.TYPE == "avwanoffline":
        from network import AVWANNetwork
        device = _get_device()
        model = AVWANNetwork().to(device)
        ckpt_path = str(getattr(pt_config, "CKPT", "")).strip()
        if ckpt_path and os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            logger.info("AVWANNetwork loaded from: %s", ckpt_path)
        else:
            logger.warning("No AVWANNetwork checkpoint; using randomly initialized model")
        model.eval()
    else:
        model = _load_hybrid_model(pt_config.CKPT)
    method = getattr(LoadLmdb, method_name)
    logger.info("pt conversion method=%s raw_data=%s to_path=%s", method_name, pt_config.RAW_DATA_PATH, pt_config.TO_PATH)
    method(pt_config.RAW_DATA_PATH, model=model, config=pt_config)
    logger.info("pt conversion finished")


def _run_eval(config, eval_config):
    sweep_cfg = getattr(eval_config, "CKPT_SWEEP", None)
    sweep_enable = bool(getattr(sweep_cfg, "ENABLE", getattr(eval_config, "CKPT_SWEEP_ENABLE", False)))
    if sweep_enable:
        _run_eval_ckpt_sweep(config, eval_config)
        logger.info("eval finished")
        return

    logger.info("task=EVAL type=%s", eval_config.TYPE)
    logger.info("eval config: %s", eval_config)
    _run_eval_once(config, eval_config)
    logger.info("eval finished")


def _run_eval_once(config, eval_config, ckpt_override: str = "", ckpt_field: str = ""):
    if eval_config.TYPE == "OfflineRL_v1_3":
        from tools.eval_hybrid_sac import run_v1_3_eval

        sac_ckpt = ckpt_override if ckpt_field == "SAC_CKPT" else eval_config.SAC_CKPT
        return run_v1_3_eval(
            config=config,
            hybrid_ckpt=eval_config.HYBRID_CKPT,
            sac_ckpt=sac_ckpt,
            episodes=eval_config.EPISODES,
            max_steps=eval_config.MAX_STEPS,
            seed=eval_config.SEED,
            stochastic_sac=eval_config.STOCHASTIC_SAC,
        )
    if eval_config.TYPE == "OfflineRL_v1_5":
        from tools.eval_hybrid_sac import run_v1_5_eval

        sac_ckpt = ckpt_override if ckpt_field == "SAC_CKPT" else eval_config.SAC_CKPT
        return run_v1_5_eval(
            config=config,
            hybrid_ckpt=eval_config.HYBRID_CKPT,
            sac_ckpt=sac_ckpt,
            episodes=eval_config.EPISODES,
            max_steps=eval_config.MAX_STEPS,
            seed=eval_config.SEED,
            stochastic_sac=eval_config.STOCHASTIC_SAC,
            temporal_ckpt=getattr(eval_config, "TEMPORAL_CKPT", ""),
            actor_ckpt=getattr(eval_config, "ACTOR_CKPT", ""),
            critic1_ckpt=getattr(eval_config, "CRITIC1_CKPT", ""),
            critic2_ckpt=getattr(eval_config, "CRITIC2_CKPT", ""),
        )
    if eval_config.TYPE == "OnlineRL":
        model_path_override = ckpt_override if ckpt_field == "ONLINE_MODEL_PATH" else ""
        return _run_onlinerl_eval(config, eval_config, model_path_override=model_path_override)
    if eval_config.TYPE == "OfflineRL_av_wan":
        from train.online_test import OnlineTest
        from network import SAC_AVWAN_CQL, AVWANNetwork

        device = _get_device()

        # Load AVWAN backbone
        av_wan_backbone = AVWANNetwork().to(device)
        av_wan_ckpt = str(getattr(eval_config, "AV_WAN_CKPT", "")).strip()
        if av_wan_ckpt and os.path.exists(av_wan_ckpt):
            av_wan_backbone.load_state_dict(torch.load(av_wan_ckpt, map_location=device))
        av_wan_backbone.eval()
        logger.info("AVWAN backbone loaded from %s", av_wan_ckpt or "(random init)")

        # Load SAC model
        offline_cfg = config.TASK_CONFIG.TRAIN.OFFLINE
        map_size = int(getattr(offline_cfg, "map_size", 9))
        sac_model = SAC_AVWAN_CQL(
            state_dim=int(offline_cfg.state_dim),
            hidden_dim=int(offline_cfg.hidden_dim),
            action_dim=map_size * map_size,
            actor_lr=float(offline_cfg.actor_lr),
            critic_lr=float(offline_cfg.critic_lr),
            alpha_lr=float(getattr(offline_cfg, "alpha_lr", 0.0001)),
            target_entropy=float(offline_cfg.target_entropy),
            tau=float(offline_cfg.tau),
            gamma=float(offline_cfg.gamma),
            beta=float(offline_cfg.beta),
            device=device,
            lstm_hidden_dim=int(getattr(offline_cfg, "lstm_hidden_dim", offline_cfg.state_dim)),
            lstm_num_layers=int(getattr(offline_cfg, "lstm_num_layers", 1)),
            map_size=map_size,
        ).to(device)
        sac_ckpt = ckpt_override if ckpt_field == "SAC_CKPT" else eval_config.SAC_CKPT
        if sac_ckpt and os.path.exists(sac_ckpt):
            sac_model.load_state_dict(torch.load(sac_ckpt, map_location=device))
            logger.info("SAC model loaded from %s", sac_ckpt)
        sac_model.eval()

        env = Env(config)
        online_test = OnlineTest(env=env, hybridmodel=av_wan_backbone, config=offline_cfg)
        total_reward, spl, sr = online_test.rollout_avwan(0, sac_model, logger)
        logger.info("AVWAN eval result: reward=%.4f spl=%.4f sr=%.4f", total_reward, spl, sr)
        return {"avg_reward": float(total_reward), "avg_spl": float(spl), "avg_sr": float(sr)}
    raise ValueError(f"Unsupported EVAL.TYPE: {eval_config.TYPE}")


def _resolve_ckpt_sweep_field(eval_config, sweep_cfg) -> str:
    configured = str(getattr(sweep_cfg, "FIELD", "")).strip()
    if configured:
        return configured
    eval_type = str(getattr(eval_config, "TYPE", "")).strip()
    if eval_type in {"OfflineRL_v1_3", "OfflineRL_v1_5", "OfflineRL_av_wan"}:
        return "SAC_CKPT"
    if eval_type == "OnlineRL":
        return "ONLINE_MODEL_PATH"
    raise ValueError(f"EVAL.CKPT_SWEEP.FIELD is required for EVAL.TYPE={eval_type}")


def _extract_eval_spl(eval_result):
    if eval_result is None:
        return None
    if hasattr(eval_result, "avg_spl"):
        return float(eval_result.avg_spl)
    if isinstance(eval_result, dict):
        if "avg_spl" in eval_result:
            return float(eval_result["avg_spl"])
        if "spl" in eval_result:
            return float(eval_result["spl"])
    if isinstance(eval_result, (tuple, list)) and len(eval_result) >= 2:
        return float(eval_result[1])
    return None


def _ckpt_sort_key(path_obj: Path, sort_by_epoch: bool):
    if not sort_by_epoch:
        return path_obj.name
    found = re.findall(r"(\d+)", path_obj.stem)
    epoch = int(found[-1]) if found else -1
    return (epoch, path_obj.name)


def _run_eval_ckpt_sweep(config, eval_config):
    logger.info("task=EVAL type=%s (ckpt sweep enabled)", eval_config.TYPE)
    logger.info("eval config: %s", eval_config)
    sweep_cfg = getattr(eval_config, "CKPT_SWEEP", None)
    if sweep_cfg is None:
        raise ValueError("EVAL.CKPT_SWEEP is required when ckpt sweep is enabled.")

    ckpt_dir_str = str(getattr(sweep_cfg, "DIR", "")).strip()
    if not ckpt_dir_str:
        raise ValueError("EVAL.CKPT_SWEEP.DIR is required when EVAL.CKPT_SWEEP.ENABLE=True")
    ckpt_dir = Path(ckpt_dir_str).expanduser()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"EVAL.CKPT_SWEEP.DIR not found: {ckpt_dir}")
    if not ckpt_dir.is_dir():
        raise NotADirectoryError(f"EVAL.CKPT_SWEEP.DIR is not a directory: {ckpt_dir}")

    pattern = str(getattr(sweep_cfg, "PATTERN", "*.pth")).strip() or "*.pth"
    recursive = bool(getattr(sweep_cfg, "RECURSIVE", False))
    sort_by_epoch = bool(getattr(sweep_cfg, "SORT_BY_EPOCH", True))
    max_ckpts = int(getattr(sweep_cfg, "MAX_CKPTS", 0))
    stop_on_error = bool(getattr(sweep_cfg, "STOP_ON_ERROR", False))
    ckpt_field = _resolve_ckpt_sweep_field(eval_config, sweep_cfg)

    ckpt_iter = ckpt_dir.rglob(pattern) if recursive else ckpt_dir.glob(pattern)
    ckpt_paths = [p for p in ckpt_iter if p.is_file()]
    if not ckpt_paths:
        raise FileNotFoundError(
            f"No checkpoint matched in {ckpt_dir} with pattern={pattern} recursive={recursive}"
        )
    ckpt_paths = sorted(ckpt_paths, key=lambda p: _ckpt_sort_key(p, sort_by_epoch))
    if max_ckpts > 0:
        ckpt_paths = ckpt_paths[:max_ckpts]

    logger.info(
        "[ckpt-sweep] begin: dir=%s pattern=%s recursive=%s field=%s total=%s",
        str(ckpt_dir),
        pattern,
        recursive,
        ckpt_field,
        len(ckpt_paths),
    )

    records = []
    for idx, ckpt_path in enumerate(ckpt_paths, start=1):
        ckpt_str = str(ckpt_path)
        logger.info("[ckpt-sweep] evaluating (%d/%d): %s", idx, len(ckpt_paths), ckpt_str)
        try:
            # Use a fresh config per checkpoint to avoid in-place mutations
            # inside downstream simulator/env code from leaking across sweep rounds.
            iter_config = config.clone()
            iter_eval_config = iter_config.TASK_CONFIG.EVAL
            result = _run_eval_once(
                iter_config,
                iter_eval_config,
                ckpt_override=ckpt_str,
                ckpt_field=ckpt_field,
            )
            spl = _extract_eval_spl(result)
            if spl is None:
                raise RuntimeError(f"Cannot parse SPL from eval result for checkpoint: {ckpt_str}")
            records.append({"ckpt": ckpt_str, "spl": float(spl), "result": result})
            logger.info("[ckpt-sweep] result ckpt=%s spl=%.6f", ckpt_str, float(spl))
        except Exception as exc:
            logger.exception("[ckpt-sweep] failed ckpt=%s error=%s", ckpt_str, exc)
            if stop_on_error:
                raise

    if not records:
        raise RuntimeError("No valid checkpoint evaluation result in ckpt sweep.")

    records_sorted = sorted(records, key=lambda x: x["spl"], reverse=True)
    best = records_sorted[0]
    logger.info("[ckpt-sweep] summary begin")
    for rank, item in enumerate(records_sorted, start=1):
        logger.info(
            "[ckpt-sweep] rank=%03d spl=%.6f ckpt=%s",
            rank,
            float(item["spl"]),
            item["ckpt"],
        )
    logger.info(
        "[ckpt-sweep] BEST ckpt=%s spl=%.6f",
        best["ckpt"],
        float(best["spl"]),
    )
    logger.info("[ckpt-sweep] summary end")


def _clone_config_with_dataset_split(config, split_name: str):
    split = str(split_name).strip()
    if not split:
        return config
    cfg = config.clone()
    cfg.defrost()
    cfg.TASK_CONFIG.defrost()
    cfg.TASK_CONFIG.DATASET.SPLIT = split
    cfg.TASK_CONFIG.freeze()
    cfg.freeze()
    return cfg


def _obs_to_inputs(obs):
    rgb = torch.as_tensor(obs["rgb"], dtype=torch.float32) / 255.0
    depth = torch.as_tensor(obs["depth"], dtype=torch.float32)
    audio = obs["spectrogram"]
    if isinstance(audio, (tuple, list)):
        audio = audio[0]
    audio = torch.as_tensor(audio, dtype=torch.float32)
    return audio, rgb, depth


def _fit_state_dim(state: torch.Tensor, state_dim: int) -> torch.Tensor:
    t = state.detach().float().view(-1).cpu()
    if t.numel() > state_dim:
        return t[:state_dim]
    if t.numel() < state_dim:
        return torch.cat([t, torch.zeros(state_dim - t.numel(), dtype=torch.float32)], dim=0)
    return t


def _get_dorl_value(dorl_config, dorl_algo: str, key: str, default):
    algo_cfg = getattr(dorl_config, dorl_algo, None)
    if algo_cfg is not None and hasattr(algo_cfg, key):
        return getattr(algo_cfg, key)
    return getattr(dorl_config, key, default)


def _build_dorl_ppo_online_eval_fn(config, dorl_config, dorl_algo: str = "ppo"):
    eval_cfg = config.TASK_CONFIG.EVAL
    eval_enable = bool(_get_dorl_value(dorl_config, dorl_algo, "online_eval_enable", True))
    if not eval_enable:
        return None

    hybrid_ckpt = str(_get_dorl_value(dorl_config, dorl_algo, "online_eval_hybrid_ckpt", "")).strip()
    if not hybrid_ckpt:
        hybrid_ckpt = str(getattr(eval_cfg, "DORL_HYBRID_CKPT", "")).strip()
    if not hybrid_ckpt:
        hybrid_ckpt = str(getattr(eval_cfg, "HYBRID_CKPT", "")).strip()
    if not hybrid_ckpt:
        logger.warning("skip PPO online eval: no hybrid checkpoint configured.")
        return None
    if not os.path.exists(hybrid_ckpt):
        logger.warning("skip PPO online eval: hybrid checkpoint not found: %s", hybrid_ckpt)
        return None

    split_name = str(
        _get_dorl_value(dorl_config, dorl_algo, "online_eval_split", getattr(eval_cfg, "DORL_DATASET_SPLIT", ""))
    ).strip()
    episodes = int(_get_dorl_value(dorl_config, dorl_algo, "online_eval_episodes", 16))
    max_steps = int(_get_dorl_value(dorl_config, dorl_algo, "online_eval_max_steps", getattr(eval_cfg, "MAX_STEPS", 200)))
    greedy = bool(_get_dorl_value(dorl_config, dorl_algo, "online_eval_greedy", True))
    every = max(1, int(_get_dorl_value(dorl_config, dorl_algo, "online_eval_every", 1)))
    log_every = int(_get_dorl_value(dorl_config, dorl_algo, "online_eval_log_every", 0))

    eval_run_cfg = _clone_config_with_dataset_split(config, split_name)
    eval_env = Env(eval_run_cfg)
    hybrid_model = _load_hybrid_model(hybrid_ckpt)
    hybrid_device = next(hybrid_model.parameters()).device
    logger.info(
        "DORL PPO online eval enabled: split=%s episodes=%s max_steps=%s greedy=%s every=%s hybrid_ckpt=%s",
        eval_run_cfg.TASK_CONFIG.DATASET.SPLIT,
        episodes,
        max_steps,
        greedy,
        every,
        hybrid_ckpt,
    )

    def _online_eval_fn(actor, state_dim: int, actor_device: torch.device, epoch: int):
        if epoch % every != 0:
            return {}

        run_episodes = max(1, episodes)
        total_reward = 0.0
        total_spl = 0.0
        total_distance = 0.0

        for ep_idx in range(run_episodes):
            obs = eval_env.reset()
            episode_reward = 0.0
            pre_rgb = None
            pre_depth = None
            info = {"spl": 0.0, "distance_to_goal": -1.0}

            for _ in range(max_steps):
                with torch.no_grad():
                    audio, rgb, depth = _obs_to_inputs(obs)
                    if pre_rgb is None:
                        pre_rgb = torch.zeros_like(rgb)
                    if pre_depth is None:
                        pre_depth = torch.zeros_like(depth)

                    trgb = torch.cat([pre_rgb, rgb], dim=2)
                    tdepth = torch.cat([pre_depth, depth], dim=2)
                    state = hybrid_model.embedding_forward(
                        audio.to(hybrid_device),
                        trgb.to(hybrid_device),
                        tdepth.to(hybrid_device),
                    )
                    state = _fit_state_dim(state, state_dim).to(actor_device).unsqueeze(0)
                    logits = actor(state)
                    probs = F.softmax(logits, dim=-1)
                    if greedy:
                        action = int(torch.argmax(probs, dim=-1).item())
                    else:
                        action = int(torch.distributions.Categorical(probs).sample().item())

                obs, reward, done, info = eval_env.step(action=action)
                episode_reward += float(reward)
                pre_rgb = rgb
                pre_depth = depth
                if done:
                    break

            total_reward += episode_reward
            total_spl += float(info.get("spl", 0.0))
            total_distance += float(info.get("distance_to_goal", -1.0))
            if log_every > 0 and ((ep_idx + 1) % log_every == 0):
                logger.info(
                    "dorl ppo online eval epoch=%s episode=%s/%s reward=%.4f spl=%.4f distance=%.4f",
                    epoch,
                    ep_idx + 1,
                    run_episodes,
                    episode_reward,
                    float(info.get("spl", 0.0)),
                    float(info.get("distance_to_goal", -1.0)),
                )

        return {
            "reward": total_reward / float(run_episodes),
            "spl": total_spl / float(run_episodes),
            "distance": total_distance / float(run_episodes),
        }

    return _online_eval_fn


def _rotation_to_list(rotation):
    if rotation is None:
        return [0.0, 0.0, 0.0, 1.0]
    if all(hasattr(rotation, k) for k in ("x", "y", "z", "w")):
        return [float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)]
    if all(hasattr(rotation, k) for k in ("w", "x", "y", "z")):
        return [float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)]
    if isinstance(rotation, (tuple, list)):
        values = [float(v) for v in rotation]
        if len(values) >= 4:
            return values[:4]
    return [0.0, 0.0, 0.0, 1.0]


def _get_pose_from_env(env):
    sim = getattr(getattr(env, "_env", None), "_sim", None)
    if sim is None or not hasattr(sim, "get_agent_state"):
        return torch.zeros(7, dtype=torch.float32)
    try:
        state = sim.get_agent_state()
        pos = [float(v) for v in state.position]
        rot = _rotation_to_list(state.rotation)
        return torch.as_tensor(pos + rot, dtype=torch.float32)
    except Exception:
        return torch.zeros(7, dtype=torch.float32)


def _run_onlinerl_eval(config, eval_config, model_path_override: str = ""):
    from network import OnlineRLV1, OnlineRLV2

    device = _get_device()
    model_path = str(model_path_override).strip() if model_path_override else str(getattr(eval_config, "ONLINE_MODEL_PATH", "")).strip()
    if not model_path:
        raise ValueError("EVAL.ONLINE_MODEL_PATH is required when EVAL.TYPE=OnlineRL.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"EVAL.ONLINE_MODEL_PATH not found: {model_path}")

    seed = int(getattr(eval_config, "SEED", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env = Env(config)
    sim = getattr(getattr(env, "_env", None), "_sim", None)

    train_online_cfg = config.TASK_CONFIG.TRAIN.ONLINE
    model_name, foundation_ckpt, freeze_backbone, experiment_mode = _resolve_onlinerl_setup(
        train_online_cfg
    )
    model_name = str(getattr(eval_config, "ONLINE_MODEL", model_name)).lower().strip()
    action_dim = int(getattr(train_online_cfg, "action_dim", 4))
    hidden_dim = int(getattr(train_online_cfg, "hidden_dim", 256))
    use_pose_encoder = bool(getattr(train_online_cfg, "use_pose_encoder", False))
    pose_dim = int(getattr(train_online_cfg, "pose_dim", 7))
    pose_hidden_dim = int(getattr(train_online_cfg, "pose_hidden_dim", 64))

    if model_name == "v1":
        if not foundation_ckpt:
            raise ValueError("TRAIN.ONLINE.FOUNDATION_CKPT is required for OnlineRL v1 eval.")
        agent = OnlineRLV1(
            action_dim=action_dim,
            foundation_ckpt=foundation_ckpt,
            hidden_dim=hidden_dim,
            use_pose_encoder=use_pose_encoder,
            pose_dim=pose_dim,
            pose_hidden_dim=pose_hidden_dim,
            freeze_backbone=freeze_backbone,
            device=device,
        )
    elif model_name == "v2":
        agent = OnlineRLV2(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            foundation_ckpt=foundation_ckpt if foundation_ckpt else None,
            use_pose_encoder=use_pose_encoder,
            pose_dim=pose_dim,
            pose_hidden_dim=pose_hidden_dim,
            freeze_backbone=freeze_backbone,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported EVAL.ONLINE_MODEL: {model_name}")

    logger.info("loading onlinerl eval ckpt: %s", model_path)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.to(device)
    agent.eval()

    max_steps = int(getattr(eval_config, "MAX_STEPS", 200))
    greedy = bool(getattr(eval_config, "ONLINE_GREEDY", True))
    log_every = int(getattr(eval_config, "ONLINE_LOG_EVERY", 1))
    episodes = int(getattr(eval_config, "EPISODES", 0))
    if episodes <= 0:
        episodes = int(getattr(getattr(env, "_env", None), "number_of_episodes", 0))
    episodes = max(1, episodes)

    total_reward = 0.0
    total_spl = 0.0
    total_distance = 0.0
    logger.info(
        "onlinerl eval begin: model=%s experiment=%s split=%s episodes=%s max_steps=%s greedy=%s seed=%s",
        model_name,
        experiment_mode,
        config.TASK_CONFIG.DATASET.SPLIT,
        episodes,
        max_steps,
        greedy,
        seed,
    )

    for ep_idx in range(episodes):
        obs = env.reset()
        info = {"spl": 0.0, "distance_to_goal": -1.0}
        pre_rgb = None
        pre_depth = None
        episode_reward = 0.0
        done = False
        steps_taken = 0

        for _step in range(max_steps):
            audio, rgb, depth = _obs_to_inputs(obs)
            pose = _get_pose_from_env(env)
            if pre_rgb is None:
                pre_rgb = torch.zeros_like(rgb)
            if pre_depth is None:
                pre_depth = torch.zeros_like(depth)
            trgb = torch.cat([pre_rgb, rgb], dim=2)
            tdepth = torch.cat([pre_depth, depth], dim=2)

            with torch.no_grad():
                emb = agent.encode(
                    audio.unsqueeze(0).to(device),
                    trgb.unsqueeze(0).to(device),
                    tdepth.unsqueeze(0).to(device),
                    pose.unsqueeze(0).to(device),
                ).float()
                logits = agent.policy_head(emb)
                probs = F.softmax(logits, dim=-1)
                if greedy:
                    action = int(torch.argmax(probs, dim=-1).item())
                else:
                    action = int(torch.distributions.Categorical(probs).sample().item())

            obs, reward, done, info = env.step(action=action)
            is_collided = False
            if sim is not None and hasattr(sim, "previous_step_collided"):
                try:
                    is_collided = bool(sim.previous_step_collided)
                except Exception:
                    is_collided = False
            logger.info(
                "take action sac model %s ,reward %s , step %s , done %s , is collided %s",
                action,
                float(reward),
                _step,
                bool(done),
                is_collided,
            )
            episode_reward += float(reward)
            steps_taken = _step + 1
            pre_rgb = rgb
            pre_depth = depth
            if done:
                break

        total_reward += episode_reward
        total_spl += float(info.get("spl", 0.0))
        total_distance += float(info.get("distance_to_goal", -1.0))
        if log_every > 0 and ((ep_idx + 1) % log_every == 0):
            logger.info(
                "onlinerl eval episode=%s/%s reward=%.4f spl=%.4f distance=%.4f steps=%s done=%s",
                ep_idx + 1,
                episodes,
                episode_reward,
                float(info.get("spl", 0.0)),
                float(info.get("distance_to_goal", -1.0)),
                steps_taken,
                bool(done),
            )

    avg_reward = total_reward / float(episodes)
    avg_spl = total_spl / float(episodes)
    avg_distance = total_distance / float(episodes)
    logger.info(
        "onlinerl eval result: reward=%.4f spl=%.4f distance=%.4f",
        avg_reward,
        avg_spl,
        avg_distance,
    )
    return {
        "avg_reward": float(avg_reward),
        "avg_spl": float(avg_spl),
        "avg_distance": float(avg_distance),
        "episodes": int(episodes),
    }


def _build_offline_agent(config, offline_config):
    from train import OnlineTest

    env = Env(config)

    # AVWAN model uses AVWANNetwork as backbone instead of HybridNetworkUnheard
    if offline_config.model == "av_wan_v1":
        from network import AVWANNetwork

        av_wan_backbone = AVWANNetwork().to(_get_device())
        av_wan_backbone_ckpt = str(getattr(offline_config, "AV_WAN_BACKBONE_CKPT", "")).strip()
        if av_wan_backbone_ckpt and os.path.exists(av_wan_backbone_ckpt):
            av_wan_backbone.load_state_dict(torch.load(av_wan_backbone_ckpt, map_location=_get_device()))
            logger.info("AVWAN backbone loaded from: %s", av_wan_backbone_ckpt)
        else:
            logger.warning("AV_WAN_BACKBONE_CKPT not set or not found; using randomly initialized AVWANNetwork")
        av_wan_backbone.eval()
        hybrid_model = av_wan_backbone
    else:
        hybrid_model = _load_hybrid_model(offline_config.ONLINE_CKPT)

    online_test = OnlineTest(env=env, hybridmodel=hybrid_model, config=offline_config)

    state_dim = offline_config.state_dim
    action_dim = offline_config.action_dim
    hidden_dim = offline_config.hidden_dim
    actor_lr = offline_config.actor_lr
    critic_lr = offline_config.critic_lr
    alpha_lr = float(getattr(offline_config, "alpha_lr", 0.0001))
    tau = offline_config.tau
    gamma = offline_config.gamma
    beta = offline_config.beta
    target_entropy = offline_config.target_entropy
    log_alpha_min = float(getattr(offline_config, "log_alpha_min", -10.0))
    log_alpha_max = float(getattr(offline_config, "log_alpha_max", 2.0))
    device = _get_device()

    if offline_config.model == "v1":
        from network import SAC_model
        from train import OfflineTrain, OfflineTrainBuffer

        sac_model = SAC_model(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            alpha_lr=alpha_lr,
            target_entropy=target_entropy,
            tau=tau,
            gamma=gamma,
            beta=beta,
            log_alpha_min=log_alpha_min,
            log_alpha_max=log_alpha_max,
            device=device,
        )
        trainer = OfflineTrainBuffer if offline_config.buffer else OfflineTrain
    elif offline_config.model == "v1_3":
        from network import SAC_Hybrid_model
        from train import OfflineAndHybrid

        sac_model = SAC_Hybrid_model(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            alpha_lr=alpha_lr,
            target_entropy=target_entropy,
            tau=tau,
            gamma=gamma,
            beta=beta,
            device=device,
            hybrid_ckpt_path=offline_config.ONLINE_CKPT,
        )
        trainer = OfflineAndHybrid
    elif offline_config.model == "v1_4":
        from network import SAC_Hybrid_LSTM_CQL_model
        from train import OfflineAndHybrid

        lstm_hidden_dim = int(getattr(offline_config, "lstm_hidden_dim", state_dim))
        lstm_num_layers = int(getattr(offline_config, "lstm_num_layers", 1))
        sac_model = SAC_Hybrid_LSTM_CQL_model(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            alpha_lr=alpha_lr,
            target_entropy=target_entropy,
            tau=tau,
            gamma=gamma,
            beta=beta,
            device=device,
            hybrid_ckpt_path=offline_config.ONLINE_CKPT,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
        )
        trainer = OfflineAndHybrid
    elif offline_config.model == "v1_5":
        from network import SAC_LSTM_CQL_v1_5
        from train import OfflineTrain

        lstm_hidden_dim = int(getattr(offline_config, "lstm_hidden_dim", state_dim))
        lstm_num_layers = int(getattr(offline_config, "lstm_num_layers", 1))
        sac_model = SAC_LSTM_CQL_v1_5(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            alpha_lr=alpha_lr,
            target_entropy=target_entropy,
            tau=tau,
            gamma=gamma,
            beta=beta,
            device=device,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
        )
        trainer = OfflineTrain
    elif offline_config.model == "v1_6":
        from network import SAC_Transformer_CQL_v1_6
        from train import OfflineTrain

        transformer_dim = int(getattr(offline_config, "transformer_dim", state_dim))
        transformer_heads = int(getattr(offline_config, "transformer_heads", 4))
        transformer_layers = int(getattr(offline_config, "transformer_layers", 2))
        transformer_ffn_dim = int(
            getattr(offline_config, "transformer_ffn_dim", transformer_dim * 2)
        )
        max_seq_len = int(getattr(offline_config, "max_seq_len", 16))
        sac_model = SAC_Transformer_CQL_v1_6(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            alpha_lr=alpha_lr,
            target_entropy=target_entropy,
            tau=tau,
            gamma=gamma,
            beta=beta,
            device=device,
            transformer_dim=transformer_dim,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            transformer_ffn_dim=transformer_ffn_dim,
            max_seq_len=max_seq_len,
        )
        trainer = OfflineTrain
    elif offline_config.model == "v2":
        from network import CQLSAC
        from train import OfflineTrain, OfflineTrainBuffer

        sac_model = CQLSAC(
            state_size=state_dim,
            action_size=action_dim,
            hidden_size=hidden_dim,
            beta=beta,
            device=device,
        )
        trainer = OfflineTrainBuffer if offline_config.buffer else OfflineTrain
    elif offline_config.model == "v4":
        from network import cql_lstm
        from train import OfflineTrain

        sac_model = cql_lstm(
            state_size=state_dim,
            action_size=action_dim,
            tau=tau,
            hidden_size=hidden_dim,
            learning_rate=actor_lr,
            with_lagrange=False,
            target_action_gap=0,
            device=device,
            lstm_seq_len=5,
            lstm_layer=1,
            lstm_out=128,
        )
        trainer = OfflineTrain
    elif offline_config.model == "v5":
        from network import cql_lstm_attention
        from train import OfflineTrain

        sac_model = cql_lstm_attention(
            state_size=state_dim,
            action_size=action_dim,
            tau=tau,
            hidden_size=hidden_dim,
            learning_rate=actor_lr,
            with_lagrange=False,
            target_action_gap=0,
            device=device,
            lstm_seq_len=5,
            lstm_layer=1,
            lstm_out=128,
        )
        trainer = OfflineTrain
    elif offline_config.model == "av_wan_v1":
        from network import SAC_AVWAN_CQL
        from train import OfflineTrain

        lstm_hidden_dim = int(getattr(offline_config, "lstm_hidden_dim", state_dim))
        lstm_num_layers = int(getattr(offline_config, "lstm_num_layers", 1))
        map_size = int(getattr(offline_config, "map_size", 9))
        action_dim_avwan = map_size * map_size
        sac_model = SAC_AVWAN_CQL(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim_avwan,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            alpha_lr=alpha_lr,
            target_entropy=float(offline_config.target_entropy),
            tau=tau,
            gamma=gamma,
            beta=beta,
            device=device,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            map_size=map_size,
        )
        trainer = OfflineTrain
    else:
        raise ValueError(f"Unsupported OFFLINE.model: {offline_config.model}")

    if offline_config.LOAD_PATH:
        logger.info("loading offline agent ckpt: %s", offline_config.MODEL_PATH)
        sac_model.load_state_dict(torch.load(offline_config.MODEL_PATH, map_location=device))
        sac_model.train()

    logger.info("offline agent built: model=%s trainer=%s", offline_config.model, trainer.__name__)
    return sac_model, trainer, online_test


def _build_onlinerl_agent(config, online_config):
    from train import OnlineRLTrain

    device = _get_device()
    env = Env(config)
    model_name, foundation_ckpt, freeze_backbone, experiment_mode = _resolve_onlinerl_setup(
        online_config
    )

    if model_name == "v1":
        from network import OnlineRLV1

        if not foundation_ckpt:
            raise ValueError("TRAIN.ONLINE.FOUNDATION_CKPT is required for OnlineRL v1.")
        agent = OnlineRLV1(
            action_dim=int(online_config.action_dim),
            foundation_ckpt=foundation_ckpt,
            hidden_dim=int(getattr(online_config, "hidden_dim", 256)),
            use_pose_encoder=bool(getattr(online_config, "use_pose_encoder", False)),
            pose_dim=int(getattr(online_config, "pose_dim", 7)),
            pose_hidden_dim=int(getattr(online_config, "pose_hidden_dim", 64)),
            freeze_backbone=freeze_backbone,
            device=device,
        )
    elif model_name == "v2":
        from network import OnlineRLV2

        agent = OnlineRLV2(
            action_dim=int(online_config.action_dim),
            hidden_dim=int(getattr(online_config, "hidden_dim", 256)),
            foundation_ckpt=foundation_ckpt if foundation_ckpt else None,
            use_pose_encoder=bool(getattr(online_config, "use_pose_encoder", False)),
            pose_dim=int(getattr(online_config, "pose_dim", 7)),
            pose_hidden_dim=int(getattr(online_config, "pose_hidden_dim", 64)),
            freeze_backbone=freeze_backbone,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported ONLINE.model: {online_config.model}")

    if bool(getattr(online_config, "LOAD_PATH", False)):
        model_path = str(getattr(online_config, "MODEL_PATH", "")).strip()
        if not model_path:
            raise ValueError("TRAIN.ONLINE.MODEL_PATH must be set when LOAD_PATH=True.")
        logger.info("loading onlinerl agent ckpt: %s", model_path)
        agent.load_state_dict(torch.load(model_path, map_location=device))
        agent.train()

    logger.info(
        "onlinerl agent built: model=%s experiment=%s freeze_backbone=%s trainer=%s",
        model_name,
        experiment_mode,
        freeze_backbone,
        OnlineRLTrain.__name__,
    )
    return agent, OnlineRLTrain, env


def _run_train(config, train_config):
    from run import Train

    logger.info("task=TRAIN type=%s", train_config.TYPE)
    logger.info("train config: %s", train_config)
    if train_config.TYPE == "HybridNetworkAudio":
        from network import AudioCRNN
        from train import HybridNetworkAudioTrain

        model = AudioCRNN()
        Train(model=model, trainer=HybridNetworkAudioTrain, config=train_config)
        logger.info("train finished: HybridNetworkAudio")
        return
    
    if train_config.TYPE == "HybridNetworkActionSequence":
        from network import HybridNetworkWan
        from train import HybridNetworkActionSequenceTrain

        model = HybridNetworkWan()
        Train(model=model, trainer= HybridNetworkActionSequenceTrain, config=train_config)
        logger.info("train finished:  HybridNetworkActionSequenceTrain")
        return

    if train_config.TYPE == "SemanticAudio":
        from network.hybrid.semantic_audio import SemanticAudioNet
        from train import SemanticAudioTrain

        model = SemanticAudioNet()
        Train(model=model, trainer=SemanticAudioTrain, config=train_config)
        logger.info("train finished: SemanticAudio")
        return

    if train_config.TYPE == "HybridNetwork":
        from network import HybridNetwork
        from train import HybridNetworkTrain

        model = HybridNetwork()
        Train(model=model, trainer=HybridNetworkTrain, config=train_config)
        logger.info("train finished: HybridNetwork")
        return

    if train_config.TYPE == "HybridNetworkUnheard":
        from network import HybridNetworkUnheard
        from train import HybridNetworkTrain

        model = HybridNetworkUnheard()
        Train(model=model, trainer=HybridNetworkTrain, config=train_config)
        logger.info("train finished: HybridNetworkUnheard")
        return

    if train_config.TYPE == "AudioOnlyNetwork":
        from network.hybrid.audio_only_model import AudioOnlyNetwork
        from train import HybridNetworkTrain

        model = AudioOnlyNetwork()
        Train(model=model, trainer=HybridNetworkTrain, config=train_config)
        logger.info("train finished: AudioOnlyNetwork")
        return

    if train_config.TYPE == "VisualOnlyNetwork":
        from network.hybrid.visual_only_model import VisualOnlyNetwork
        from train import HybridNetworkTrain

        model = VisualOnlyNetwork()
        Train(model=model, trainer=HybridNetworkTrain, config=train_config)
        logger.info("train finished: VisualOnlyNetwork")
        return

    if train_config.TYPE == "AVWANHybridNetwork":
        from network import AVWANNetwork
        from train import AVWANHybridTrain

        model = AVWANNetwork()
        Train(model=model, trainer=AVWANHybridTrain, config=train_config)
        logger.info("train finished: AVWANHybridNetwork")
        return

    if train_config.TYPE == "AVWANHybridNetworkFoundation":
        from network import AVWANNetwork
        from train import AVWANHybridTrain

        map_size = int(getattr(train_config, "MAP_SIZE", 9))
        model = AVWANNetwork(action_dim=map_size * map_size)
        Train(model=model, trainer=AVWANHybridTrain, config=train_config)
        logger.info("train finished: AVWANHybridNetworkFoundation")
        return

    if train_config.TYPE == "OfflineRL":
        offline_config = train_config.OFFLINE
        sac_model, trainer, online_test = _build_offline_agent(config, offline_config)
        Train(
            model=sac_model,
            trainer=trainer,
            config=offline_config,
            online_test=online_test,
        )
        logger.info("train finished: OfflineRL")
        return

    if train_config.TYPE == "OnlineRL":
        online_config = train_config.ONLINE
        agent, trainer, env = _build_onlinerl_agent(config, online_config)
        Train(
            model=agent,
            trainer=trainer,
            config=online_config,
            env=env,
        )
        logger.info("train finished: OnlineRL")
        return

    if train_config.TYPE == "DORL":
        from DORL.train_rl import DORLTrainConfig, run_dorl_train
        from DORL.train_rl_ppo import DORLPPOTrainConfig, run_dorl_ppo_train

        dorl_config = train_config.DORL
        dorl_algo = str(getattr(dorl_config, "type", "sac")).strip().lower()
        dorl_get = lambda key, default: _get_dorl_value(dorl_config, dorl_algo, key, default)
        dorl_pt_root = str(dorl_get("DORL_PT_ROOT", "media/pt/offline_muti_embedding_DORL"))
        mismatch_reward = float(dorl_get("mismatch_reward", -10.0))

        if dorl_algo == "sac":
            target_entropy = dorl_get("target_entropy", None)
            if target_entropy in ("", "None"):
                target_entropy = None
            elif target_entropy is not None:
                target_entropy = float(target_entropy)

            dorl_cfg = DORLTrainConfig(
                action_dim=int(dorl_get("action_dim", 4)),
                hidden_dim=int(dorl_get("hidden_dim", 256)),
                actor_lr=float(dorl_get("actor_lr", 3e-4)),
                critic_lr=float(dorl_get("critic_lr", 3e-4)),
                alpha_lr=float(dorl_get("alpha_lr", 1e-4)),
                target_entropy=target_entropy,
                gamma=float(dorl_get("gamma", 0.99)),
                tau=float(dorl_get("tau", 0.005)),
                batch_size=int(dorl_get("batch_size", 256)),
                buffer_size=int(dorl_get("buffer_size", 200000)),
                warmup_steps=int(dorl_get("warmup_steps", 2000)),
                updates_per_step=int(dorl_get("updates_per_step", 1)),
                max_grad_norm=float(dorl_get("max_grad_norm", 1.0)),
                q_target_min=float(dorl_get("q_target_min", -100.0)),
                q_target_max=float(dorl_get("q_target_max", 100.0)),
                log_alpha_min=float(dorl_get("log_alpha_min", -10.0)),
                log_alpha_max=float(dorl_get("log_alpha_max", 2.0)),
                train_epochs=int(dorl_get("train_epochs", 50)),
                episodes_per_epoch=int(dorl_get("episodes_per_epoch", 0)),
                max_steps=int(dorl_get("max_steps", 200)),
                save_every=int(dorl_get("save_every", 5)),
                ckpt_dir=str(dorl_get("ckpt_dir", "media/DORL/ckpt")),
                stochastic_policy=bool(dorl_get("stochastic_policy", True)),
                greedy_prob_start=float(dorl_get("greedy_prob_start", 0.15)),
                greedy_prob_end=float(dorl_get("greedy_prob_end", 0.02)),
                greedy_decay_epochs=int(dorl_get("greedy_decay_epochs", 25)),
                tb_log_dir=str(dorl_get("tb_log_dir", "media/DORL/log")),
                log_interval=int(dorl_get("log_interval", 100)),
                reward_unscale_enable=bool(dorl_get("reward_unscale_enable", False)),
                reward_scale_factor=float(dorl_get("reward_scale_factor", 1.0)),
                mismatch_reward=mismatch_reward,
            )
            run_dorl_train(
                dorl_pt_root=dorl_pt_root,
                mismatch_reward=mismatch_reward,
                config=dorl_cfg,
            )
            logger.info("train finished: DORL (sac)")
            return

        if dorl_algo == "ppo":
            online_eval_fn = _build_dorl_ppo_online_eval_fn(config=config, dorl_config=dorl_config, dorl_algo=dorl_algo)
            dorl_cfg = DORLPPOTrainConfig(
                action_dim=int(dorl_get("action_dim", 4)),
                hidden_dim=int(dorl_get("hidden_dim", 256)),
                actor_lr=float(dorl_get("actor_lr", 3e-4)),
                critic_lr=float(dorl_get("critic_lr", 3e-4)),
                gamma=float(dorl_get("gamma", 0.99)),
                gae_lambda=float(dorl_get("gae_lambda", 0.95)),
                ppo_clip=float(dorl_get("ppo_clip", 0.2)),
                ppo_epochs=int(dorl_get("ppo_epochs", 4)),
                minibatch_size=int(dorl_get("minibatch_size", 256)),
                value_coef=float(dorl_get("value_coef", 0.5)),
                entropy_coef=float(dorl_get("entropy_coef", 0.01)),
                entropy_coef_end=float(dorl_get("entropy_coef_end", 0.001)),
                entropy_decay_epochs=int(dorl_get("entropy_decay_epochs", 80)),
                max_grad_norm=float(dorl_get("max_grad_norm", 1.0)),
                target_kl=float(dorl_get("target_kl", 0.02)),
                value_clip=float(dorl_get("value_clip", 0.2)),
                lr_decay_enable=bool(dorl_get("lr_decay_enable", True)),
                eval_greedy_episodes=int(dorl_get("eval_greedy_episodes", 64)),
                train_epochs=int(dorl_get("train_epochs", 50)),
                episodes_per_epoch=int(dorl_get("episodes_per_epoch", 0)),
                max_steps=int(dorl_get("max_steps", 200)),
                save_every=int(dorl_get("save_every", 5)),
                ckpt_dir=str(dorl_get("ckpt_dir", "media/DORL/ckpt_ppo")),
                tb_log_dir=str(dorl_get("tb_log_dir", "media/DORL/log_ppo")),
                stochastic_policy=bool(dorl_get("stochastic_policy", True)),
                greedy_prob_start=float(dorl_get("greedy_prob_start", 0.15)),
                greedy_prob_end=float(dorl_get("greedy_prob_end", 0.02)),
                greedy_decay_epochs=int(dorl_get("greedy_decay_epochs", 25)),
                log_interval=int(dorl_get("log_interval", 100)),
                mismatch_reward=mismatch_reward,
                advantage_norm_eps=float(dorl_get("advantage_norm_eps", 1e-8)),
            )
            run_dorl_ppo_train(
                dorl_pt_root=dorl_pt_root,
                mismatch_reward=mismatch_reward,
                config=dorl_cfg,
                online_eval_fn=online_eval_fn,
            )
            logger.info("train finished: DORL (ppo)")
            return

        raise ValueError(f"Unsupported TRAIN.DORL.type: {dorl_algo} (expected 'sac' or 'ppo')")

    raise ValueError(f"Unsupported TRAIN.TYPE: {train_config.TYPE}")


def main():
    task_config = config.TASK_CONFIG
    logger.info("run begin")
    logger.info(
        "task flags: collect=%s pt=%s eval=%s train=%s",
        task_config.COLLECT.OPEN,
        task_config.PT.OPEN,
        task_config.EVAL.OPEN,
        task_config.TRAIN.OPEN,
    )
    any_open = (
        task_config.COLLECT.OPEN
        or task_config.PT.OPEN
        or task_config.EVAL.OPEN
        or task_config.TRAIN.OPEN
    )
    if not any_open:
        raise ValueError("No task is enabled. Please set one OPEN field to True.")

    if task_config.COLLECT.OPEN:
        _run_collect(config, task_config.COLLECT)
    if task_config.PT.OPEN:
        _run_pt(task_config.PT)
    if task_config.EVAL.OPEN:
        _run_eval(config, task_config.EVAL)
    if task_config.TRAIN.OPEN:
        _run_train(config, task_config.TRAIN)
    logger.info("run finished")


config = get_config()

if __name__ == "__main__":
    run_name = "run"
    task_cfg = config.TASK_CONFIG
    if task_cfg.COLLECT.OPEN:
        run_name = f"collect_{task_cfg.COLLECT.TYPE}"
    elif task_cfg.PT.OPEN:
        run_name = f"pt_{task_cfg.PT.TYPE}"
    elif task_cfg.EVAL.OPEN:
        run_name = f"eval_{task_cfg.EVAL.TYPE}"
    elif task_cfg.TRAIN.OPEN:
        run_name = f"train_{task_cfg.TRAIN.TYPE}"

    log_path = setup_run_logger(base_dir="logs", run_name=run_name)
    logger.info("python run.py started, log_path=%s", log_path)
    exp_note = os.environ.get("EXP_NOTE", "").strip()
    journal_path = append_experiment_journal(
        run_name=run_name,
        note=exp_note,
        journal_path="logs/experiment_journal.md",
    )
    logger.info("experiment journal appended: %s", journal_path)
    try:
        main()
    except Exception:
        logger.exception("run failed with exception")
        raise
