"""Experiment package exports, loaded lazily to keep task packages isolated."""


def __getattr__(name):
    if name == "AVWANRLEnv":
        from experiment.avwan.environment.avwan_env import AVWANRLEnv
        return AVWANRLEnv
    if name == "get_avwan_config":
        from experiment.avwan.config.default import get_config
        return get_config
    raise AttributeError(name)


__all__ = ["AVWANRLEnv", "get_avwan_config"]
