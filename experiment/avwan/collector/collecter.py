"""Select and run a collector using TASK_CONFIG.COLLECT.TYPE."""
from .register import CollectRegister


class COLLECTER:
    def __init__(self, config, env):
        self.collect_config = config.TASK_CONFIG.COLLECT
        cls = CollectRegister.get(self.collect_config.TYPE)
        self.collecter = cls(env, config)

    def collect(self):
        return self.collecter.collect()
