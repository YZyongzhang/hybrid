"""Collector registry, following the AVNAV collector interface."""
from .collect import GreedyCollect


class REGISTER:
    def __init__(self):
        self._registry = {}

    def register(self, name):
        def decorator(cls):
            if name in self._registry:
                raise ValueError(f'{name} is already registered')
            self._registry[name] = cls
            return cls
        return decorator

    def get(self, name):
        if name not in self._registry:
            raise KeyError(f'Unknown collector {name}; available: {list(self._registry)}')
        return self._registry[name]

    def available(self):
        return list(self._registry)


CollectRegister = REGISTER()
CollectRegister.register('greedy')(GreedyCollect)
CollectRegister.register('avwan_greedy')(GreedyCollect)
