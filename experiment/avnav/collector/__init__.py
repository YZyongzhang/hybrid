"""AVNav episode collection helpers, imported lazily to avoid simulator side effects."""


def __getattr__(name):
    if name == "COLLECTER":
        from experiment.avnav.collector.collecter import COLLECTER
        return COLLECTER
    if name == "CollectRegister":
        from experiment.avnav.collector.register import CollectRegister
        return CollectRegister
    raise AttributeError(name)


__all__ = ["COLLECTER", "CollectRegister"]
