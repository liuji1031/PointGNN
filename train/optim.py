from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


class Optimization:
    OPTIMIZERS = {"adam": Adam}
    SCHEDULERS = {"step": StepLR, "plateau": ReduceLROnPlateau}

    @classmethod
    def optimizer(cls, params, name: str, config: dict):
        assert name.lower() in cls.OPTIMIZERS, f"Unknown optimizer {name}"
        return cls.OPTIMIZERS[name.lower()](params=params, **config)

    @classmethod
    def scheduler(cls, optimizer, name: str, config: dict):
        assert name.lower() in cls.SCHEDULERS, f"Unknown scheduler {name}"
        return cls.SCHEDULERS[name.lower()](optimizer, **config)
