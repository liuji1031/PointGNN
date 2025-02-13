import sys

import torch
import torch_geometric


class ModuleRegistry:
    """Register all module classes."""

    REGISTRY = {}
    TORCH_REGISTRY = {"torch_gcn_conv": torch_geometric.nn.GCNConv,
                      "batch_norm_1d": torch.nn.BatchNorm1d,}

    @classmethod
    def register(cls, subclass_name):
        def decorator(subclass):
            if subclass_name in cls.REGISTRY:
                raise Warning(f"Overwriting {subclass_name}")
            cls.REGISTRY[subclass_name] = subclass
            return subclass

        return decorator

    @classmethod
    def build(cls, subclass_name, module_name=None, **kwargs):
        if (
            subclass_name not in cls.REGISTRY
            and subclass_name not in cls.TORCH_REGISTRY
        ):
            raise ValueError(f"Unknown class {subclass_name}")
        module_name = module_name if module_name is not None else subclass_name
        if subclass_name in cls.REGISTRY:
            return cls.REGISTRY[subclass_name](name=module_name, **kwargs)
        else:
            return cls.TORCH_REGISTRY[subclass_name](**kwargs)


class Module:
    """Base class for all modules."""

    def __init__(
        self,
        name="module",
        **kwargs,
    ):
        super().__init__()
        self.name = name


class NNModule(Module, torch.nn.Module):
    """Base class for all neural network modules."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
