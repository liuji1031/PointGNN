import sys
import torch
sys.path.append('.')

from util import camel_to_snake

class ModuleRegistry:
    """Register all module classes."""
    REGISTRY = {}
    # def __init_subclass__(cls, **kwargs):
    #     ModuleRegistry.REGISTRY[camel_to_snake(cls.__name__)] = cls
    #     super().__init_subclass__(**kwargs)

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            if name in cls.REGISTRY:
                raise ValueError(f"Cannot register duplicate class ({name})")
            cls.REGISTRY[name] = subclass
            return subclass
        return decorator
    
    @classmethod
    def build(cls, name, **kwargs):
        if name not in cls.REGISTRY:
            raise ValueError(f"Unknown class {name}")
        return cls.REGISTRY[name](**kwargs)

class Module(torch.nn.Module):
    """Base class for all modules."""
    def __init__(self, **kwargs):
        super().__init__()
        assert "return_dict" in kwargs, "return_dict (bool) must be specified"
        self.return_dict = kwargs["return_dict"]