import sys
sys.path.append('.')

from util import camel_to_snake

class ModuleRegistry:
    """Register all module classes."""
    REGISTRY = {}
    def __init_subclass__(cls, **kwargs):
        ModuleRegistry.REGISTRY[camel_to_snake(cls.__name__)] = cls
        super().__init_subclass__(**kwargs)