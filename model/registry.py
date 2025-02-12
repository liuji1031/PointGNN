import sys

import torch
import torch_geometric


class ModuleRegistry:
    """Register all module classes."""

    REGISTRY = {}
    TORCH_REGISTRY = {"torch_gcn_conv": torch_geometric.nn.GCNConv}

    @classmethod
    def register(cls, subclass_name):
        def decorator(subclass):
            if subclass_name in cls.REGISTRY:
                raise Warning(f"Overwriting {subclass_name}")
            print(f"Registering {subclass_name}")
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
        out_varname=None | str | list[str],
        prefix_module_name=True,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.out_varname = out_varname
        self.prefix_module_name = prefix_module_name


class NNModule(Module, torch.nn.Module):
    """Base class for all neural network modules."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        out_varname = kwargs.get("out_varname", None)
        if out_varname is not None:
            self.return_dict = True
            assert isinstance(out_varname, str), (
                f"out_varname must be a string for {self._get_name()}"
            )
        else:
            self.return_dict = False

    def _construct_result(self, result):
        if not self.return_dict:
            return result
        else:
            # return as a dictionary otherwise
            return {f"{self.name}:{self.out_varname}": result}


@ModuleRegistry.register("entry")
class Entry(Module):
    """Entry point module."""

    def __init__(self, out_varname: list, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(out_varname, list), (
            f"out_varname must be a list for {self._get_name()}"
        )
        self.out_varname = out_varname

    def __call__(self, *args):
        out = {}
        assert len(args) == len(self.out_varname), (
            "Number of arguments does not match number of input variables"
        )
        for varname, arg in zip(self.out_varname, args):
            if self.prefix_module_name:
                key = f"{self.name}:{varname}"
            else:
                key = varname
            out[key] = arg

        return out


@ModuleRegistry.register("exit")
class Exit(Entry):
    """Exit point module.

    Behaves the same as Entry module.
    """

    ...
