from typing import List, Tuple, Union

import torch
from loguru import logger

from model.registry import ModuleRegistry


class ComposableModel(torch.nn.Module):
    """Build full model based on config file."""

    def __init__(self, model_name: str, config: dict):
        super().__init__()
        self.name = model_name
        self._module_info = {}
        self._inp_num = {}
        self._module_output = {}
        self._des = {}
        self._build(config)
        self._op_seq = self._forward_dry_run()

    def _build(self, config_list: List):
        """Build the model based on config file.

        Args:
            config (dict): configuration dictionary
        """
        for module_config in config_list:
            self._build_module(module_config)

    def _build_module(self, module_config):
        """Build a single module based on config file.

        Args:
            module_config (dict): configuration dictionary for a single module
        """
        module_name = module_config["name"]

        self._module_info[module_name] = {}

        if module_name not in ["entry", "exit"]:
            # build the module
            cls_name = module_config["cls"]
            module = ModuleRegistry.build(
                cls_name, module_name, **module_config["config"]
            )
            self.register_module(module_name, module)

        self._module_info[module_name]["inp_src"] = module_config[
            "inp_src"
        ]  # inp_src is a list
        self._inp_num[module_name] = len(module_config["inp_src"])

        if module_name not in ["entry", "exit"]:
            for iarg, src in enumerate(module_config["inp_src"]):
                # create the list of destinations for each source
                if src not in self._des:
                    self._des[src] = []
                self._des[src].append(f"{module_name}:{iarg}")

    def _forward_dry_run(self):
        """Figure out the sequence of operations by simulating forward pass."""
        op_seq = []
        # create counter of input number ready for each module
        inp_cntr = {}
        for module_name in self._module_info.keys():
            inp_cntr[module_name] = 0

        # queue of modules to be processed, start with entry
        q = ["entry"]
        while len(q) > 0:
            module_name = q.pop(0)
            # construct output dictionary for the module
            if module_name == "entry":
                # entry module takes input from positional arguments
                n_input = self._inp_num[module_name]
                out_varname = [f"args:{i}" for i in range(n_input)]
            else:
                out_varname = self._modules[module_name].out_varname
                out_varname = [f"{module_name}:{out_varname}"]

            for varname in out_varname:
                if varname not in self._des:
                    continue
                # update the input counter for destination modules that depend
                # on this output
                for des in self._des[varname]:
                    des_module_name, des_port = des.split(":")
                    inp_cntr[des_module_name] += 1
                    if (
                        inp_cntr[des_module_name]
                        == self._inp_num[des_module_name]
                    ):
                        # all inputs are ready, add module to queue
                        q.append(des_module_name)

            if module_name not in ["entry", "exit"]:
                op_seq.append(module_name)  # add module to the sequence

        return op_seq

    def forward(self, *args):
        """Forward pass of the model.

        Args:
            inp1 (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        module_output = {}
        for iarg, arg in enumerate(args):
            module_output[f"args:{iarg}"] = arg
        final_output = {}

        for module_name in self._op_seq:
            # construct input dictionary for the module
            inp = [None for _ in range(self._inp_num[module_name])]
            for iarg, src in enumerate(
                self._module_info[module_name]["inp_src"]
            ):
                inp[iarg] = module_output[src]
            # print(f"calling {module_name} with inp: {inp}")
            try:
                out = self._modules[module_name](*inp)
            except Exception as e:
                logger.error(f"Error in module {module_name}: {e}")
                raise e
            module_output = {**module_output, **out}

        for des, src in self._module_info["exit"][
            "inp_src"
        ].items():  # inp_src of exit is a dict
            final_output[des] = module_output[src]
        return final_output
