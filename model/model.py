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
        self._out_num = {}
        self._module_output = {}
        self._des = {}
        self._return_dict = False
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
        if isinstance(module_config["inp_src"], list):
            for isrc, src in enumerate(module_config["inp_src"]):
                parsed_src = src.split(":")
                if len(parsed_src) == 1: 
                    # positional argument number not specified, assume it is 0
                    module_config["inp_src"][isrc] = f"{src}:0"
        elif isinstance(module_config["inp_src"], dict):
            for des, src in module_config["inp_src"].items():
                parsed_src = src.split(":")
                if len(parsed_src) == 1: 
                    # positional argument number not specified, assume it is 0
                    module_config["inp_src"][des] = f"{src}:0"

        self._module_info[module_name]["inp_src"] = module_config[
            "inp_src"
        ]  # inp_src is a list

        self._inp_num[module_name] = len(module_config["inp_src"])
        self._out_num[module_name] = module_config.get("out_num", 1)

        if module_name not in ["entry", "exit"]:
            for iarg, src in enumerate(module_config["inp_src"]):
                # create the list of destinations for each source
                if src not in self._des:
                    self._des[src] = []
                self._des[src].append(f"{module_name}:{iarg}")

        if module_name == "exit" and isinstance(module_config["inp_src"], dict):
            self._return_dict = True

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
                for i in range(self._out_num[module_name]):
                    if i == 0:
                        out_varname = [f"{module_name}:{i}"]
                    else:
                        out_varname.append(f"{module_name}:{i}")

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

            # out is either a single tensor or a tuple of tensors
            if not isinstance(out, tuple):
                out = (out,)

            # out_dict = {}
            # for iout, out_val in enumerate(out):
            #     out_dict[f"{module_name}:{iout}"] = out_val

            # module_output = {**module_output, **out_dict}

            for iout, out_val in enumerate(out):
                module_output[f"{module_name}:{iout}"] = out_val

        if self._return_dict:
            final_output = {}
            for des, src in self._module_info["exit"][
                "inp_src"
            ].items():  # inp_src of exit is a dict
                final_output[des] = module_output[src]
            return final_output
        else:
            final_output = []
            for src in self._module_info["exit"]["inp_src"]:
                final_output.append(module_output[src])
            if len(final_output) == 1:
                return final_output[0]
            else:
                return tuple(final_output)
