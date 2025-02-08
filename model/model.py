from typing import List, Tuple, Union

from loguru import logger
import torch

from model.registry import ModuleRegistry


class ComposableModel(torch.nn.Module):
    """Build full model based on config file."""
    def __init__(self, model_name:str, config:dict):
        super().__init__()
        self.model_name = model_name
        self._module_info = {}
        self._inp_num = {}
        self._module_output = {}
        self._des = {}
        self._entrypoint = None
        self._exitpoint = None
        self._build(config)

    def _build(self, config_list: List):
        """Build the model based on config file.

        Args:
            config (dict): configuration dictionary
        """
        for module_config in config_list:
            self._build_module(module_config)

    def _parse_pattern(pattern):
        module_name, port = pattern.split(":")
        return module_name, port

    def _build_module(self, module_config):
        """Build a single module based on config file.

        Args:
            module_config (dict): configuration dictionary for a single module
        """
        module_name = module_config["name"]
        cls_name = module_config["cls"]
        self._module_info[module_name] = {}
        if cls_name not in ["entry", "exit"]:
            assert cls_name in ModuleRegistry.REGISTRY, (
                f"Module class {cls_name} not found"
            )
            module = ModuleRegistry.REGISTRY[cls_name](
                **module_config["build_cfg"]
            )
            # build the module
            self.register_module(module_name, module)

        self._module_info[module_name]["inp_src"] = module_config["inp_src"]
        if not module_name.startswith("entry") and not module_name.startswith("exit"):
            self._module_info[module_name]["out_varname"] = module_config["out_varname"]
        self._inp_num[module_name] = len(module_config["inp_src"])

        for des, src in module_config["inp_src"].items():
            # create the list of destinations for each source
            if src not in self._des:
                self._des[src] = []
            self._des[src].append(f"{module_name}:{des}")

        if cls_name=="entry":
            self._entrypoint = module_name
        elif cls_name=="exit":
            self._exitpoint = module_name

    def _init_inp_cntr(self):
        """Build input dictionary for each module."""
        inp_cntr = {}
        for module_name, _ in self._module_info.items():
            inp_cntr[module_name] = 0
        return inp_cntr

    @staticmethod
    def _all_inputs_ready(inp_dict, module_name):
        """Check if all inputs are ready for a module.

        Args:
            inp_dict (dict): input dictionary
            module_name (str): module name

        Returns:
            bool: True if all inputs are ready, False otherwise
        """
        return all([inp is not None for inp in inp_dict[module_name]])
    
    def _entrypoint_output(self, args):
        """Mimic actual module output for entrypoint.
        
        Generate a dictionary of output for the entrypoint module.
        """
        out = {}
        module_name = self._entrypoint
        for arg, des in zip(args, self._module_info[module_name]["inp_src"].keys()):
            out[f"{des}"] = arg
        return out

    # def to_device(self, device):
    #     """Move the model to a specific device.

    #     Args:
    #         device (torch.device): device to move the model to
    #     """
    #     for _, module in self._module_info.items():
    #         if module["module"] is not None:
    #             module["module"] = module["module"].to(device)

    def forward(self,*args):
        """Forward pass of the model.

        Args:
            inp1 (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        # x, pos, edge_index = args
        # point_net_encoder = self._module_info["point_net_encoder"]["module"]
        # x = point_net_encoder(x, pos, edge_index)
        # point_gnn_l1 = self._module_info["point_gnn_l1"]["module"]
        # x = point_gnn_l1(x, pos, edge_index)
        # point_gnn_l2 = self._module_info["point_gnn_l2"]["module"]
        # x = point_gnn_l2(x, pos, edge_index)
        # point_gnn_l3 = self._module_info["point_gnn_l3"]["module"]
        # x = point_gnn_l3(x, pos, edge_index)
        # return x

        module_output = {}
        inp_cntr = self._init_inp_cntr()

        operation_queue = []

        # enqueue the entrypoint
        operation_queue.append(self._entrypoint)

        while len(operation_queue) > 0:
            # any module in the queue must already have all inputs ready
            module_name = operation_queue.pop(0)
            # logger.debug(f"Processing module {module_name}")
            module = self._modules[module_name]
            # out_varname = self._module_info[module_name]["out_varname"]

            # out is a dictionary of output, example: {"x": tensor}
            if not module_name.startswith("entry"):
                # construct input dictionary for the module
                inp = {}
                for des, src in self._module_info[module_name]["inp_src"].items():
                    # assert src in module_output, f"Missing output from {src}"
                    inp[des] = module_output[src]
                # call the module with kwargs
                out = module(**inp)
            else:
                out = self._entrypoint_output(args)
            assert isinstance(out, dict), "Output of the module must be a dictionary"

            for varname, val in out.items():
                full_varname = f"{module_name}:{varname}"
                module_output[full_varname] = val

                if full_varname not in self._des:
                    continue

                # update input counter for the corresponding destination modules
                for des in self._des[full_varname]:
                    # example des = "mlp_2:inp1"
                    des_module_name, des_port = ComposableModel._parse_pattern(des)
                    inp_cntr[des_module_name] += 1

                    if des_module_name == self._exitpoint:
                        continue

                    # check if all inputs are ready, enqueue if so
                    if inp_cntr[des_module_name] == self._inp_num[des_module_name]:
                        operation_queue.append(des_module_name)

        final_output = {}
        for varname, src in self._module_info[self._exitpoint]["inp_src"].items():
            final_output[varname] = module_output[src]

        return final_output