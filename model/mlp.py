import typing

import torch
from torch.nn import Linear

from model.registry import NNModule, ModuleRegistry


@ModuleRegistry.register("mlp")
class Mlp(NNModule):
    """multi-layer perceptron

    Args:
        torch (_type_): _description_
    """

    ACTIVATION_DICT = {
        "relu": torch.nn.ReLU,
        "tanh": torch.nn.Tanh,
        "elu": torch.nn.ELU,
        "leaky_relu": torch.nn.LeakyReLU,
        "none": torch.nn.Identity,
    }

    def __init__(
        self,
        inp_dim: int,
        hidden_dim_lst: typing.List[int],
        out_dim: int,
        activation: str = "relu",
        output_activation: str = "none",
        norm: str = "none",
        **kwargs,
    ):
        """_summary_

        Args:
            dim_lst (typing.List[int]): list of input and hidden layer dims
            the output dimension is always 3
            activation (str, optional): activation function of intermediate
            layers. defaults to "relu".
            output_activation (str, optional): activation function of the output
            layer. defaults to "none".
            norm (str, optional): normalization layer. defaults to "none". other options
            include "batch_norm" and "layer_norm".
            return_dict (bool, optional): whether to return the output as a dict
        """
        super().__init__(**kwargs)
        self.inp_dim = inp_dim
        self.hidden_dim_lst = hidden_dim_lst
        self.out_dim = out_dim
        layers = []
        in_dims = [inp_dim] + hidden_dim_lst
        out_dims = hidden_dim_lst + [out_dim]
        n_layers = len(in_dims)
        for i in range(n_layers):
            lin_layer = Linear(in_dims[i], out_dims[i])
            layers.append(lin_layer)

            if i < n_layers - 1:
                if activation != "none":
                    layers.append(self.ACTIVATION_DICT[activation]())
            else:  # last layer
                if output_activation != "none":
                    layers.append(self.ACTIVATION_DICT[output_activation]())

            if norm == "batch_norm":
                layers.append(torch.nn.BatchNorm1d(out_dims[i]))
            elif norm == "layer_norm":
                layers.append(torch.nn.LayerNorm(out_dims[i]))

        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
