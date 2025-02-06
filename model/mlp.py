import typing
import torch
from torch.nn import Linear

from model.registry import ModuleRegistry
from config.activations import activation_dict

class Mlp(torch.nn.Module, ModuleRegistry):
    """multi-layer perceptron

    Args:
        torch (_type_): _description_
    """
    def __init__(self,
                 inp_dim : int,
                 hidden_dim_lst : typing.List[int],
                 out_dim : int,
                 activation : str = "relu",
                 output_activation : str = "none",
                 add_batch_norm : bool = False,
                 return_dict : bool = False):
        """_summary_

        Args:
            dim_lst (typing.List[int]): list of input and hidden layer dims
            the output dimension is always 3
            activation (str, optional): activation function of intermediate 
            layers. defaults to "relu".
            output_activation (str, optional): activation function of the output
            layer. defaults to "none". 
            add_batch_norm (bool, optional): whether to add batch normalization
            return_dict (bool, optional): whether to return the output as a dict
        """
        super().__init__()
        layers = []
        in_dims = [inp_dim] + hidden_dim_lst
        out_dims = hidden_dim_lst + [out_dim]
        n_layers = len(in_dims)
        for i in range(n_layers):
            layers.append(Linear(in_dims[i], out_dims[i]))
            if i < n_layers - 1:
                if activation != "none":
                    layers.append(activation_dict[activation]())
            else: # last layer
                if output_activation != "none":
                    layers.append(activation_dict[output_activation]())
            
            if add_batch_norm:
                layers.append(torch.nn.BatchNorm1d(out_dims[i]))

        self.mlp = torch.nn.Sequential(*layers)
        self.return_dict = return_dict
    
    def forward(self, x):
        if self.return_dict:
            return {"x": self.mlp(x)}
        else:
            return self.mlp(x)