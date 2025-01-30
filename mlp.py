import typing
import torch
from torch.nn import Linear

from config.activations import activation_dict

class Mlp(torch.nn.Module):
    """multi-layer perceptron

    Args:
        torch (_type_): _description_
    """
    def __init__(self,
                 dim_lst : typing.List[int],
                 activation : str = "relu",
                 output_activation : str = "none",
                 add_batch_norm : bool = False):
        """_summary_

        Args:
            dim_lst (typing.List[int]): list of input and hidden layer dims
            the output dimension is always 3
            activation (str, optional): activation function of intermediate 
            layers. defaults to "relu".
            output_activation (str, optional): activation function of the output
            layer. defaults to "none". 
            add_batch_norm (bool, optional): whether to add batch normalization
        """
        super().__init__()
        layers = []
        for i in range(len(dim_lst)-1):
            layers.append(Linear(dim_lst[i], dim_lst[i+1]))
            if i < len(dim_lst)-2:
                if activation != "none":
                    layers.append(activation_dict[activation]())
            else:
                if output_activation != "none":
                    layers.append(activation_dict[output_activation]())
            
            if add_batch_norm:
                layers.append(torch.nn.BatchNorm1d(dim_lst[i+1]))

        self.mlp = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)