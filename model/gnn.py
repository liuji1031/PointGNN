import typing
import torch

from torch_geometric.nn import PointGNNConv, PointNetConv

from model.mlp import Mlp
from model.registry import ModuleRegistry
    
class MlpH(Mlp):
    """ mlp that maps from node features to the delta x shifts"""

class MlpF(Mlp):
    """ multi-layer perceptron that transforms features from neighboring nodes
    and relative positions into edge features
    """
    def __init__(self, **kwargs):
        # add the edge feature dimension to the kwargs
        assert "inp_dim" in kwargs, "Must provide inp_dim for MlpF"
        kwargs["inp_dim"] = kwargs["inp_dim"]+3 # add 3 for the relative position features
        super().__init__(**kwargs)

class MlpG(Mlp):
    """ mlp that maps from aggregated edge features to node features"""

class PointGnnLayer(torch.nn.Module, ModuleRegistry):
    def __init__(self, 
                 mlp_h : dict,
                 mlp_f : dict,
                 mlp_g : dict,
                 **kwargs,
                 ):
        super().__init__()

        # give the layer a name
        self.name = kwargs.get("name", "PointGnnLayer")

        # for each gnn layer, define the 3 MLP modules for delta_x, edge, and
        # feature calculation
        self.mlph = MlpH(**mlp_h)
        self.mlpf = MlpF(**mlp_f)
        self.mlpg = MlpG(**mlp_g)

        self.conv = PointGNNConv(self.mlph, self.mlpf, self.mlpg)
        self.return_dict = kwargs.get("return_dict", False)

    def forward(self, x, pos, edge_index):
        if not self.return_dict:
            return self.conv(x, pos, edge_index)
        else:
            return {"x": self.conv(x, pos, edge_index)}
    
class GNN(torch.nn.Module):
    def __init__(self, 
                 layer_configs : dict,
                 ):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [PointGnnLayer(lc) for lc in layer_configs])

    def forward(self, x, pos, edge_index):
        for layer in self.layers:
            x = layer(x, pos, edge_index)
        return x
    
class PointNetEncoder(torch.nn.Module, ModuleRegistry):
    def __init__(self, 
                 local_nn : dict,
                 global_nn : dict,
                 **kwargs,
                 ):
        super().__init__()
        self.name = kwargs.get("name", "PointNetEncoder")
        assert "inp_dim" in local_nn, "Must provide inp_dim for PointNetEncoder local_nn"
        local_nn["inp_dim"] = local_nn["inp_dim"]+3 # add 3 for the relative position features
        self.local_nn = Mlp(**local_nn)
        self.global_nn = Mlp(**global_nn)
        self.return_dict = kwargs.get("return_dict", False)
        self.conv = PointNetConv(self.local_nn, self.global_nn)

    def forward(self, x, pos, edge_index):
        if self.return_dict:
            return {"x": self.conv(x, pos, edge_index)}
        else:
            return self.conv(x, pos, edge_index)