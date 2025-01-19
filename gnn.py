import typing
import torch

from torch_geometric.nn import PointGNNConv

from mlp import Mlp
    
class MlpH(Mlp):
    """ mlp that maps from node features to the delta x shifts"""

class MlpF():
    """ multi-layer perceptron that transforms features from neighboring nodes
    and relative positions into edge features
    """
    def __init__(self,
                 feat_dim : int,
                 hidden_dim_lst : typing.List[int],
                 activation : str = "relu",
                 output_activation : str = "none"):
        
        # the input consists of two tensors: the feature tensor and the relative
        # position tensor which has 3 dimensions
        self.mlp = Mlp([feat_dim+3]+hidden_dim_lst, 
                         activation, output_activation)

    def forward(self, rel_pos, x):
        return self.mlp(torch.cat([rel_pos, x], dim=1))

class MlpG(Mlp):
    """ mlp that maps from aggregated edge features to node features"""

class GNNLayer(torch.nn.Module):
    def __init__(self, 
                 config : dict,
                 ):
        super().__init__()

        # give the layer a name
        self.name = config["name"]

        # for each gnn layer, define the 3 MLP modules for delta_x, edge, and
        # feature calculation
        self.mlph = MlpH(**config["mlp_h"])
        self.mlpf = MlpF(**config["mlp_f"])
        self.mlpg = MlpG(**config["mlp_g"])

        self.conv = PointGNNConv(self.mlph, self.mlpf, self.mlpg)

    def forward(self, x, pos, edge_index):
        
        return self.conv(x, pos, edge_index)
    
class GNN(torch.nn.Module):
    def __init__(self, 
                 layer_configs : dict,
                 ):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [GNNLayer(lc) for lc in layer_configs])

    def forward(self, x, pos, edge_index):
        for layer in self.layers:
            x = layer(x, pos, edge_index)
        return x