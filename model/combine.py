import torch
from torch import jit
from model.registry import ModuleRegistry

class Combine(torch.nn.Module, ModuleRegistry):
    def __init__(self, operation : str, return_dict : bool = False):
        super(Combine, self).__init__()
        self.operation = operation
        self.return_dict = return_dict

    def forward(self, inp1, inp2):
        if self.operation == 'concat':
            out = torch.cat([inp1, inp2], dim=1)
        elif self.operation == 'add':
            out = inp1 + inp2
        elif self.operation == 'multiply':
            out = inp1 * inp2
        elif self.operation == 'subtract':
            out = inp1 - inp2
        else:
            raise ValueError(f"Operation {self.operation} not supported")
        
        if self.return_dict:
            return {"out": out}
        else:
            return out