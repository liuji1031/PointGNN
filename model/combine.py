import torch
from model.registry import Module, ModuleRegistry

@ModuleRegistry.register("combine")
class Combine(Module):
    def __init__(self, operation : str, **kwargs):
        super().__init__(**kwargs)
        self.operation = operation

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
        
        return self._construct_result(out)