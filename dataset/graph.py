from torch_geometric.transforms import KNNGraph, RadiusGraph

class GraphGenFactory:
    REGISTRY = {"knn": KNNGraph, "radius": RadiusGraph}

    @classmethod
    def build(cls,method_name:str, config:dict):
        assert method_name.lower() in cls.REGISTRY, f"Unknown graph generation method {method_name}"
        return cls.REGISTRY[method_name.lower()](**config)