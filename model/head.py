import torch
from torch.nn.functional import softmax, log_softmax
from model.registry import ModuleRegistry
from model.mlp import Mlp

@ModuleRegistry.register("classification_head")
class ClassificationHead(Mlp):
    """ the classification head for predicting the object class of each point.
    returns the logits for each class
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_format = kwargs.get("output_format", "logits")
        assert self.output_format in ["logits", "softmax","log_softmax"], \
            "output_format must be 'logits','softmax', or 'log_softmax'"

    def forward(self, x):
        out = self.mlp(x)
        if self.output_format == "logits":
            pass
        elif self.output_format == "softmax":
            out = softmax(out, dim=-1)
        elif self.output_format == "log_softmax":
            out = log_softmax(out, dim=-1)
        if self.return_dict:
            return {"x": out}
        else:
            return out

@ModuleRegistry.register("background_class_head")
class BackgroundClassHead(ClassificationHead):
    """ the binary classification head for predicting the background class of each point.
    returns the logits for each class
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert "out_dim" in kwargs, "out_dim must be specified for BackgroundClassificationHead"
        assert kwargs["out_dim"] == 2, "out_dim must be 2 for BackgroundClassificationHead"

@ModuleRegistry.register("object_class_head")
class ObjectClassHead(ClassificationHead):
    """ the classification head for predicting the object class of each point.
    returns the logits for each class
    """ 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

@ModuleRegistry.register("box_size_head")
class BoxSizeHead(Mlp):
    """Regress the delta values for the size of the bounding box.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert "out_dim" in kwargs, "out_dim must be specified for BoxSizeHead"
        assert kwargs["out_dim"] == 3, "out_dim must be 3 for BoxSizeHead"
        assert kwargs["output_activation"] != "none", "output_activation cannot be none for BoxSizeHead"

@ModuleRegistry.register("localization_head")
class LocalizationHead(Mlp):
    """Regress the delta values for the shift in x, y, z for bounding box.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert "out_dim" in kwargs, "out_dim must be specified for LocalizationHead"
        assert kwargs["out_dim"] == 3, "out_dim must be 7 for LocalizationHead"

@ModuleRegistry.register("orientation_head")
class OrientationHead(Mlp):
    """Regress the orientation of the bounding box.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert "out_dim" in kwargs, "out_dim must be specified for OrientationHead"
        assert kwargs["out_dim"] == 1, "out_dim must be 1 for OrientationHead"
        self.clip_output = kwargs.get("clip_output", [-1.,1.])

    def forward(self, x):
        out = self.mlp(x)
        out = torch.clip(out, self.clip_output[0], self.clip_output[1])
        if self.return_dict:
            return {"x": out}
        else:
            return out
    
