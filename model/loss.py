import torch

from model.registry import NNModule, ModuleRegistry
from loguru import logger

def unpack_result(pred_dict, target_dict, unpack_sequence, mask=None):
    """Unpack from prediction and target dictionaries into a sequence of
    arguments according to the keys specified in unpack_sequence.

    Args:
        pred_dict (_type_): _description_
        target_dict (_type_): _description_
        unpack_sequence (_type_): _description_
        mask (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    args = []
    for key in unpack_sequence:
        if key in pred_dict and key in target_dict:
            args.append(pred_dict[key])
            args.append(target_dict[key])
        elif key == "mask":
            args.append(mask)
        else:
            raise ValueError(
                f"Key {key} not found in prediction or target dictionary"
            )

    return args


@ModuleRegistry.register("huber_loss")
class HuberLoss(NNModule):
    def __init__(self, reduction="mean", delta=1.0, **kwargs):
        """_summary_

        Args:
            kwargs: _description_
        """
        super().__init__(**kwargs)
        self.reduction = reduction
        if reduction == "custom":
            self.loss = torch.nn.HuberLoss(reduction="none", delta=delta)
        elif reduction in ["mean", "sum", "none"]:
            self.loss = torch.nn.HuberLoss(reduction=reduction, delta=delta)
        else:
            raise ValueError(f"Reduction {reduction} not supported")

    def forward(self, pred, target, mask=None):
        """_summary_

        Args:
            pred (_type_): _description_
            target (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.reduction == "custom":
            loss = torch.mean(self.loss(pred[mask], target[mask]))
        else:
            loss = self.loss(pred, target)

        return loss


@ModuleRegistry.register("nll_loss")
class NLLLoss(NNModule):
    def __init__(self, weight=None, reduction="mean", **kwargs):
        """_summary_

        Args:
            kwargs: _description_
        """
        super().__init__(**kwargs)
        self.reduction = reduction
        if weight is not None and not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight)

        if reduction == "custom":
            self.loss = torch.nn.NLLLoss(
                weight=weight,
                reduction="none",
            )
        elif reduction in  ["mean", "sum", "none"]:
            self.loss = torch.nn.NLLLoss(
                weight=weight,
                reduction=reduction,
            )
        else:
            raise ValueError(f"Reduction {reduction} not supported")

    def forward(self, pred, target, mask=None):
        """_summary_

        Args:
            pred (_type_): _description_
            target (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.reduction == "custom":
            loss = torch.mean(self.loss(pred[mask,:], target[mask]))
        else:
            loss = self.loss(pred, target)
        return loss
    
@ModuleRegistry.register("smooth_l1_loss")
class SmoothL1Loss(NNModule):
    def __init__(self, reduction="mean",beta=1.0, **kwargs):
        """_summary_

        Args:
            kwargs: _description_
        """
        super().__init__(**kwargs)
        self.reduction = reduction
        if reduction == "custom":
            self.loss = torch.nn.SmoothL1Loss(reduction="none", beta=beta)
        elif reduction in ["mean", "sum", "none"]:
            self.loss = torch.nn.SmoothL1Loss(reduction=reduction, beta=beta)
        else:
            raise ValueError(f"Reduction {reduction} not supported")
        
    def forward(self, pred, target, mask=None):
        """_summary_

        Args:
            pred (_type_): _description_
            target (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.reduction == "custom":
            loss = torch.mean(self.loss(pred[mask], target[mask]))
        else:
            loss = self.loss(pred, target)
        return loss


@ModuleRegistry.register("total_loss")
class TotalLoss(NNModule):
    def __init__(self, loss_coef: list, **kwargs):
        """_summary_

        Args:
            kwargs: _description_
        """
        super().__init__(**kwargs)
        self.loss_coef = loss_coef

    def forward(self, *args):
        """_summary_

        Args:
            pred (_type_): _description_
            target (_type_): _description_

        Returns:
            _type_: _description_
        """
        total_loss = 0
        assert len(args) == len(self.loss_coef), (
            f"Number of losses ({len(args)}) does not match number of coefficients ({len(self.loss_coef)})"
        )
        for coef, loss_val in zip(self.loss_coef, args):
            total_loss += coef * loss_val

        return total_loss
