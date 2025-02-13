import numpy as np
import torch
from loguru import logger
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import time

from model import ComposableModel, unpack_result

class Optimization:
    OPTIMIZERS = {"adam": Adam}
    SCHEDULERS = {"step": StepLR, "plateau": ReduceLROnPlateau}

    @classmethod
    def optimizer(cls, params, name: str, config: dict):
        assert name.lower() in cls.OPTIMIZERS, f"Unknown optimizer {name}"
        return cls.OPTIMIZERS[name.lower()](params=params, **config)

    @classmethod
    def scheduler(cls, optimizer, name: str, config: dict):
        assert name.lower() in cls.SCHEDULERS, f"Unknown scheduler {name}"
        return cls.SCHEDULERS[name.lower()](optimizer, **config)


def _to_scalar(t: torch.Tensor):
    return t.cpu().detach().numpy().item()


def loop_through_dataset(
    epoch: int,
    ds_loader: DataLoader,
    model: ComposableModel,
    loss: ComposableModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    mode: str,
    result_unpack_sequence: list,
    device: str,
    target_loss_name="total",
    raise_oom=False,
):
    pbar = tqdm(
        iter(ds_loader),
        total=len(ds_loader),
        desc=f"Epoch {epoch}, {mode}",
        ncols=150,
    )

    result = {}

    if mode == "train":
        model.train()
    else:
        model.eval()

    for idata, data in enumerate(pbar):

        data: Data

        if mode == "train":
            optimizer.zero_grad()

        pred = model(data.x, data.pos, data.edge_index)
        loss_out = loss(
            *unpack_result(pred, data.target, result_unpack_sequence, data.mask)
        )
        target_loss = loss_out[target_loss_name]

        if mode == "train":
            # try:
            #     target_loss.backward()
            # except RuntimeError as e:  # handle out of memory error
            #     if "out of memory" in str(e) and not raise_oom:
            #         logger.info("| WARNING: ran out of memory, skipping batch")
            #         for p in model.parameters():
            #             if p.grad is not None:
            #                 del p.grad  # free some memory
            #         torch.cuda.empty_cache()
            #     else:
            #         raise e
            target_loss.backward()
            optimizer.step()

            # print the mean gradient of all parameters
            # for m in model.named_modules():
            #     logger.info(f"{m[0]},{m[1].__class__.__name__}")
            #     for p in m[1].parameters():
            #         if p.grad is not None:
            #             logger.info(f"mean grad {p.grad.mean():.6f}")

        # append the loss values to the result dictionary
        post_fix = {}
        for name, val in loss_out.items():
            if name not in result:
                result[name] = np.zeros(len(ds_loader))
            result[name][idata] = _to_scalar(val)
            post_fix[name] = f"{result[name][np.maximum(0, idata-19):idata+1].mean():.2f}"
        pbar.set_postfix(post_fix)

    # average the loss values
    for name, val in result.items():
        result[name] = np.mean(val)

    if mode == "val":
        scheduler.step(result[target_loss_name])

    return result
