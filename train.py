import sys

sys.path.append(".")
import pathlib
from pprint import pformat
from typing import Annotated

import torch
import typer
from easydict import EasyDict as edict
from loguru import logger

from dataset import DatasetRegistry
from model import ComposableModel, Loss
from train import Optimization
from util import read_config

app = typer.Typer()


@app.command()
def train(
    config: pathlib.Path = Annotated[
        pathlib.Path,
        typer.Option(
            ...,
            help="Path to the dataset configuration file",
            readable=True,
            exists=True,
            resolve_path=True,
        ),
    ],
    chkpt: pathlib.Path = Annotated[
        pathlib.Path,
        typer.Option(
            None,
            help="Path to the checkpoint file",
            readable=True,
            exists=True,
            resolve_path=True,
        ),
    ],
):
    """Train the model."""

    # load the dataset
    logger.debug(f"Loading dataset...")
    config = read_config(config)
    config: edict
    logger.info(f"Configuration file\n {pformat(config)} loaded.")
    ds_train = DatasetRegistry.build(
        mode="train",
        name=config.Dataset.name,
        config=config.Dataset.config,
    )
    logger.info(f"Dataset {ds_train} loaded.")

    ds_val = DatasetRegistry.build(
        mode="val",
        name=config.Dataset.name,
        config=config.Dataset.config,
    )

    ds_test = DatasetRegistry.build(
        mode="test",
        name=config.Dataset.name,
        config=config.Dataset.config,
    )

    # load the model
    logger.debug(f"Loading model...")
    model = ComposableModel(config.ModelName, config.Modules)
    logger.info(f"Model\n {model} loaded.")

    # load training configuration
    loss = Loss(**config.Loss.config)

    # load optimization configuration
    optimizer = Optimization.optimizer(
        params=model.parameters(),
        name=config.Optimizer.name,
        config=config.Optimizer.config,
    )
    scheduler = Optimization.scheduler(
        optimizer=optimizer,
        name=config.Scheduler.name,
        config=config.Scheduler.config,
    )

    # load the checkpoint
    if chkpt:
        logger.info(f"Loading checkpoint from {chkpt}")
        chkpt_state = torch.load(chkpt)
        missing_key, unexpected_key = model.load_state_dict(
            chkpt_state["model"], strict=True
        )

        for key in missing_key:
            logger.warning(f"Missing key: {key}")

        for key in unexpected_key:
            logger.warning(f"Unexpected key: {key}")

        optimizer.load_state_dict(chkpt_state["optimizer"])
        scheduler.load_state_dict(chkpt_state["scheduler"])

        start_epoch = chkpt_state["epoch"] + 1
        logger.info(f"Checkpoint loaded. Starting from epoch {start_epoch}")

    logger.info(f"current learning rate: {optimizer.param_groups[0]['lr']}")
    logger.info(
        f"current regularization strength: {optimizer.param_groups[0]['weight_decay']}"
    )
    

if __name__ == "__main__":
    app()
