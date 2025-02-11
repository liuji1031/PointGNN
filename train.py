import sys

sys.path.append(".")
import pathlib
from datetime import datetime
from pprint import pformat

import numpy as np
import torch
import torch._dynamo
import typer
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from dataset import DatasetRegistry
from model import ComposableModel
from train import Optimization, loop_through_dataset
from util import read_config

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def train(
    data_cfg: pathlib.Path = typer.Option(
        ...,
        "--data-cfg",
        "-d",
        help="Path to the dataset configuration file",
        readable=True,
        exists=True,
        resolve_path=True,
    ),
    model_cfg: pathlib.Path = typer.Option(
        ...,
        "--model-cfg",
        "-m",
        help="Path to the model configuration file",
        readable=True,
        exists=True,
        resolve_path=True,
    ),
    train_cfg: pathlib.Path = typer.Option(
        ...,
        "--train-cfg",
        "-t",
        help="Path to the training configuration file",
        readable=True,
        exists=True,
        resolve_path=True,
    ),
    chkpt: pathlib.Path = typer.Option(
        None,
        "--chkpt",
        "-c",
        help="Path to the checkpoint file",
        dir_okay=False,
        readable=True,
        exists=True,
        resolve_path=True,
    ),
    summary: pathlib.Path = typer.Option(
        None,
        "--summary",
        "-s",
        help="Path to the summary file",
        writable=True,
        resolve_path=True,
    ),
    log_lvl: str = typer.Option(
        "INFO",
        "--log-lvl",
        "-l",
        help="Logging level",
        show_default=True,
        case_sensitive=False,
    ),
):
    """Train the model."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger.level(log_lvl)
    torch.set_float32_matmul_precision("high")

    # load the dataset configuration
    logger.info(f"Loading dataset configuration...")
    data_cfg = read_config(data_cfg)
    logger.debug(f"Configuration file:\n {pformat(data_cfg)}")

    # load model configuration
    logger.info(f"Loading model configuration...")
    model_cfg = read_config(model_cfg)
    logger.debug(f"Configuration file:\n {pformat(model_cfg)}")

    # load optimization configuration
    logger.info(f"Loading training configuration...")
    train_cfg = read_config(train_cfg)
    logger.debug(f"Configuration file:\n {pformat(train_cfg)}")

    # create the dataset and loader
    ds_train = DatasetRegistry.build(
        mode="train",
        name=data_cfg.name,
        config=data_cfg.config,
    )
    logger.info(f"Dataset {ds_train} loaded.")

    ds_val = DatasetRegistry.build(
        mode="val",
        name=data_cfg.name,
        config=data_cfg.config,
    )

    ds_test = DatasetRegistry.build(
        mode="test",
        name=data_cfg.name,
        config=data_cfg.config,
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=train_cfg.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=train_cfg.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=train_cfg.batch_size,
        shuffle=False,
    )

    # load the model
    model = ComposableModel(model_cfg.model.name, model_cfg.model.modules)
    model = model.to(ds_train.device)
    loss = ComposableModel(model_cfg.loss.name, model_cfg.loss.modules)
    loss = loss.to(ds_train.device)
    logger.info(f"Model\n {model} loaded.")

    if train_cfg.torch_dynamo:
        model = torch._dynamo.optimize("inductor")(model)
        loss = torch._dynamo.optimize("inductor")(loss)

    # load the optimizer and scheduler
    optimizer = Optimization.optimizer(
        params=model.parameters(),
        name=train_cfg.optimizer.name,
        config=train_cfg.optimizer.config,
    )
    scheduler = Optimization.scheduler(
        optimizer=optimizer,
        name=train_cfg.scheduler.name,
        config=train_cfg.scheduler.config,
    )
    logger.info(f"Optimizer and scheduler loaded.")

    # load the checkpoint if available
    start_epoch = 0
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
        logger.info(
            f"Checkpoint {chkpt} loaded. Starting from epoch {start_epoch}"
        )
        # get parent dir of chkpt
        chkpt = chkpt.parent
    else:
        # create checkpoint directory
        chkpt = pathlib.Path(train_cfg.logging_dir) / timestamp / "chkpt"
        chkpt.mkdir(parents=True, exist_ok=True)

    # create summary directory
    if summary:
        writer = SummaryWriter(summary)
    else:
        summary = pathlib.Path(train_cfg.logging_dir) / timestamp / "summary"
        summary.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(summary)

    logger.info(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
    logger.info(
        f"Current regularization strength: {optimizer.param_groups[0]['weight_decay']}"
    )

    # training loop
    best_val_loss = np.inf
    for epoch in range(start_epoch, train_cfg.epochs):
        result_train = loop_through_dataset(
            epoch,
            train_loader,
            model,
            loss,
            optimizer,
            scheduler,
            mode="train",
            result_unpack_sequence=model_cfg.loss.unpack_sequence,
            device=ds_train.device,
            target_loss_name=train_cfg.target_loss_name,
        )

        result_val = loop_through_dataset(
            epoch,
            val_loader,
            model,
            loss,
            optimizer,
            scheduler,
            mode="val",
            result_unpack_sequence=model_cfg.loss.unpack_sequence,
            device=ds_val.device,
            target_loss_name=train_cfg.target_loss_name,
        )

        # update summary writer
        writer.add_scalars("train", result_train, epoch)
        writer.add_scalars("val", result_val, epoch)
        writer.add_scalar(
            "learning_rate", optimizer.param_groups[0]["lr"], epoch
        )
        writer.flush()

        # save checkpoint
        chkpt_state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        }
        curr_val_loss = result_val[train_cfg.target_loss_name]
        if curr_val_loss < best_val_loss:
            best_val_loss = curr_val_loss
            logger.info(f"Saving best model at epoch {epoch}")
            torch.save(chkpt_state, chkpt / "best_model.pt")

        torch.save(chkpt_state, chkpt / "last_model.pt")


if __name__ == "__main__":
    app()
