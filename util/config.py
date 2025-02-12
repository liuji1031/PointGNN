import re
import yaml
import pathlib
from easydict import EasyDict as edict
from model import ComposableModel
from dataset import DatasetRegistry


def read_config(path: str | pathlib.Path) -> edict:
    """read a yaml file and return a dictionary

    Args:
        path (str): path to the yaml file

    Returns:
        dict: dictionary of the yaml file
    """
    with open(path, "r") as file:
        return edict(yaml.safe_load(file))


def camel_to_snake(name):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def build_model(config_file: str):
    """Build full model based on config file.

    Args:
        config_file (dic
    Returns:
        torch.nn.Module: the model
    """
    config = read_config(config_file)
    assert "model" in config, "Model configuration not found in the config file"
    model = ComposableModel(
        config.model.get("name", "model"), config.model.modules
    )

    return model


def build_dataset(config_file: str, mode="train"):
    data_cfg = read_config(config_file)
    return DatasetRegistry.build(
        mode=mode,
        name=data_cfg.name,
        config=data_cfg.config,
    )
