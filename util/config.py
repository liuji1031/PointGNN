import re
import yaml
import pathlib
from easydict import EasyDict as edict

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
