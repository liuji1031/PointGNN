import yaml

def read_config(path : str) -> dict:
    """read a yaml file and return a dictionary

    Args:
        path (str): path to the yaml file

    Returns:
        dict: dictionary of the yaml file
    """
    with open(path, 'r') as file:
        return yaml.safe_load(file)