from pathlib import Path
from os import PathLike

import os
import yaml


class Loader(yaml.SafeLoader):

    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]

        super().__init__(stream)

    def include(self, node):

        filename = os.path.join(self._root, self.construct_scalar(node))

        with open(filename, "r") as f:
            return yaml.load(f, self.__class__)


Loader.add_constructor("!include", Loader.include)


def load_yaml(filename, loader=Loader, return_path=False) -> dict:
    if isinstance(filename, dict):
        return filename  # filename already yaml dict
    with open(filename) as fid:
        if return_path:
            return yaml.load(fid, loader), Path(filename).parent.absolute()
        else:
            return yaml.load(fid, loader)


def check_create_folder(filepath):
    already_exists = True
    if not os.path.isdir(filepath):
        os.makedirs(filepath, exist_ok=True)
        already_exists = False
    return already_exists


def write_yaml(filename, data):
    if not ".yaml" in filename:
        filename = filename + ".yaml"

    with open(filename, "w+") as file:
        yaml.dump(data, file, sort_keys=False, encoding=None, default_flow_style=False)


def load_turbine_spec(
    filename_turbine_spec: PathLike,
):
    """
    Utility to load turbine spec. files, & transform relative to absolute links.

    Parameters
    ----------
    filename_turbine_spec : Pathlike
        filename of a turbine specification to load

    Returns
    -------
    dict
        a turbine specification dictionary
    """

    filename_turbine_spec = Path(filename_turbine_spec)
    dir_turbine_spec = filename_turbine_spec.parent
    turbine_spec = load_yaml(filename_turbine_spec)
    filename_powercurve = (
        dir_turbine_spec / turbine_spec["performance_data_ccblade"]["power_thrust_csv"]
    )
    turbine_spec["performance_data_ccblade"]["power_thrust_csv"] = filename_powercurve

    return turbine_spec


def replace_key_value(
    target_dict: dict, target_key: str, new_value, replace_none_only=True
) -> dict:
    """
    Recursively replace the value of a target key in a dictionary.

    Parameters:
    - target_dict (dict): The dictionary to process.
    - target_key (str): The key whose value needs to be replaced.
    - new_value: The new value to assign to the target key.
    - replace_none_only (bool): if True, only 'None' values will be replaced.

    Returns:
    - dict: The updated dictionary.
    """
    for key, value in target_dict.items():
        if key == target_key:
            if (value != None) and (replace_none_only):
                continue
            target_dict[key] = new_value
        elif isinstance(value, dict):
            # Recurse into nested dictionaries
            replace_key_value(
                value, target_key, new_value, replace_none_only=replace_none_only
            )
        elif isinstance(value, list):
            # Handle lists of dictionaries
            for item in value:
                if isinstance(item, dict):
                    replace_key_value(
                        item, target_key, new_value, replace_none_only=replace_none_only
                    )
    return target_dict
