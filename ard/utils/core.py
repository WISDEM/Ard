import copy
from os import PathLike
from pathlib import Path
import yaml
import jax.numpy as jnp
import jax

import numpy as np

from wisdem.inputs.validation import load_yaml


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


def create_FLORIS_turbine(
    input_turbine_spec: dict | PathLike,
    filename_turbine_FLORIS: PathLike = None,
) -> dict:
    """
    Create a FLORIS turbine from a generic Ard turbine specification.

    Parameters
    ----------
    input_turbine_spec : dict | PathLike
        a turbine specification from which to extract a FLORIS turbine
    filename_turbine_FLORIS : PathLike, optional
        a path to save a FLORIS turbine configuration yaml file, optionally

    Returns
    -------
    dict
        a FLORIS turbine configuration in dictionary form

    Raises
    ------
    TypeError
        if the turbine specification input is not the correct type
    """

    if isinstance(input_turbine_spec, PathLike):
        with open(input_turbine_spec, "r") as file_turbine_spec:
            turbine_spec = load_turbine_spec(file_turbine_spec)
    elif type(input_turbine_spec) == dict:
        turbine_spec = input_turbine_spec
    else:
        raise TypeError(
            "create_FLORIS_yamlfile requires either a dict input or a filename input.\n"
            + f"received a {type(input_turbine_spec)}"
        )

    # load speed/power/thrust file
    filename_power_thrust = turbine_spec["performance_data_ccblade"]["power_thrust_csv"]
    pt_raw = np.genfromtxt(filename_power_thrust, delimiter=",").T.tolist()

    # create FLORIS config dict
    turbine_FLORIS = dict()
    turbine_FLORIS["turbine_type"] = turbine_spec["description"]["name"]
    turbine_FLORIS["hub_height"] = turbine_spec["geometry"]["height_hub"]
    turbine_FLORIS["rotor_diameter"] = turbine_spec["geometry"]["diameter_rotor"]
    turbine_FLORIS["TSR"] = turbine_spec["nameplate"]["TSR"]
    # turbine_FLORIS["multi_dimensional_cp_ct"] = True
    # turbine_FLORIS["power_thrust_data_file"] = filename_power_thrust
    turbine_FLORIS["power_thrust_table"] = {
        "cosine_loss_exponent_yaw": turbine_spec["model_specifications"]["FLORIS"][
            "exponent_penalty_yaw"
        ],
        "cosine_loss_exponent_tilt": turbine_spec["model_specifications"]["FLORIS"][
            "exponent_penalty_tilt"
        ],
        "peak_shaving_fraction": turbine_spec["model_specifications"]["FLORIS"][
            "fraction_peak_shaving"
        ],
        "peak_shaving_TI_threshold": 0.0,
        "ref_air_density": turbine_spec["performance_data_ccblade"][
            "density_ref_cp_ct"
        ],
        "ref_tilt": turbine_spec["performance_data_ccblade"]["tilt_ref_cp_ct"],
        "wind_speed": pt_raw[0],
        "power": (
            0.5
            * turbine_spec["performance_data_ccblade"]["density_ref_cp_ct"]
            * (np.pi / 4.0 * turbine_spec["geometry"]["diameter_rotor"] ** 2)
            * np.array(pt_raw[0]) ** 3
            * pt_raw[1]
            / 1e3
        ).tolist(),
        "thrust_coefficient": pt_raw[2],
    }

    # If an export filename is given, write it out
    if filename_turbine_FLORIS is not None:
        with open(filename_turbine_FLORIS, "w") as file_turbine_FLORIS:
            yaml.safe_dump(turbine_FLORIS, file_turbine_FLORIS)

    return copy.deepcopy(turbine_FLORIS)
