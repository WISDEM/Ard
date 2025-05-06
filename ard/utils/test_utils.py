from fileinput import filename
from os import PathLike
from pathlib import Path

import numpy as np


def pyrite_validator(
    validation_data: dict,
    filename_pyrite: PathLike,
    rewrite: bool = False,
    rtol_val: float = 1e-6,
):
    """
    TO DO!!!
    """

    # get the basename if a suffix is provided
    filename_pyrite = Path(filename_pyrite).with_suffix("")

    if rewrite:
        # this helper function can write a file to hold pyrite-standard data

        # write out a npz file that holds the variables we want to be able to check
        np.savez(filename_pyrite, **validation_data)
        assert False
    else:
        # ... or it can check to make sure that an existing pyrite file matches the current data

        # load an existing pyrite-standard data file
        pyrite_data = np.load(filename_pyrite.with_suffix(".npz"))

        # for each of the variables in the pyrite-standard data file
        for k, v in pyrite_data.items():
            # count how many of the values in the data match the equivalent validation data
            sum_isclose = np.sum(
                np.isclose(np.array(v), validation_data[k], rtol=rtol_val)
            )
            vd_size = np.array(validation_data[k], dtype=np.float64).size
            # assert all of the values match
            validation_matches = sum_isclose == vd_size

            if not validation_matches:
                print(f"for variable {k}:")
                print(
                    f"\t{sum_isclose} values match of {vd_size} total validation values"
                )
                print(f"\tto a tolerance of {rtol_val:e}")
                print(f"saved data for {k}: {v}")
                print(f"computed data for {k}: {validation_data[k]}")
            assert validation_matches
