from pathlib import Path
import pytest

# import platform


import numpy as np
import matplotlib.pyplot as plt

import floris
import openmdao.api as om

from wisdem.optimization_drivers.nlopt_driver import NLoptDriver

import ard
import ard.utils.test_utils
import ard.utils.io
import ard.wind_query as wq
import ard.layout.sunflower as sunflower
import ard.farm_aero.floris as farmaero_floris
import ard.collection.optiwindnet_wrap as inter

from optiwindnet.plotting import gplot  # DEBUG!!!!! REMOVE ME


@pytest.mark.usefixtures("subtests")
class TestoptiwindnetLayout:

    def setup_method(self):

        # create the wind query
        wind_rose_wrg = floris.wind_data.WindRoseWRG(
            Path(ard.__file__).parents[1] / "examples" / "data" / "wrg_example.wrg"
        )
        wind_rose_wrg.set_wd_step(90.0)
        wind_rose_wrg.set_wind_speeds(
            np.array([5.0, 10.0, 15.0, 20.0], dtype=np.float64)
        )
        wind_rose = wind_rose_wrg.get_wind_rose_at_point(0.0, 0.0)
        wind_query = wq.WindQuery.from_FLORIS_WindData(wind_rose)

        # specify the configuration/specification files to use
        filename_turbine_spec = (
            Path(ard.__file__).parents[1]
            / "examples"
            / "data"
            / "turbine_spec_IEA-3p4-130-RWT.yaml"
        )  # toolset generalized turbine specification
        data_turbine_spec = ard.utils.io.load_turbine_spec(filename_turbine_spec)

        # set up the modeling options
        self.modeling_options = {
            "farm": {
                "N_turbines": 25,
                "N_substations": 1,
            },
            "turbine": data_turbine_spec,
            "offshore": False,
            "collection": {
                "max_turbines_per_string": 8,
                "solver_name": "appsi_highs",
                "solver_options": dict(
                    time_limit=60,
                    mip_rel_gap=0.005,  # TODO ???
                ),
            },
        }

        # create the OpenMDAO model
        self.model = om.Group()
        self.model.add_subsystem(  # layout component
            "layout",
            sunflower.SunflowerFarmLayout(modeling_options=self.modeling_options),
            promotes=["*"],
        )
        self.model.add_subsystem(  # landuse component
            "landuse",
            sunflower.SunflowerFarmLanduse(modeling_options=self.modeling_options),
            promotes_inputs=["*"],
        )
        self.model.add_subsystem(  # FLORIS AEP component
            "aepFLORIS",
            farmaero_floris.FLORISAEP(
                modeling_options=self.modeling_options,
                wind_rose=wind_rose,
                case_title="letsgo",
            ),
            # promotes=["AEP_farm"],
            promotes=["x_turbines", "y_turbines", "AEP_farm"],
        )
        self.model.add_subsystem(
            "optiwindnet_coll",
            inter.optiwindnetCollection(modeling_options=self.modeling_options),
            promotes=["x_turbines", "y_turbines"],
        )

        self.prob = om.Problem(self.model)
        self.prob.setup()

    def test_model(self, subtests):

        # set up the working/design variables
        self.prob.set_val("spacing_target", 7.0)
        self.prob.set_val("optiwindnet_coll.x_substations", [0.0])
        self.prob.set_val("optiwindnet_coll.y_substations", [0.0])

        # run the model
        self.prob.run_model()

        # approximated circle area should be close to the sunflower area
        area_circle = (
            np.pi
            / 4
            * np.ptp(self.prob.get_val("x_turbines", units="km"))
            * np.ptp(self.prob.get_val("y_turbines", units="km"))
        )

        with subtests.test("land use area"):
            assert np.isclose(
                self.prob.get_val("landuse.area_tight"), area_circle, rtol=0.1
            )

        # collect optiwindnet data to validate
        validation_data = {
            "length_cables": self.prob.get_val("optiwindnet_coll.length_cables")
            / 1.0e3,
            "load_cables": self.prob.get_val("optiwindnet_coll.load_cables"),
            "total_length_cables": self.prob.get_val(
                "optiwindnet_coll.total_length_cables"
            ),
            "max_load_cables": self.prob.get_val("optiwindnet_coll.max_load_cables"),
        }

        with subtests.test("pyrite validator"):
            ard.utils.test_utils.pyrite_validator(
                validation_data,
                Path(__file__).parent / "test_optiwindnet_pyrite.npz",
                rtol_val=5e-3,
                # rewrite=True,  # uncomment to write new pyrite file
            )

        # os_name = platform.system()

        # if os_name == 'Linux':
        #     # Run Linux specific tests
        #     # validate data against pyrite file

        # elif os_name == 'Darwin':
        #     # Run macos specific tests
        #     # validate data against pyrite file
        #     with subtests.test("pyrite validator"):
        #         ard.test_utils.pyrite_validator(
        #             validation_data,
        #             Path(__file__).parent / "test_optiwindnet_pyrite_macos.npz",
        #             rtol_val=5e-3,
        #             # rewrite=True,  # uncomment to write new pyrite file
        #         )
        # elif os_name == "Windows":
        #     # Run Windows specific tests
        #     # validate data against pyrite file
        #     with subtests.test("pyrite validator"):
        #         ard.test_utils.pyrite_validator(
        #             validation_data,
        #             Path(__file__).parent / "test_optiwindnet_pyrite_macos.npz",
        #             rtol_val=5e-3,
        #             # rewrite=True,  # uncomment to write new pyrite file
        #         )
        # else:
        #     raise(ValueError("Invalid OS for pyrite validation test"))
