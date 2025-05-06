from pathlib import Path

import numpy as np
import openmdao.api as om

import floris

import ard.utils.io
import ard.utils.test_utils
import ard.wind_query as wq
import ard.farm_aero.floris as farmaero_floris


class TestFLORISFarmComponent:

    def setup_method(self):
        pass


class TestFLORISBatchPower:

    def setup_method(self):

        # create the wind query
        directions = np.linspace(0.0, 360.0, 21)
        speeds = np.linspace(0.0, 30.0, 21)[1:]
        WS, WD = np.meshgrid(speeds, directions)
        wind_query = wq.WindQuery(WD.flatten(), WS.flatten())
        wind_query.set_TI_using_IEC_method()

        # create the farm layout specification
        farm_spec = {}
        farm_spec["xD_farm"], farm_spec["yD_farm"] = [
            7 * v.flatten()
            for v in np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
        ]

        # specify the configuration/specification files to use
        filename_turbine_spec = Path(
            Path(ard.__file__).parents[1],
            "examples",
            "data",
            "turbine_spec_IEA-3p4-130-RWT.yaml",
        ).absolute()  # toolset generalized turbine specification
        data_turbine_spec = ard.utils.io.load_turbine_spec(filename_turbine_spec)

        # set up the modeling options
        modeling_options = {
            "farm": {
                "N_turbines": len(farm_spec["xD_farm"]),
            },
            "turbine": data_turbine_spec,
        }

        # create the OpenMDAO model
        model = om.Group()
        self.FLORIS = model.add_subsystem(
            "batchFLORIS",
            farmaero_floris.FLORISBatchPower(
                modeling_options=modeling_options,
                wind_query=wind_query,
                case_title="letsgo",
            ),
        )

        self.prob = om.Problem(model)
        self.prob.setup()

    def test_setup(self):
        "make sure the modeling_options has what we need for farmaero"

        assert "case_title" in [k for k, _ in self.FLORIS.options.items()]
        assert "modeling_options" in [k for k, _ in self.FLORIS.options.items()]

        assert "farm" in self.FLORIS.options["modeling_options"].keys()
        assert "N_turbines" in self.FLORIS.options["modeling_options"]["farm"].keys()

        # make sure that the inputs in the component match what we planned
        input_list = [k for k, v in self.FLORIS.list_inputs(val=False)]
        for var_to_check in [
            "x_turbines",
            "y_turbines",
            "yaw_turbines",
        ]:
            assert var_to_check in input_list

        # make sure that the outputs in the component match what we planned
        output_list = [k for k, v in self.FLORIS.list_outputs(val=False)]
        for var_to_check in [
            "power_farm",
            "power_turbines",
            "thrust_turbines",
        ]:
            assert var_to_check in output_list

    def test_compute_pyrite(self):

        x_turbines = 7.0 * 130.0 * np.arange(-2, 2.1, 1)
        y_turbines = 7.0 * 130.0 * np.arange(-2, 2.1, 1)
        X, Y = [v.flatten() for v in np.meshgrid(x_turbines, y_turbines)]
        yaw_turbines = np.zeros_like(X)
        self.prob.set_val("batchFLORIS.x_turbines", X)
        self.prob.set_val("batchFLORIS.y_turbines", Y)
        self.prob.set_val("batchFLORIS.yaw_turbines", yaw_turbines)

        self.prob.run_model()

        # collect data to validate
        validation_data = {
            "power_farm": self.prob.get_val("batchFLORIS.power_farm", units="MW"),
            "power_turbines": self.prob.get_val(
                "batchFLORIS.power_turbines", units="MW"
            ),
            "thrust_turbines": self.prob.get_val(
                "batchFLORIS.thrust_turbines", units="kN"
            ),
        }
        # validate data against pyrite file
        ard.utils.test_utils.pyrite_validator(
            validation_data,
            Path(__file__).parent / "test_floris_batch_pyrite.npz",
            rtol_val=5e-3,
            # rewrite=True,  # uncomment to write new pyrite file
        )


class TestFLORISAEP:

    def setup_method(self):

        # create the wind query
        directions = np.linspace(0.0, 360.0, 21)
        speeds = np.linspace(0.0, 30.0, 21)[1:]
        wind_rose = floris.WindRose(
            wind_directions=directions,
            wind_speeds=speeds,
            ti_table=0.06,
        )

        # create the farm layout specification
        farm_spec = {}
        farm_spec["xD_farm"], farm_spec["yD_farm"] = [
            7 * v.flatten()
            for v in np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
        ]

        # specify the configuration/specification files to use
        filename_turbine_spec = (
            Path(ard.__file__).parents[1]
            / "examples"
            / "data"
            / "turbine_spec_IEA-3p4-130-RWT.yaml"
        )  # toolset generalized turbine specification
        data_turbine_spec = ard.utils.io.load_turbine_spec(filename_turbine_spec)

        # set up the modeling options
        modeling_options = {
            "farm": {
                "N_turbines": len(farm_spec["xD_farm"]),
            },
            "turbine": data_turbine_spec,
        }

        # create the OpenMDAO model
        model = om.Group()
        self.FLORIS = model.add_subsystem(
            "aepFLORIS",
            farmaero_floris.FLORISAEP(
                modeling_options=modeling_options,
                wind_rose=wind_rose,
                case_title="letsgo",
            ),
        )

        self.prob = om.Problem(model)
        self.prob.setup()

    def test_setup(self):
        "make sure the modeling_options has what we need for farmaero"
        assert "case_title" in [k for k, _ in self.FLORIS.options.items()]
        assert "modeling_options" in [k for k, _ in self.FLORIS.options.items()]

        assert "farm" in self.FLORIS.options["modeling_options"].keys()
        assert "N_turbines" in self.FLORIS.options["modeling_options"]["farm"].keys()

        # make sure that the inputs in the component match what we planned
        input_list = [k for k, v in self.FLORIS.list_inputs(val=False)]
        for var_to_check in [
            "x_turbines",
            "y_turbines",
            "yaw_turbines",
        ]:
            assert var_to_check in input_list

        # make sure that the outputs in the component match what we planned
        output_list = [k for k, v in self.FLORIS.list_outputs(val=False)]
        for var_to_check in [
            "AEP_farm",
            "power_farm",
            "power_turbines",
            "thrust_turbines",
        ]:
            assert var_to_check in output_list

    def test_compute_pyrite(self):

        x_turbines = 7.0 * 130.0 * np.arange(-2, 2.1, 1)
        y_turbines = 7.0 * 130.0 * np.arange(-2, 2.1, 1)
        X, Y = [v.flatten() for v in np.meshgrid(x_turbines, y_turbines)]
        yaw_turbines = np.zeros_like(X)
        self.prob.set_val("aepFLORIS.x_turbines", X)
        self.prob.set_val("aepFLORIS.y_turbines", Y)
        self.prob.set_val("aepFLORIS.yaw_turbines", yaw_turbines)

        self.prob.run_model()

        # collect data to validate
        test_data = {
            "aep_farm": self.prob.get_val("aepFLORIS.AEP_farm", units="GW*h"),
            "power_farm": self.prob.get_val("aepFLORIS.power_farm", units="MW"),
            "power_turbines": self.prob.get_val("aepFLORIS.power_turbines", units="MW"),
            "thrust_turbines": self.prob.get_val(
                "aepFLORIS.thrust_turbines", units="kN"
            ),
        }
        # validate data against pyrite file
        ard.utils.test_utils.pyrite_validator(
            test_data,
            Path(__file__).parent / "test_floris_aep_pyrite.npz",
            rtol_val=5e-3,
            # rewrite=True,  # uncomment to write new pyrite file
        )
