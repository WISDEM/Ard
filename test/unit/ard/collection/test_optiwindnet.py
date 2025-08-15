import copy
from pathlib import Path
import platform, sys

import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

import pytest

optiwindnet = pytest.importorskip("optiwindnet")

from optiwindnet.plotting import gplot

import ard.utils.io
import ard.utils.test_utils
import ard.collection.optiwindnet_wrap as ard_own


def make_modeling_options(x_turbines, y_turbines, x_substations, y_substations):

    # specify the configuration/specification files to use
    filename_turbine_spec = (
        Path(ard.__file__).parents[1]
        / "examples"
        / "data"
        / "turbine_spec_IEA-3p4-130-RWT.yaml"
    )  # toolset generalized turbine specification
    data_turbine_spec = ard.utils.io.load_turbine_spec(filename_turbine_spec)

    # set up the modeling options
    N_turbines = len(x_turbines)
    N_substations = len(x_substations)
    modeling_options = {
        "layout": {
            "N_turbines": N_turbines,
            "N_substations": N_substations,
            "x_substations": x_substations,
            "y_substations": y_substations,
            "x_turbines": x_turbines,
            "y_turbines": y_turbines,
        },
        "turbine": data_turbine_spec,
        "collection": {
            "max_turbines_per_string": 8,
            "model_options": dict(
                topology="branched",
                feeder_route="segmented",
                feeder_limit="unlimited",
            ),
            "solver_name": "highs",
            "solver_options": dict(
                time_limit=10,
                mip_gap=0.005,  # TODO ???
            ),
        },
    }

    return modeling_options


@pytest.mark.usefixtures("subtests")
class TestOptiWindNetCollection:

    def setup_method(self):

        # create the farm layout specification
        n_turbines = 25
        x_turbines, y_turbines = [
            130.0 * 7 * v.flatten()
            for v in np.meshgrid(
                np.linspace(-2, 2, int(np.sqrt(n_turbines)), dtype=int),
                np.linspace(-2, 2, int(np.sqrt(n_turbines)), dtype=int),
            )
        ]
        x_substations = np.array([-500.0, 500.0], dtype=np.float64)
        y_substations = np.array([-500.0, 500.0], dtype=np.float64)

        modeling_options = make_modeling_options(
            x_turbines=x_turbines,
            y_turbines=y_turbines,
            x_substations=x_substations,
            y_substations=y_substations,
        )

        # create the OpenMDAO model
        model = om.Group()
        self.optiwindnet_coll = model.add_subsystem(
            "optiwindnet_coll",
            ard_own.OptiwindnetCollection(
                modeling_options=modeling_options,
            ),
        )

        self.prob = om.Problem(model)
        self.prob.setup()

    def test_modeling(self, subtests):
        """
        make sure the modeling_options has what we need for farmaero
        """

        with subtests.test("modeling_options"):
            assert "modeling_options" in [
                k for k, _ in self.optiwindnet_coll.options.items()
            ]
        with subtests.test("layout"):
            assert "layout" in self.optiwindnet_coll.options["modeling_options"].keys()
        with subtests.test("N_turbines"):
            assert (
                "N_turbines"
                in self.optiwindnet_coll.options["modeling_options"]["layout"].keys()
            )
        with subtests.test("N_substations"):
            assert (
                "N_substations"
                in self.optiwindnet_coll.options["modeling_options"]["layout"].keys()
            )

        # context manager to spike the warning since we aren't running the model yet
        with pytest.warns(Warning) as warning:
            # make sure that the inputs in the component match what we planned
            input_list = [k for k, v in self.optiwindnet_coll.list_inputs()]
            for var_to_check in [
                "x_turbines",
                "y_turbines",
                "x_substations",
                "y_substations",
            ]:
                with subtests.test("inputs"):
                    assert var_to_check in input_list

            # make sure that the outputs in the component match what we planned
            output_list = [k for k, v in self.optiwindnet_coll.list_outputs()]
            for var_to_check in [
                "total_length_cables",
            ]:
                assert var_to_check in output_list

            # make sure that the outputs in the component match what we planned
            discrete_output_list = [
                k for k, v in self.optiwindnet_coll._discrete_outputs.items()
            ]
            for var_to_check in [
                "length_cables",
                "load_cables",
                "max_load_cables",
                "terse_links",
            ]:
                assert var_to_check in discrete_output_list

    def test_compute_pyrite(self, subtests):

        # run optiwindnet
        self.prob.run_model()

        # collect data to validate
        validation_data = {
            "terse_links": self.prob.get_val("optiwindnet_coll.terse_links"),
            "length_cables": self.prob.get_val("optiwindnet_coll.length_cables"),
            "load_cables": self.prob.get_val("optiwindnet_coll.load_cables"),
            "total_length_cables": self.prob.get_val(
                "optiwindnet_coll.total_length_cables"
            ),
            "max_load_cables": self.prob.get_val("optiwindnet_coll.max_load_cables"),
        }

        # validate data against pyrite file
        pyrite_data = ard.utils.test_utils.pyrite_validator(
            validation_data,
            Path(__file__).parent / "test_optiwindnet_pyrite.npz",
            # rtol_val=5e-3, # only for check in validator
            #  rewrite=True,  # uncomment to write new pyrite file
            load_only=True,
        )

        for key in validation_data:
            with subtests.test(key):
                assert np.allclose(validation_data[key], pyrite_data[key], rtol=5e-3)


class TestOptiWindNetCollection12Turbines:

    def setup_method(self):

        x_turbines = np.array(
            [1940, 1920, 1475, 1839, 1277, 442, 737, 1060, 522, 87, 184, 71],
            dtype=np.float64,
        )
        y_turbines = np.array(
            [279, 703, 696, 1250, 1296, 1359, 435, 26, 176, 35, 417, 878],
            dtype=np.float64,
        )
        x_substations = np.array([696], dtype=np.float64)
        y_substations = np.array([1063], dtype=np.float64)

        self.modeling_options = make_modeling_options(
            x_turbines=x_turbines,
            y_turbines=y_turbines,
            x_substations=x_substations,
            y_substations=y_substations,
        )

        # create the OpenMDAO model
        model = om.Group()
        self.optiwindnet_coll = model.add_subsystem(
            "optiwindnet_coll",
            ard_own.OptiwindnetCollection(
                modeling_options=self.modeling_options,
            ),
        )

        self.prob = om.Problem(model)
        self.prob.setup()

    def test_example_location(self):

        # deep copy modeling options and adjust
        modeling_options = self.modeling_options
        modeling_options["collection"]["max_turbines_per_string"] = 4

        # create the OpenMDAO model
        model = om.Group()
        optiwindnet_coll_example = model.add_subsystem(
            "optiwindnet_coll",
            ard_own.OptiwindnetCollection(
                modeling_options=modeling_options,
            ),
        )
        prob = om.Problem(model)
        prob.setup()

        prob.set_val(
            "optiwindnet_coll.x_border",
            np.array(
                [1951, 1951, 386, 650, 624, 4, 4, 1152, 917, 957], dtype=np.float64
            ),
        )
        prob.set_val(
            "optiwindnet_coll.y_border",
            np.array(
                [200, 1383, 1383, 708, 678, 1036, 3, 3, 819, 854], dtype=np.float64
            ),
        )

        # run optiwindnet
        prob.run_model()

        assert (
            abs(
                prob.get_val("optiwindnet_coll.total_length_cables")
                - 6564.7653295074515
            )
            < 1e-7
        )
        cpJ = prob.check_partials(out_stream=None)
        assert_check_partials(cpJ, atol=1.0e-5, rtol=1.0e-3)


class TestOptiWindNetCollection5Turbines:

    def setup_method(self):
        n_turbines = 5
        theta_turbines = np.linspace(0.0, 2 * np.pi, n_turbines + 1)[:-1]
        x_turbines = 7.0 * 130.0 * np.sin(theta_turbines)
        y_turbines = 7.0 * 130.0 * np.cos(theta_turbines)
        x_substations = np.array([0.0])
        y_substations = np.array([0.0])
        self.modeling_options = make_modeling_options(
            x_turbines, y_turbines, x_substations, y_substations
        )

        # create the OpenMDAO model
        model = om.Group()
        self.optiwindnet_coll = model.add_subsystem(
            "optiwindnet_coll",
            ard_own.OptiwindnetCollection(
                modeling_options=self.modeling_options,
            ),
        )

        self.prob = om.Problem(model)
        self.prob.setup()

    def test_compute_partials_mini_pentagon(self):
        """
        run a really small case so that qualititative changes do not occur s.t.
        we can validate the differences using the OM built-ins; use a pentagon
        with a centered substation so there is no chaining.
        """

        # deep copy modeling options and adjust
        modeling_options = copy.deepcopy(self.modeling_options)
        modeling_options["layout"]["N_turbines"] = 5
        modeling_options["layout"]["N_substations"] = 1
        modeling_options["layout"]["x_substations"] = [0.0]
        modeling_options["layout"]["y_substations"] = [0.0]

        # create the OpenMDAO model
        model = om.Group()
        optiwindnet_coll_mini = model.add_subsystem(
            "optiwindnet_coll",
            ard_own.OptiwindnetCollection(
                modeling_options=modeling_options,
            ),
        )

        prob = om.Problem(model)
        prob.setup()

        # run optiwindnet
        prob.run_model()

        if False:  # for hand-debugging
            J0 = prob.compute_totals(
                "optiwindnet_coll.length_cables", "optiwindnet_coll.x_turbines"
            )
            prob.model.approx_totals()
            J0p = prob.compute_totals(
                "optiwindnet_coll.length_cables", "optiwindnet_coll.x_turbines"
            )

            print("J0:")
            print(J0)
            print("\n\n\n\n\nJ0p:")
            print(J0p)

            assert False

        # automated OpenMDAO fails because it re-runs the network work
        cpJ = prob.check_partials(out_stream=None)
        assert_check_partials(cpJ, atol=1.0e-5, rtol=1.0e-3)

    def test_compute_partials_mini_line(self):
        """
        run a really small case so that qualititative changes do not occur s.t.
        we can validate the differences using the OM built-ins; use a linear
        layout with a continuing substation so there is no variation.
        """

        # deep copy modeling options and adjust
        modeling_options = copy.deepcopy(self.modeling_options)
        modeling_options["layout"]["N_turbines"] = 5
        modeling_options["layout"]["N_substations"] = 1
        modeling_options["layout"]["x_substations"] = [5.0]  # overridden by set_val
        modeling_options["layout"]["y_substations"] = [5.0]  # overridden by set_val

        # create the OpenMDAO model
        model = om.Group()
        optiwindnet_coll_mini = model.add_subsystem(
            "optiwindnet_coll",
            ard_own.OptiwindnetCollection(
                modeling_options=modeling_options,
            ),
        )

        prob = om.Problem(model)
        prob.setup()
        # set in the variables
        s_turbines = np.array([1, 2, 3, 4, 5])
        X_turbines = 7.0 * 130.0 * s_turbines
        Y_turbines = np.log(7.0 * 130.0 * s_turbines)
        X_substations = np.array([-3.5 * 130.0])
        Y_substations = np.array([-3.5 * 130.0])
        prob.set_val("optiwindnet_coll.x_turbines", X_turbines)
        prob.set_val("optiwindnet_coll.y_turbines", Y_turbines)
        prob.set_val("optiwindnet_coll.x_substations", X_substations)
        prob.set_val("optiwindnet_coll.y_substations", Y_substations)

        # run optiwindnet
        prob.run_model()

        if False:  # for hand-debugging
            J0 = prob.compute_totals(
                "optiwindnet_coll.length_cables", "optiwindnet_coll.x_turbines"
            )
            prob.model.approx_totals()
            J0p = prob.compute_totals(
                "optiwindnet_coll.length_cables", "optiwindnet_coll.x_turbines"
            )

            print("J0:")
            print(J0)
            print("\n\n\n\n\nJ0p:")
            print(J0p)

            assert False

        # automated OpenMDAO fails because it re-runs the network work
        cpJ = prob.check_partials(out_stream=None)
        assert_check_partials(cpJ, atol=1.0e-5, rtol=1.0e-3)
