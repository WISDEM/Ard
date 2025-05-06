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


@pytest.mark.usefixtures("subtests")
class TestOptiWindNetCollection:

    def setup_method(self):

        # create the farm layout specification
        self.farm_spec = {}
        self.farm_spec["xD_farm"], self.farm_spec["yD_farm"] = [
            7 * v.flatten()
            for v in np.meshgrid(
                np.linspace(-2, 2, 5, dtype=int), np.linspace(-2, 2, 5, dtype=int)
            )
        ]
        self.farm_spec["x_substations"] = np.array([-500.0, 500.0], dtype=np.float64)
        self.farm_spec["y_substations"] = np.array([-500.0, 500.0], dtype=np.float64)

        # specify the configuration/specification files to use
        filename_turbine_spec = (
            Path(ard.__file__).parents[1]
            / "examples"
            / "data"
            / "turbine_spec_IEA-3p4-130-RWT.yaml"
        )  # toolset generalized turbine specification
        data_turbine_spec = ard.utils.io.load_turbine_spec(filename_turbine_spec)

        # set up the modeling options
        self.N_turbines = len(self.farm_spec["xD_farm"])
        self.N_substations = len(self.farm_spec["x_substations"])
        self.modeling_options = modeling_options = {
            "farm": {
                "N_turbines": self.N_turbines,
                "N_substations": self.N_substations,
            },
            "turbine": data_turbine_spec,
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
        model = om.Group()
        self.optiwindnet_coll = model.add_subsystem(
            "optiwindnet_coll",
            ard_own.optiwindnetCollection(
                modeling_options=modeling_options,
            ),
        )

        self.prob = om.Problem(model)
        self.prob.setup()

    def test_distance_function_vs_optiwindnet(self, subtests):

        name_case = "farm"
        capacity = 8  # maximum load on a chain #TODO make the capacity a user input

        # roll up the coordinates into a form that optiwindnet
        XY_turbines = np.vstack(
            [130 * self.farm_spec["xD_farm"], 130 * self.farm_spec["yD_farm"]]
        ).T

        x_min = np.min(XY_turbines[:, 0]) - 0.25 * np.ptp(XY_turbines[:, 0])
        x_max = np.max(XY_turbines[:, 0]) + 0.25 * np.ptp(XY_turbines[:, 0])
        y_min = np.min(XY_turbines[:, 1]) - 0.25 * np.ptp(XY_turbines[:, 1])
        y_max = np.max(XY_turbines[:, 1]) + 0.25 * np.ptp(XY_turbines[:, 1])
        XY_boundaries = np.array(
            [
                [x_max, y_max],
                [x_min, y_max],
                [x_min, y_min],
                [x_max, y_min],
            ]
        )
        XY_substations = np.vstack(
            [self.farm_spec["x_substations"], self.farm_spec["y_substations"]]
        ).T

        result, S, G, H = ard_own.optiwindnet_wrapper(
            XY_turbines, XY_substations, XY_boundaries, name_case, capacity
        )

        # extract the outputs
        edges = H.edges()
        self.graph = H

        lengths = []

        for idx_edge, edge in enumerate(edges):
            e0, e1 = edge
            x0, y0 = (
                XY_substations[self.N_substations + e0, :]
                if e0 < 0
                else XY_turbines[e0, :]
            )
            x1, y1 = (
                XY_substations[self.N_substations + e1, :]
                if e1 < 0
                else XY_turbines[e1, :]
            )

            with subtests.test(f"edge: {idx_edge}"):
                lengths.append(edges[edge]["length"])
                assert np.isclose(
                    edges[edge]["length"], ard_own.distance_function(x0, y0, x1, y1)
                )

    def test_modeling(self, subtests):
        """
        make sure the modeling_options has what we need for farmaero
        """

        with subtests.test("modeling_options"):
            assert "modeling_options" in [
                k for k, _ in self.optiwindnet_coll.options.items()
            ]
        with subtests.test("farm"):
            assert "farm" in self.optiwindnet_coll.options["modeling_options"].keys()
        with subtests.test("N_turbines"):
            assert (
                "N_turbines"
                in self.optiwindnet_coll.options["modeling_options"]["farm"].keys()
            )
        with subtests.test("N_substations"):
            assert (
                "N_substations"
                in self.optiwindnet_coll.options["modeling_options"]["farm"].keys()
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
                # "length_cables",
                # "load_cables",
                "total_length_cables",
                # "max_load_cables",
            ]:
                assert var_to_check in output_list

            # make sure that the outputs in the component match what we planned
            discrete_output_list = [
                k for k, v in self.optiwindnet_coll._discrete_outputs.items()
            ]
            for var_to_check in [
                "length_cables",
                "load_cables",
                # "total_length_cables",
                "max_load_cables",
            ]:
                assert var_to_check in discrete_output_list

    @pytest.mark.skipif(
        platform.system() in ["Linux", "Windows"], reason="Test does not pass on Linux"
    )
    def test_compute_pyrite(self):

        # set in the variables
        X_turbines = 130.0 * self.farm_spec["xD_farm"]
        Y_turbines = 130.0 * self.farm_spec["yD_farm"]
        X_substations = self.farm_spec["x_substations"]
        Y_substations = self.farm_spec["y_substations"]
        self.prob.set_val("optiwindnet_coll.x_turbines", X_turbines)
        self.prob.set_val("optiwindnet_coll.y_turbines", Y_turbines)
        self.prob.set_val("optiwindnet_coll.x_substations", X_substations)
        self.prob.set_val("optiwindnet_coll.y_substations", Y_substations)

        # run optiwindnet
        self.prob.run_model()

        # # DEBUG!!!!! viz for verification
        # gplot(self.optiwindnet_coll.graph)
        # plt.savefig("/Users/cfrontin/Downloads/dummy.png")  # DEBUG!!!!!

        # collect data to validate
        validation_data = {
            "length_cables": self.prob.get_val("optiwindnet_coll.length_cables")
            / 1.0e3,
            "load_cables": self.prob.get_val("optiwindnet_coll.load_cables"),
        }

        os_name = platform.system()

        if os_name == "Linux":
            pass
            # Run Linux specific tests
            # validate data against pyrite file
            ard.utils.test_utils.pyrite_validator(
                validation_data,
                Path(__file__).parent / "test_optiwindnet_pyrite_macos.npz",
                rtol_val=5e-3,
                # rewrite=True,  # uncomment to write new pyrite file
            )
        elif os_name == "Darwin":
            # Run macos specific tests
            # validate data against pyrite file
            ard.utils.test_utils.pyrite_validator(
                validation_data,
                Path(__file__).parent / "test_optiwindnet_pyrite_macos.npz",
                rtol_val=5e-3,
                # rewrite=True,  # uncomment to write new pyrite file
            )
        elif os_name == "Windows":
            # Run Windows specific tests
            # validate data against pyrite file
            ard.utils.test_utils.pyrite_validator(
                validation_data,
                Path(__file__).parent / "test_optiwindnet_pyrite_macos.npz",
                rtol_val=5e-3,
                # rewrite=True,  # uncomment to write new pyrite file
            )
        else:
            pass

    def test_compute_partials_mini_pentagon(self):
        """
        run a really small case so that qualititative changes do not occur s.t.
        we can validate the differences using the OM built-ins; use a pentagon
        with a centered substation so there is no chaining.
        """

        # deep copy modeling options and adjust
        modeling_options = copy.deepcopy(self.modeling_options)
        modeling_options["farm"]["N_turbines"] = 5
        modeling_options["farm"]["N_substations"] = 1

        # create the OpenMDAO model
        model = om.Group()
        optiwindnet_coll_mini = model.add_subsystem(
            "optiwindnet_coll",
            ard_own.optiwindnetCollection(
                modeling_options=modeling_options,
            ),
        )

        prob = om.Problem(model)
        prob.setup()
        # set in the variables
        theta_turbines = np.linspace(
            0.0, 2 * np.pi, modeling_options["farm"]["N_turbines"] + 1
        )[:-1]
        X_turbines = 7.0 * 130.0 * np.sin(theta_turbines)
        Y_turbines = 7.0 * 130.0 * np.cos(theta_turbines)
        X_substations = np.array([0.0])
        Y_substations = np.array([0.0])
        prob.set_val("optiwindnet_coll.x_turbines", X_turbines)
        prob.set_val("optiwindnet_coll.y_turbines", Y_turbines)
        prob.set_val("optiwindnet_coll.x_substations", X_substations)
        prob.set_val("optiwindnet_coll.y_substations", Y_substations)

        # run optiwindnet
        prob.run_model()

        # # DEBUG!!!!! viz for verification
        # gplot(optiwindnet_coll_mini.graph)
        # plt.savefig("/Users/cfrontin/Downloads/dummy.png")  # DEBUG!!!!!

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
        modeling_options["farm"]["N_turbines"] = 5
        modeling_options["farm"]["N_substations"] = 1

        # create the OpenMDAO model
        model = om.Group()
        optiwindnet_coll_mini = model.add_subsystem(
            "optiwindnet_coll",
            ard_own.optiwindnetCollection(
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

        # # DEBUG!!!!! viz for verification
        # gplot(optiwindnet_coll_mini.graph)
        # plt.savefig("dummy.png")  # DEBUG!!!!!

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
