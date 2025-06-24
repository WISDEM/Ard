from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import openmdao.api as om

import pytest

import ard
import ard.utils.io
import ard.layout.boundary
import ard.layout.sunflower


@pytest.mark.usefixtures("subtests")
class TestConstraints:

    def setup_method(self):

        # specify the configuration/specification files to use
        filename_turbine_spec = (
            Path(ard.__file__).parents[1]
            / "examples"
            / "data"
            / "turbine_spec_IEA-3p4-130-RWT.yaml"
        ).absolute()  # toolset generalized turbine specification

        # load the turbine specification
        data_turbine = ard.utils.io.load_turbine_spec(filename_turbine_spec)

        self.N_turbines = 25
        region_assignments_single = np.zeros(self.N_turbines, dtype=int)

        # set up the modeling options
        self.modeling_options = {
            "farm": {
                "N_turbines": self.N_turbines,
                "boundary": {
                    "type": "polygon",
                    "vertices": [
                        np.array(
                            [
                                [-2.0, -2.0],
                                [2.0, -2.0],
                                [2.0, 2.0],
                                [-2.0, 2.0],
                            ]
                        )
                    ],
                    "turbine_region_assignments": region_assignments_single,
                },
            },
            "turbine": data_turbine,
        }

        # create a model
        model = om.Group()

        # add the sunflower layout thing
        model.add_subsystem(
            "layout",
            ard.layout.sunflower.SunflowerFarmLayout(
                modeling_options=self.modeling_options
            ),
            promotes=["spacing_target", "x_turbines", "y_turbines"],
        )
        model.add_subsystem(
            "landuse",
            ard.layout.sunflower.SunflowerFarmLanduse(
                modeling_options=self.modeling_options
            ),
            promotes=["x_turbines", "y_turbines", "area_tight"],
        )
        model.add_subsystem(
            "boundary",
            ard.layout.boundary.FarmBoundaryDistancePolygon(
                modeling_options=self.modeling_options
            ),
            promotes=["x_turbines", "y_turbines", "boundary_distances"],
        )

        # create, save, and setup the problem
        self.prob = om.Problem(model)
        self.prob.setup()

    def test_constraint_evaluation(self, subtests):
        """test one-shot evaluation of contraint distances (no derivatives)"""

        # loop over validation cases
        for spacing in [2.0, 5.0, 7.0]:

            # set in the spacing
            self.prob.set_val("spacing_target", spacing)
            self.prob.run_model()

            # load validation data from pyrite file using ard.utils.io
            validation_data = {
                "boundary_distances": self.prob.get_val("boundary_distances"),
            }
            with subtests.test(f"boundary_violations pyrite validation at {spacing}D"):
                ard.utils.test_utils.pyrite_validator(
                    validation_data,
                    Path(__file__).parent
                    / f"test_boundary_distances_{spacing:.01f}D_pyrite".replace(
                        ".", "p"
                    ),
                    rtol_val=5e-3,
                    # rewrite=True,  # uncomment to write new pyrite file
                )

    def test_constraint_optimization(self, subtests):
        """test boundary-constrained optimization distances (yes derivatives)"""

        # setup the working/design variables
        self.prob.model.add_design_var("spacing_target", lower=2.0, upper=13.0)
        self.prob.model.add_constraint("boundary_distances", upper=0.0)
        self.prob.model.add_objective("area_tight", scaler=-1.0)

        # configure the driver
        self.prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP")
        self.prob.driver.options["maxiter"] = 10  # short run

        # setupt the problem
        self.prob.setup()

        # set up the working/design variables
        self.prob.set_val("spacing_target", 7.0)

        # run the optimization driver
        self.prob.run_driver()

        # after 10 iterations, should have near-zero boundary distances
        with subtests.test("boundary distances near zero"):
            assert np.all(
                np.isclose(self.prob.get_val("boundary_distances"), 0.0)
                | (self.prob.get_val("boundary_distances") < 0.0)
            )

        # make sure the target spacing matches well
        spacing_target_validation = 5.46721656  # from a run on 24 June 2025
        area_target_validation = 10.49498327  # from a run on 24 June 2025
        with subtests.test("validation spacing matches"):
            assert np.isclose(
                self.prob.get_val("spacing_target"), spacing_target_validation
            )
        with subtests.test("validation area matches"):
            assert np.isclose(self.prob.get_val("area_tight"), area_target_validation)
