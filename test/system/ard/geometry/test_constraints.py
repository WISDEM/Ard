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

        # hand-generated violation distances
        spacing_validation_pyrite_data = {
            2.0: np.array(
                [
                    -1.919091609396035,
                    -1.8107057809829714,
                    -1.8053161650896081,
                    -1.7141740024089815,
                    -1.7222969830036166,
                    -1.648610979318619,
                    -1.6489591896533966,
                    -1.6008808016777039,
                    -1.5818793177604684,
                    -1.566867858171463,
                    -1.5202935636043575,
                    -1.5447720885276797,
                    -1.4642487168312115,
                    -1.533650875091553,
                    -1.4140970110893256,
                    -1.5329259037971499,
                    -1.3703071475029005,
                    -1.5396649559367659,
                    -1.3333776593208313,
                    -1.4739657640457162,
                    -1.3037922978401195,
                    -1.4094654917717009,
                    -1.2819936275482182,
                    -1.3474184870719939,
                    -1.268366217613221,
                ]
            ),
            5.0: np.array(
                [
                    -1.7977597167483654,
                    -1.5267644524574289,
                    -1.5132904052734402,
                    -1.2854350209236154,
                    -1.3057424426078827,
                    -1.1215274930000305,
                    -1.1223979592323319,
                    -1.0022019743919377,
                    -0.9546983242034932,
                    -0.9171695709228516,
                    -0.8007339239120488,
                    -0.861930251121521,
                    -0.6606217622756985,
                    -0.8341271877288818,
                    -0.5352425575256481,
                    -0.8323147296905539,
                    -0.4257678985595719,
                    -0.8498109705387995,
                    -0.3334441184997579,
                    -0.6849144697189331,
                    -0.2594807147979741,
                    -0.5236637592315809,
                    -0.2049840688705466,
                    -0.3685461282730295,
                    -0.1709156036376953,
                ]
            ),
            7.0: np.array(
                [
                    -1.7168636321167143,
                    -1.3374702334403994,
                    -1.3186065554618835,
                    -0.9996089935302739,
                    -1.0280394554138252,
                    -0.7701385021209739,
                    -0.771357059478762,
                    -0.6030827760696441,
                    -0.536577582359314,
                    -0.48403751850128174,
                    -0.3210273981094416,
                    -0.40670228004455566,
                    -0.12487041950226185,
                    -0.36777806282043457,
                    0.05066037178043057,
                    -0.36524057388305664,
                    0.20392513275148785,
                    -0.38976888673497756,
                    0.3331782817840617,
                    -0.15888023376469315,
                    0.43672704696655373,
                    0.06687068939208984,
                    0.5130224227905273,
                    0.28403544425964355,
                    0.5607180595397981,
                ]
            ),
        }

        # loop over validation cases
        for spacing, validation_data in spacing_validation_pyrite_data.items():

            # set in the spacing
            self.prob.set_val("spacing_target", spacing)
            self.prob.run_model()

            # run a subtest to make sure its close
            with subtests.test(f"boundary_violations at {spacing}"):
                assert np.allclose(
                    self.prob.get_val("boundary_distances"),
                    validation_data,
                    atol=1e-5,
                )

    def test_constraint_optimization(self, subtests):

        # setup the working/design variables
        self.prob.model.add_design_var("spacing_target", lower=2.0, upper=13.0)
        self.prob.model.add_constraint("boundary_distances", upper=0.0)
        self.prob.model.add_objective("area_tight", scaler=-1.0)

        # configure the driver
        self.prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP")
        self.prob.driver.options["maxiter"] = 10  # DEBUG!!!!! short
        # self.prob.driver.options["debug_print"] = ["desvars", "nl_cons", "ln_cons", "objs"]

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
