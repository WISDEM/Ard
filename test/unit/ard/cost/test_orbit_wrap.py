from pathlib import Path
import pytest

import openmdao.api as om

import ard
import numpy as np
import ard.utils.io
import ard.layout.gridfarm as gridfarm
import ard.collection
import ard.cost.orbit_wrap as ocost


@pytest.mark.usefixtures("subtests")
class TestORBITNoApproxBranch:

    def test_raise_error(self):

        # specify the configuration/specification files to use
        filename_turbine = (
            Path(ard.__file__).parents[1]
            / "examples"
            / "data"
            / "windIO-plant_turbine_IEA-22MW-284m-RWT.yaml"
        ).absolute()  # toolset generalized turbine specification

        # set up the modeling options
        windIO_plant = {
            "site": {
                "energy_resource": {
                    "wind_resource": {
                        "shear": 0.2,
                    },
                },
            },
            "wind_farm": {
                "turbine": ard.utils.io.load_yaml(filename_turbine),
                "electrical_substations": [{
                    "electrical_substation": {
                        "coordinates": {
                            "x": [100.0],
                            "y": [100.0],
                        },
                    },
                }],
            },
        }
        modeling_options = {
            "windIO_plant": windIO_plant,
            "layout": {
                "N_turbines": 7,
                "N_substations": 1,
                "x_turbines": np.linspace(-7.0 * 400.0, 7.0 * 400.0, 7),
                "y_turbines": np.linspace(7.0 * 400.0, -7.0 * 400.0, 7),
            },
            "offshore": True,
            "floating": True,
            "platform": {
                "N_anchors": 3,
                "min_mooring_line_length_m": 500.0,
                "N_anchor_dimensions": 2,
            },
            "site_depth": 50.0,
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

        # create an OM model and problem
        model = om.Group()
        coll = model.add_subsystem(  # collection component
            "optiwindnet_coll",
            ard.collection.OptiwindnetCollection(
                modeling_options=modeling_options,
            ),
            promotes=[
                "x_turbines",
                "y_turbines",
                "x_substations",
                "y_substations",
            ],
        )

        orbit = model.add_subsystem(
            "orbit",
            ocost.ORBITDetailedGroup(
                modeling_options=modeling_options,
            ),
            promotes=[
                "x_turbines",
                "y_turbines",
                "x_substations",
                "y_substations",
            ],
        )
        model.connect("optiwindnet_coll.graph", "orbit.graph")

        model.set_input_defaults(
            "x_turbines", modeling_options["layout"]["x_turbines"], units="km"
        )
        model.set_input_defaults(
            "y_turbines", modeling_options["layout"]["y_turbines"], units="km"
        )
        model.set_input_defaults(
            "x_substations", [
                li["electrical_substation"]["coordinates"]["x"]
                for li in windIO_plant["wind_farm"]["electrical_substations"]
            ], units="km"
        )
        model.set_input_defaults(
            "y_substations", [
                li["electrical_substation"]["coordinates"]["y"]
                for li in windIO_plant["wind_farm"]["electrical_substations"]
            ], units="km"
        )

        prob = om.Problem(model)
        prob.setup()

        prob.set_val("x_turbines", modeling_options["layout"]["x_turbines"], units="m")
        prob.set_val("y_turbines", modeling_options["layout"]["y_turbines"], units="m")

        prob.set_val(
            "x_substations", [
                li["electrical_substation"]["coordinates"]["x"]
                for li in windIO_plant["wind_farm"]["electrical_substations"]
            ], units="km"
        )
        prob.set_val(
            "y_substations", [
                li["electrical_substation"]["coordinates"]["y"]
                for li in windIO_plant["wind_farm"]["electrical_substations"]
            ], units="km"
        )

        # this configuration should not work
        with pytest.raises(ValueError):
            prob.run_model()

    def test_baseline_farm(self, subtests):

        # specify the configuration/specification files to use
        filename_turbine_spec = (
            Path(ard.__file__).parents[1]
            / "examples"
            / "data"
            / "turbine_spec_IEA-22-284-RWT.yaml"
        ).absolute()  # toolset generalized turbine specification

        # load the turbine specification
        data_turbine = ard.utils.io.load_turbine_spec(filename_turbine_spec)

        # set up the modeling options
        modeling_options = {
            "farm": {
                "N_turbines": 7,
                "N_substations": 1,
                "x_turbines": np.linspace(-7.0 * 400.0, 7.0 * 400.0, 7),
                "y_turbines": np.linspace(7.0 * 400.0, -7.0 * 400.0, 7),
                "x_substations": np.array([0.1]),
                "y_substations": np.array([0.1]),
            },
            "turbine": data_turbine,
            "offshore": True,
            "floating": True,
            "platform": {
                "N_anchors": 3,
                "min_mooring_line_length_m": 500.0,
                "N_anchor_dimensions": 2,
            },
            "site_depth": 50.0,
            "collection": {
                "max_turbines_per_string": 8,
                "model_options": dict(
                    topology="radial",
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

        # create an OM model and problem
        model = om.Group()
        coll = model.add_subsystem(  # collection component
            "optiwindnet_coll",
            ard.collection.OptiwindnetCollection(
                modeling_options=modeling_options,
            ),
            promotes=[
                "x_turbines",
                "y_turbines",
                "x_substations",
                "y_substations",
            ],
        )

        orbit = model.add_subsystem(
            "orbit",
            ocost.ORBITDetail(
                modeling_options=modeling_options,
                floating=modeling_options["floating"],
                approximate_branches=False,
            ),
            promotes=[
                "x_turbines",
                "y_turbines",
                "x_substations",
                "y_substations",
            ],
        )
        model.connect("optiwindnet_coll.graph", "orbit.graph")

        model.set_input_defaults(
            "x_turbines", modeling_options["farm"]["x_turbines"], units="km"
        )
        model.set_input_defaults(
            "y_turbines", modeling_options["farm"]["y_turbines"], units="km"
        )
        model.set_input_defaults(
            "x_substations", modeling_options["farm"]["x_substations"], units="km"
        )
        model.set_input_defaults(
            "y_substations", modeling_options["farm"]["y_substations"], units="km"
        )

        prob = om.Problem(model)
        prob.setup()

        # setup the latent variables for ORBIT and FinanceSE
        ocost.ORBIT_setup_latents(prob, modeling_options)
        # wcost.FinanceSE_setup_latents(prob, modeling_options)

        prob.set_val("x_turbines", modeling_options["farm"]["x_turbines"], units="m")
        prob.set_val("y_turbines", modeling_options["farm"]["y_turbines"], units="m")

        prob.set_val(
            "x_substations", modeling_options["farm"]["x_substations"], units="km"
        )
        prob.set_val(
            "y_substations", modeling_options["farm"]["y_substations"], units="km"
        )

        prob.run_model()

        bos_capex = float(prob.get_val("orbit.bos_capex", units="MUSD"))
        total_capex = float(prob.get_val("orbit.total_capex", units="MUSD"))

        bos_capex_ref = 469.4374151677202
        total_capex_ref = 720.0174151677202

        with subtests.test(f"orbit_skew_bos"):
            assert np.isclose(bos_capex, bos_capex_ref, rtol=1e-3)
        with subtests.test(f"orbit_skew_total"):
            assert np.isclose(total_capex, total_capex_ref, rtol=1e-3)


@pytest.mark.usefixtures("subtests")
class TestORBITApproxBranch:

    def setup_method(self):

        # specify the configuration/specification files to use
        filename_turbine_spec = (
            Path(ard.__file__).parents[1]
            / "examples"
            / "data"
            / "turbine_spec_IEA-22-284-RWT.yaml"
        ).absolute()  # toolset generalized turbine specification

        # load the turbine specification
        data_turbine = ard.utils.io.load_turbine_spec(filename_turbine_spec)

        # set up the modeling options
        self.modeling_options = {
            "farm": {
                "N_turbines": 7,
                "N_substations": 1,
                "x_turbines": np.linspace(-7.0 * 400.0, 7.0 * 400.0, 7),
                "y_turbines": np.linspace(7.0 * 400.0, -7.0 * 400.0, 7),
                "x_substations": np.array([0.1]),
                "y_substations": np.array([0.1]),
            },
            "turbine": data_turbine,
            "offshore": True,
            "floating": True,
            "platform": {
                "N_anchors": 3,
                "min_mooring_line_length_m": 500.0,
                "N_anchor_dimensions": 2,
            },
            "site_depth": 50.0,
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

        # create an OM model and problem
        self.model = om.Group()
        self.coll = self.model.add_subsystem(  # collection component
            "optiwindnet_coll",
            ard.collection.OptiwindnetCollection(
                modeling_options=self.modeling_options,
            ),
            promotes=[
                "x_turbines",
                "y_turbines",
                "x_substations",
                "y_substations",
            ],
        )

        self.orbit = self.model.add_subsystem(
            "orbit",
            ocost.ORBITDetail(
                modeling_options=self.modeling_options,
                floating=self.modeling_options["floating"],
                approximate_branches=True,
            ),
            promotes=[
                "x_turbines",
                "y_turbines",
                "x_substations",
                "y_substations",
            ],
        )
        self.model.connect("optiwindnet_coll.graph", "orbit.graph")

        self.model.set_input_defaults(
            "x_turbines", self.modeling_options["farm"]["x_turbines"], units="km"
        )
        self.model.set_input_defaults(
            "y_turbines", self.modeling_options["farm"]["y_turbines"], units="km"
        )
        self.model.set_input_defaults(
            "x_substations", self.modeling_options["farm"]["x_substations"], units="km"
        )
        self.model.set_input_defaults(
            "y_substations", self.modeling_options["farm"]["y_substations"], units="km"
        )

        self.prob = om.Problem(self.model)
        self.prob.setup()

        # setup the latent variables for ORBIT and FinanceSE
        ocost.ORBIT_setup_latents(self.prob, self.modeling_options)
        # wcost.FinanceSE_setup_latents(self.prob, self.modeling_options)

    def test_baseline_farm(self, subtests):

        self.prob.set_val(
            "x_turbines", self.modeling_options["farm"]["x_turbines"], units="m"
        )
        self.prob.set_val(
            "y_turbines", self.modeling_options["farm"]["y_turbines"], units="m"
        )

        self.prob.set_val(
            "x_substations", self.modeling_options["farm"]["x_substations"], units="km"
        )
        self.prob.set_val(
            "y_substations", self.modeling_options["farm"]["y_substations"], units="km"
        )

        self.prob.run_model()

        bos_capex = float(self.prob.get_val("orbit.bos_capex", units="MUSD"))
        total_capex = float(self.prob.get_val("orbit.total_capex", units="MUSD"))

        bos_capex_ref = 469.4374151677202
        total_capex_ref = 720.0174151677202

        with subtests.test(f"orbit_skew_bos"):
            assert np.isclose(bos_capex, bos_capex_ref, rtol=1e-3)
        with subtests.test(f"orbit_skew_total"):
            assert np.isclose(total_capex, total_capex_ref, rtol=1e-3)
