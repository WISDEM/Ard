from pathlib import Path
import pytest

import openmdao.api as om

import ard
import numpy as np
import ard.utils.io

# import ard.utils.test_utils
import ard.layout.gridfarm as gridfarm
import ard.collection
import ard.cost.orbit_wrap as ocost


@pytest.mark.usefixtures("subtests")
class TestORBIT:

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
                "N_turbines": 25,
                "N_substations": 1,
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
                "solver_name": "appsi_highs",
                "solver_options": dict(
                    time_limit=60,
                    mip_rel_gap=0.005,  # TODO ???
                ),
            },
        }

        # create an OM model and problem
        self.model = om.Group()
        self.gf = self.model.add_subsystem(
            "gridfarm",
            gridfarm.GridFarmLayout(modeling_options=self.modeling_options),
            promotes=["*"],
        )

        self.coll = self.model.add_subsystem(  # collection component
            "optiwindnet_coll",
            ard.collection.optiwindnetCollection(
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
            ),
            promotes=[
                "x_turbines",
                "y_turbines",
                "x_substations",
                "y_substations",
            ],
        )
        self.model.connect("optiwindnet_coll.graph", "orbit.graph")

        self.model.set_input_defaults("x_substations", units="km")
        self.model.set_input_defaults("y_substations", units="km")

        self.prob = om.Problem(self.model)
        self.prob.setup()

        # setup the latent variables for ORBIT and FinanceSE
        ocost.ORBIT_setup_latents(self.prob, self.modeling_options)
        # wcost.FinanceSE_setup_latents(self.prob, self.modeling_options)

    def test_baseline_farm(self, subtests):

        values_ref = {
            0.0: {
                "bos_capex": 954.936955639988,
                "total_capex": 1460.936955639988,
            },
            2.5: {
                "bos_capex": 959.6976715808297,
                "total_capex": 1465.6976715808296,
            },
            5.0: {
                "bos_capex": 976.4715590100313,
                "total_capex": 1482.4715590100311,
            },
        }

        for angle_skew in values_ref.keys():

            self.prob.set_val("x_substations", [0.1], units="km")
            self.prob.set_val("y_substations", [0.1], units="km")

            self.prob.set_val("gridfarm.spacing_primary", 7.0)
            self.prob.set_val("gridfarm.spacing_secondary", 7.0)
            self.prob.set_val("gridfarm.angle_orientation", 0.0)
            self.prob.set_val("gridfarm.angle_skew", angle_skew)

            self.prob.run_model()

            bos_capex = float(self.prob.get_val("orbit.bos_capex", units="MUSD"))
            total_capex = float(self.prob.get_val("orbit.total_capex", units="MUSD"))

            bos_capex_ref = values_ref[angle_skew]["bos_capex"]
            total_capex_ref = values_ref[angle_skew]["total_capex"]

            with subtests.test(f"orbit_skew{angle_skew:.1f}_bos"):
                assert np.isclose(bos_capex, bos_capex_ref, rtol=1e-3)
            with subtests.test(f"orbit_skew{angle_skew:.1f}_total"):
                assert np.isclose(total_capex, total_capex_ref, rtol=1e-3)
