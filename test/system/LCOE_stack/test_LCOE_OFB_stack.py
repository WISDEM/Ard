from pathlib import Path

import numpy as np

import floris
import openmdao.api as om

from wisdem.optimization_drivers.nlopt_driver import NLoptDriver

import ard
import ard.test_utils
import ard.utils
import ard.wind_query as wq
import ard.glue.prototype as glue
import ard.cost.wisdem_wrap as cost_wisdem


class TestLCOE_OFB_stack:

    def setup_method(self):

        # create the wind query
        wind_rose_wrg = floris.wind_data.WindRoseWRG(
            Path(
                Path(ard.__file__).parent,
                "..",
                "examples",
                "data",
                "wrg_example.wrg",
            )
        )
        wind_rose_wrg.set_wd_step(90.0)
        wind_rose_wrg.set_wind_speeds(np.array([5.0, 10.0, 15.0, 20.0]))
        wind_rose = wind_rose_wrg.get_wind_rose_at_point(0.0, 0.0)
        wind_query = wq.WindQuery.from_FLORIS_WindData(wind_rose)

        # specify the configuration/specification files to use
        filename_turbine_spec = Path(
            Path(ard.__file__).parent,
            "..",
            "examples",
            "data",
            "turbine_spec_IEA-22-284-RWT.yaml",
        )  # toolset generalized turbine specification
        data_turbine_spec = ard.utils.load_turbine_spec(filename_turbine_spec)

        # set up the modeling options
        self.modeling_options = {
            "farm": {"N_turbines": 25},
            "site_depth": 50.0,
            "turbine": data_turbine_spec,
            "offshore": True,
            "floating": False,
        }

        # create the OM problem
        self.prob = glue.create_setup_OM_problem(
            modeling_options=self.modeling_options,
            wind_rose=wind_rose,
        )

    def test_model(self):

        # setup the latent variables for Orbit and FinanceSE
        cost_wisdem.Orbit_setup_latents(self.prob, self.modeling_options)
        cost_wisdem.FinanceSE_setup_latents(self.prob, self.modeling_options)

        # set up the working/design variables
        self.prob.set_val("spacing_primary", 7.0)
        self.prob.set_val("spacing_secondary", 7.0)
        self.prob.set_val("angle_orientation", 0.0)
        self.prob.set_val("angle_skew", 0.0)

        # run the model
        self.prob.run_model()

        # collapse the test result data
        test_data = {
            "AEP_val": float(self.prob.get_val("AEP_farm", units="GW*h")[0]),
            "CapEx_val": float(self.prob.get_val("tcc.tcc", units="MUSD")[0]),
            "BOS_val": float(
                self.prob.get_val("orbit.installation_capex", units="MUSD")[0]
            ),
            "OpEx_val": float(self.prob.get_val("opex.opex", units="MUSD/yr")[0]),
            "LCOE_val": float(self.prob.get_val("financese.lcoe", units="USD/MW/h")[0]),
        }

        # check the data against a pyrite file
        ard.test_utils.pyrite_validator(
            test_data,
            Path(
                Path(ard.__file__).parent,
                "..",
                "test",
                "system",
                "LCOE_stack",
                "test_LCOE_OFB_stack_pyrite.npz",
            ),
            # rewrite=True,  # uncomment to write new pyrite file
            rtol_val=5e-3,
        )


#
