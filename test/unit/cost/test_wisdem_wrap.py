import os
from pathlib import Path

import numpy as np
import openmdao.api as om
import openmdao.utils.assert_utils

import ard
import ard.utils
import ard.test_utils
import ard.layout.gridfarm as gridfarm
import ard.cost.wisdem_wrap as wcost
import ard.glue.prototype as glue


class TestLandBOSSE:

    def setup_method(self):

        # specify the configuration/specification files to use
        filename_turbine_spec = os.path.abspath(
            os.path.join(
                os.path.split(ard.__file__)[0],
                "..",
                "examples",
                "data",
                "turbine_spec_IEA-3p4-130-RWT.yaml",
            )
        )  # toolset generalized turbine specification

        # load the turbine specification
        data_turbine = ard.utils.load_turbine_spec(filename_turbine_spec)

        # set up the modeling options
        self.modeling_options = {
            "farm": {
                "N_turbines": 25,
            },
            "turbine": data_turbine,
        }

        # create an OM model and problem
        self.model = om.Group()
        self.gf = self.model.add_subsystem(
            "gridfarm",
            gridfarm.GridFarmLayout(modeling_options=self.modeling_options),
            promotes=["*"],
        )
        self.landbosse = self.model.add_subsystem(
            "landbosse",
            wcost.LandBOSSE(),
        )
        self.model.connect(  # effective primary spacing for BOS
            "spacing_effective_primary", "landbosse.turbine_spacing_rotor_diameters"
        )
        self.model.connect(  # effective secondary spacing for BOS
            "spacing_effective_secondary", "landbosse.row_spacing_rotor_diameters"
        )

        self.prob = om.Problem(self.model)
        self.prob.setup()

        # setup the latent variables for LandBOSSE and FinanceSE
        wcost.LandBOSSE_setup_latents(self.prob, self.modeling_options)
        # wcost.FinanceSE_setup_latents(self.prob, self.modeling_options)

    def test_baseline_farm(self):

        self.prob.set_val("gridfarm.spacing_primary", 7.0)
        self.prob.set_val("gridfarm.spacing_secondary", 7.0)
        self.prob.set_val("gridfarm.angle_orientation", 0.0)
        self.prob.set_val("gridfarm.angle_skew", 0.0)

        self.prob.run_model()

        # use a file of pyrite-standard data to validate against
        fn_pyrite = os.path.join(
            os.path.split(__file__)[0],
            "test_wisdem_wrap_baseline_farm.npz",
        )
        test_data = {
            "bos_capex_kW": self.prob.get_val("landbosse.bos_capex_kW", units="USD/kW"),
            "total_capex": self.prob.get_val("landbosse.total_capex", units="MUSD"),
        }
        # validate data against pyrite file
        ard.test_utils.pyrite_validator(
            test_data,
            fn_pyrite,
            rtol_val=5e-3,
            # rewrite=True,  # uncomment to write new pyrite file
        )


class TestPlantFinance:

    def setup_method(self):
        pass


class TestTurbineCapitalCosts:

    def setup_method(self):

        # create the OpenMDAO model
        model = om.Group()
        self.TCC = model.add_subsystem("tcc", wcost.TurbineCapitalCosts())

        # create the OpenMDAO problem
        self.prob = om.Problem(model)
        self.prob.setup()

    def test_compute(self):

        # generated with rng to scale roughly
        machine_rating_vec = np.array([8.06, 1.66, 1.99, 0.680, 6.11])
        tcc_per_kW_vec = np.array([7013.37, 283.41, 12949.30, 3482.23, 16025.91])
        offset_tcc_per_kW_vec = np.array([0.0, 46.84, 0.0, 1765.20, 0.0])
        number_turbines_vec = np.array([4, 3, 24, 1, 7])

        # compute using equation
        tcc_exact = (
            number_turbines_vec
            * machine_rating_vec
            * (tcc_per_kW_vec + offset_tcc_per_kW_vec)
        )

        # go over library values and compute using component
        for idx, tcc_val in enumerate(tcc_exact):
            self.prob.set_val("tcc.machine_rating", machine_rating_vec[idx])
            self.prob.set_val("tcc.tcc_per_kW", tcc_per_kW_vec[idx])
            self.prob.set_val("tcc.offset_tcc_per_kW", offset_tcc_per_kW_vec[idx])
            self.prob.set_val("tcc.turbine_number", number_turbines_vec[idx])

            # model output correct
            self.prob.run_model()
            assert np.isclose(self.prob.get_val("tcc.tcc"), tcc_val)

            # check exact derivatives against FD using OM tools
            check_partials = self.prob.check_partials(
                out_stream=None, show_only_incorrect=True
            )
            openmdao.utils.assert_utils.assert_check_partials(
                check_partials,
                atol=1e-3,
                rtol=1e-5,  # calibrated tolerances
            )


class TestOperatingExpenses:

    def setup_method(self):

        # create the OpenMDAO model
        model = om.Group()
        self.opex = model.add_subsystem("opex", wcost.OperatingExpenses())

        # create the OpenMDAO problem
        self.prob = om.Problem(model)
        self.prob.setup()

    def test_compute(self):

        # generated with rng to scale roughly
        machine_rating_vec = np.array([8.06, 1.66, 1.99, 0.680, 6.11])
        opex_per_kW_vec = np.array([701.33, 28.34, 1294.93, 348.22, 1602.59])
        number_turbines_vec = np.array([4, 3, 24, 1, 7])

        # compute using equation
        opex_exact = number_turbines_vec * machine_rating_vec * opex_per_kW_vec

        # go over library values and compute using component
        for idx, opex_val in enumerate(opex_exact):
            self.prob.set_val("opex.machine_rating", machine_rating_vec[idx])
            self.prob.set_val("opex.opex_per_kW", opex_per_kW_vec[idx])
            self.prob.set_val("opex.turbine_number", number_turbines_vec[idx])

            # model output correct
            self.prob.run_model()
            assert np.isclose(self.prob.get_val("opex.opex"), opex_val)

            # check exact derivatives against FD using OM tools
            check_partials = self.prob.check_partials(
                out_stream=None, show_only_incorrect=True
            )
            openmdao.utils.assert_utils.assert_check_partials(
                check_partials,
                atol=1e-3,
                rtol=1e-5,  # calibrated tolerances
            )
