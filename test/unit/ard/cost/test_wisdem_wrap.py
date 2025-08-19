from pathlib import Path

import numpy as np
import openmdao.api as om

import ard
import ard.utils.io
import ard.utils.test_utils
import ard.layout.gridfarm as gridfarm
import ard.cost.wisdem_wrap as wcost


class TestLandBOSSE:

    def setup_method(self):

        # set up the modeling options
        filename_turbine = (
            Path(ard.__file__).parents[1]
            / "examples"
            / "data"
            / "windIO-plant_turbine_IEA-3.4MW-130m-RWT.yaml"
        )
        self.modeling_options = {
            "windIO_plant": {
                "site": {
                    "energy_resource": {
                        "wind_resource": {
                            "shear": 0.2,
                        },
                    },
                },
                "wind_farm": {
                    "turbine": ard.utils.io.load_yaml(filename_turbine),
                },
            },
            "layout": {
                "N_turbines": 25,
                "spacing_primary": 0.0,  # reset in test_setup
                "spacing_secondary": 0.0,  # reset in test_setup
                "angle_orientation": 0.0,  # reset in test_setup
                "angle_skew": 0.0,  # reset in test_setup
            },
            "costs": {
                "num_blades": 3,
                "rated_thrust_N": 645645.83964671,
                "gust_velocity_m_per_s": 52.5,
                "blade_surface_area": 69.7974979,
                "tower_mass": 620440.7337521,
                "nacelle_mass": 101985.82836439,
                "hub_mass": 8384.07517646,
                "blade_mass": 14563.41339641,
                "foundation_height": 0.0,
                "commissioning_cost_kW": 44.0,
                "decommissioning_cost_kW": 58.0,
                "trench_len_to_substation_km": 50.0,
                "distance_to_interconnect_mi": 4.97096954,
                "interconnect_voltage_kV": 130.0,
            },
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
            wcost.LandBOSSEArdComp(),
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
        fn_pyrite = Path(__file__).parent / "test_landbosse_wrap_baseline_farm.npz"
        test_data = {
            "bos_capex_kW": self.prob.get_val("landbosse.bos_capex_kW", units="USD/kW"),
            "total_capex": self.prob.get_val("landbosse.total_capex", units="MUSD"),
        }
        # validate data against pyrite file
        ard.utils.test_utils.pyrite_validator(
            test_data,
            fn_pyrite,
            rtol_val=5e-3,
            # rewrite=True,  # uncomment to write new pyrite file
        )


class TestORBIT:

    def setup_method(self):

        filename_turbine = (
            Path(ard.__file__).parents[1]
            / "examples"
            / "data"
            / "windIO-plant_turbine_IEA-3.4MW-130m-RWT.yaml"
        )

        # set up the modeling options
        self.modeling_options = {
            "windIO_plant": {
                "wind_farm": {
                    "turbine": ard.utils.io.load_yaml(filename_turbine),
                },
            },
            "layout": {
                "N_turbines": 25,
                "spacing_primary": 0.0,  # reset in test
                "spacing_secondary": 0.0,  # reset in test
                "angle_orientation": 0.0,  # reset in test
                "angle_skew": 0.0,  # reset in test
            },
            "site_depth": 50.0,
            "offshore": True,
            "floating": True,
        }

        # create an OM model and problem
        self.model = om.Group()
        self.gf = self.model.add_subsystem(
            "gridfarm",
            gridfarm.GridFarmLayout(modeling_options=self.modeling_options),
            promotes=["*"],
        )
        self.orbit = self.model.add_subsystem(
            "orbit",
            wcost.ORBIT(),
        )
        self.model.connect(  # effective primary spacing for BOS
            "spacing_effective_primary", "orbit.plant_turbine_spacing"
        )
        self.model.connect(  # effective secondary spacing for BOS
            "spacing_effective_secondary", "orbit.plant_row_spacing"
        )

        self.prob = om.Problem(self.model)
        self.prob.setup()

        # setup the latent variables for ORBIT and FinanceSE
        wcost.ORBIT_setup_latents(self.prob, self.modeling_options)
        # wcost.FinanceSE_setup_latents(self.prob, self.modeling_options)

    def test_baseline_farm(self, subtests):

        self.prob.set_val("gridfarm.spacing_primary", 7.0)
        self.prob.set_val("gridfarm.spacing_secondary", 7.0)
        self.prob.set_val("gridfarm.angle_orientation", 0.0)
        self.prob.set_val("gridfarm.angle_skew", 0.0)

        self.prob.run_model()

        # use a file of pyrite-standard data to validate against
        fn_pyrite = Path(__file__).parent / "test_orbit_wrap_baseline_farm.npz"
        test_data = {
            "bos_capex": self.prob.get_val("orbit.bos_capex", units="USD"),
            "total_capex": self.prob.get_val("orbit.total_capex", units="MUSD"),
        }
        # validate data against pyrite file
        pyrite_data = ard.utils.test_utils.pyrite_validator(
            test_data,
            fn_pyrite,
            rtol_val=5e-3,
            load_only=True,
            # rewrite=True,  # uncomment to write new pyrite file
        )

        # Validate each key-value pair using subtests
        for key, value in test_data.items():
            with subtests.test(key=key):
                assert np.isclose(value, pyrite_data[key], rtol=5e-3), (
                    f"Mismatch for {key}: " f"expected {pyrite_data[key]}, got {value}"
                )


class TestPlantFinance:

    def setup_method(self):
        pass


class TestTurbineCapitalCosts:

    def setup_method(self):
        pass


class TestOperatingExpenses:

    def setup_method(self):
        pass


class TestSetValues:

    def setup_method(self):

        # build paraboloid model for testing
        prob = om.Problem()
        prob.model.add_subsystem(
            "paraboloid", om.ExecComp("f = (x-3)**2 + x*y + (y+4)**2 - 3")
        )
        prob.setup()

        # Set initial values.
        wcost.set_values(prob, {"x": 3.0, "y": -4.0})

        self.prob = prob

    def test_x(self):
        assert self.prob["paraboloid.x"] == 3.0

    def test_y(self):
        assert self.prob["paraboloid.y"] == -4.0
