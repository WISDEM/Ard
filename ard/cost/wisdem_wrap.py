import warnings
import numpy as np

import openmdao.api as om
from wisdem.plant_financese.plant_finance import PlantFinance as PlantFinance_orig
from wisdem.landbosse.landbosse_omdao.landbosse import LandBOSSE as LandBOSSE_orig
from wisdem.orbit.orbit_api import Orbit as Orbit_orig

from ard.cost.approximate_turbine_spacing import SpacingApproximations


class LandBOSSEWithSpacingApproximations(om.Group):
    """
    OpenMDAO group that connects the SpacingApproximations component to the LandBOSSE component.

    This group calculates the turbine spacing using the SpacingApproximations and passes it
    to the LandBOSSE component for further cost estimation.
    """

    def initialize(self):
        """Initialize the group and declare options."""
        self.options.declare(
            "modeling_options", types=dict, desc="Ard modeling options"
        )

    def setup(self):
        """Set up the group by adding and connecting components."""
        # Add the PrimarySpacingApproximations component
        self.add_subsystem(
            "spacing_approximations",
            SpacingApproximations(modeling_options=self.options["modeling_options"]),
            promotes_inputs=["total_length_cables"],
        )

        # Add the LandBOSSE component
        self.add_subsystem(
            "landbosse",
            LandBOSSEGroup(modeling_options=self.options["modeling_options"]),
            promotes_inputs=[
                "*",
                (
                    "turbine_spacing_rotor_diameters",
                    "internal_turbine_spacing_rotor_diameters",
                ),
                (
                    "row_spacing_rotor_diameters",
                    "internal_row_spacing_rotor_diameters",
                ),
            ],
            promotes_outputs=["*"],  # Expose all outputs from LandBOSSE
        )

        # Connect the turbine and row spacing outputs from the approximations to LandBOSSE
        self.connect(
            "spacing_approximations.primary_turbine_spacing_diameters",
            "internal_turbine_spacing_rotor_diameters",
        )

        self.connect(
            "spacing_approximations.secondary_turbine_spacing_diameters",
            "internal_row_spacing_rotor_diameters",
        )


class LandBOSSEGroup(om.Group):

    def initialize(self):
        """Initialize the group and declare options."""
        self.options.declare(
            "modeling_options", types=dict, desc="Ard modeling options"
        )

    def setup(self):

        # add IVCs for landbosse
        variable_mapping = LandBOSSE_setup_latents(
            modeling_options=self.options["modeling_options"]
        )

        # create source independent variable components for LandBOSSE inputs
        for key, meta in variable_mapping.items():
            if key in ["num_turbines", "number_of_blades"]:
                comp = om.IndepVarComp()
                comp.add_discrete_output(name=key, val=meta["val"])
                self.add_subsystem(f"IVC_landbosse_{key}", comp, promotes=["*"])
            else:
                self.add_subsystem(
                    f"IVC_landbosse_{key}",
                    om.IndepVarComp(key, val=meta["val"], units=meta["units"]),
                    promotes=["*"],
                )

        # add landbosse
        self.add_subsystem(
            "landbosse",
            LandBOSSE_orig(),
            promotes=[
                "total_capex",
                "total_capex_kW",
                "bos_capex_kW",
                "turbine_spacing_rotor_diameters",
                "row_spacing_rotor_diameters",
            ],
        )

        # connect
        for key, val in variable_mapping.items():
            self.connect(key, f"landbosse.{key}")


class ORBITGroup(om.Group):

    def initialize(self):
        """Initialize the group and declare options."""
        self.options.declare(
            "modeling_options", types=dict, desc="Ard modeling options"
        )

    def setup(self):

        # add IVCs for landbosse
        variable_mapping = ORBIT_setup_latents(
            modeling_options=self.options["modeling_options"]
        )

        # create source independent variable components for LandBOSSE inputs
        for key, meta in variable_mapping.items():
            if key in ["number_of_turbines", "number_of_blades", "num_mooring_lines"]:
                comp = om.IndepVarComp()
                comp.add_discrete_output(name=key, val=meta["val"])
                self.add_subsystem(f"IVC_orbit_{key}", comp, promotes=["*"])
            else:
                self.add_subsystem(
                    f"IVC_orbit_{key}",
                    om.IndepVarComp(key, val=meta["val"], units=meta["units"]),
                    promotes=["*"],
                )

        # add orbit
        self.add_subsystem(
            "orbit",
            Orbit_orig(
                floating=self.options["modeling_options"]["floating"],
                jacket=self.options["modeling_options"].get("jacket"),
                jacket_legs=self.options["modeling_options"].get("jacket_legs"),
            ),
            promotes=[
                "total_capex",
                "total_capex_kW",
                "bos_capex",
                "installation_capex",
                "plant_turbine_spacing",
                "plant_row_spacing",
            ],
        )

        # connect
        for key, val in variable_mapping.items():
            self.connect(key, f"orbit.{key}")


class FinanceSEGroup(om.Group):

    def initialize(self):
        """Initialize the group and declare options."""
        self.options.declare(
            "modeling_options", types=dict, desc="Ard modeling options"
        )

    def setup(self):

        # add IVCs for landbosse
        variable_mapping = FinanceSE_setup_latents(
            modeling_options=self.options["modeling_options"]
        )

        # create source independent variable components for LandBOSSE inputs
        for key, meta in variable_mapping.items():
            if key in [
                "turbine_number",
            ]:
                comp = om.IndepVarComp()
                comp.add_discrete_output(name=key, val=meta["val"])
                self.add_subsystem(f"IVC_financese_{key}", comp, promotes=["*"])
            else:
                self.add_subsystem(
                    f"IVC_financese_{key}",
                    om.IndepVarComp(key, val=meta["val"], units=meta["units"]),
                    promotes=["*"],
                )

        # add financese #TODO check promotes
        self.add_subsystem(
            "financese",
            PlantFinance_orig(),
            promotes=[
                "offset_tcc_per_kW",
                "plant_aep_in",
                "bos_per_kW",
                # "tcc_per_kW",
                "lcoe",
            ],
        )

        # connect
        for key, val in variable_mapping.items():
            self.connect(key, f"financese.{key}")


class TurbineCapitalCosts(om.ExplicitComponent):
    """
    A simple component to compute the turbine capital costs.

    Inputs
    ------
    machine_rating : float
        rating of the wind turbine in kW
    tcc_per_kW : float
        turbine capital costs per kW (as output from WISDEM tools)
    offset_tcc_per_kW : float
        additional tcc per kW (offset)

    Discrete Inputs
    ---------------
    turbine_number : int
        number of turbines in the farm

    Outputs
    -------
    tcc : float
        turbine capital costs in USD
    """

    def setup(self):
        """Setup of OM component."""
        self.add_input("machine_rating", 0.0, units="kW")
        self.add_input("tcc_per_kW", 0.0, units="USD/kW")
        self.add_input("offset_tcc_per_kW", 0.0, units="USD/kW")
        self.add_discrete_input("turbine_number", 0)
        self.add_output("tcc", 0.0, units="USD")

    def setup_partials(self):
        """Derivative setup for OM component."""
        # complex step for simple gradients
        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """Computation for the OM compoent."""
        # Unpack parameters
        t_rating = inputs["machine_rating"]
        n_turbine = discrete_inputs["turbine_number"]
        tcc_per_kW = inputs["tcc_per_kW"] + inputs["offset_tcc_per_kW"]
        outputs["tcc"] = n_turbine * tcc_per_kW * t_rating


class OperatingExpenses(om.ExplicitComponent):
    """
    A simple component to compute the operating costs.

    Inputs
    ------
    machine_rating : float
        rating of the wind turbine in kW
    opex_per_kW : float
        annual operating and maintenance costs per kW (as output from WISDEM
        tools)

    Discrete Inputs
    ---------------
    turbine_number : int
        number of turbines in the farm

    Outputs
    -------
    opex : float
        annual operating and maintenance costs in USD
    """

    def setup(self):
        """Setup of OM component."""
        self.add_input("machine_rating", 0.0, units="kW")
        self.add_input("opex_per_kW", 0.0, units="USD/kW/yr")
        self.add_discrete_input("turbine_number", 0)
        self.add_output("opex", 0.0, units="USD/yr")

    def setup_partials(self):
        """Derivative setup for OM component."""
        # complex step for simple gradients
        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """Computation for the OM compoent."""
        # Unpack parameters
        t_rating = inputs["machine_rating"]
        n_turbine = discrete_inputs["turbine_number"]
        opex_per_kW = inputs["opex_per_kW"]
        outputs["opex"] = n_turbine * opex_per_kW * t_rating


def LandBOSSE_setup_latents(modeling_options: dict) -> None:
    """
    A function to set up the LandBOSSE latent variables using modeling options.

    Parameters
    ----------
    prob : openmdao.api.Problem
        an OpenMDAO problem for which we want to setup the LandBOSSE latent
        variables
    modeling_options : dict
        a modeling options dictionary
    """

    # Define the mapping between OpenMDAO variable names and modeling_options keys
    offshore_fixed_keys = [
        "monopile_mass",
        "monopile_cost",
    ]

    offshore_floating_keys = [
        "num_mooring_lines",
        "mooring_line_mass",
        "mooring_line_diameter",
        "mooring_line_length",
        "anchor_mass",
        "floating_substructure_cost",
    ]

    def _base_common():
        return {
            "num_turbines": {
                "val": modeling_options["layout"]["N_turbines"],
                "units": None,
            },
            "turbine_rating_MW": {
                "val": modeling_options["windIO_plant"]["wind_farm"]["turbine"][
                    "performance"
                ]["rated_power"]
                / 1.0e6,
                "units": "MW",
            },
            "hub_height_meters": {
                "val": modeling_options["windIO_plant"]["wind_farm"]["turbine"][
                    "hub_height"
                ],
                "units": "m",
            },
            "rotor_diameter_m": {
                "val": modeling_options["windIO_plant"]["wind_farm"]["turbine"][
                    "rotor_diameter"
                ],
                "units": "m",
            },
            "number_of_blades": {
                "val": modeling_options["costs"]["num_blades"],
                "units": None,
            },
            "tower_mass": {
                "val": modeling_options["costs"]["tower_mass"],
                "units": "t",
            },
            "nacelle_mass": {
                "val": modeling_options["costs"]["nacelle_mass"],
                "units": "t",
            },
            "blade_mass": {
                "val": modeling_options["costs"]["blade_mass"],
                "units": "t",
            },
            "commissioning_cost_kW": {
                "val": modeling_options["costs"]["commissioning_cost_kW"],
                "units": "USD/kW",
            },
            "decommissioning_cost_kW": {
                "val": modeling_options["costs"]["decommissioning_cost_kW"],
                "units": "USD/kW",
            },
        }

    if any(key in modeling_options["costs"] for key in offshore_fixed_keys):
        variable_mapping = _base_common()
        variable_mapping.update(
            {
                "monopile_mass": {
                    "val": modeling_options["costs"]["monopile_mass"],
                    "units": "kg",
                },
                "monopile_cost": {
                    "val": modeling_options["costs"]["monopile_cost"],
                    "units": "USD",
                },
            }
        )
    elif any(key in modeling_options["costs"] for key in offshore_floating_keys):
        variable_mapping = _base_common()
        variable_mapping.update(
            {
                "num_mooring_lines": {
                    "val": modeling_options["costs"]["num_mooring_lines"],
                    "units": None,
                },
                "mooring_line_mass": {
                    "val": modeling_options["costs"]["mooring_line_mass"],
                    "units": "kg",
                },
                "mooring_line_diameter": {
                    "val": modeling_options["costs"]["mooring_line_diameter"],
                    "units": "m",
                },
                "mooring_line_length": {
                    "val": modeling_options["costs"]["mooring_line_length"],
                    "units": "m",
                },
                "anchor_mass": {
                    "val": modeling_options["costs"]["anchor_mass"],
                    "units": "kg",
                },
                "floating_substructure_cost": {
                    "val": modeling_options["costs"]["floating_substructure_cost"],
                    "units": "USD",
                },
            }
        )
    else:
        variable_mapping = _base_common()
        variable_mapping.update(
            {
                "rated_thrust_N": {
                    "val": modeling_options["costs"]["rated_thrust_N"],
                    "units": "N",
                },
                "gust_velocity_m_per_s": {
                    "val": modeling_options["costs"]["gust_velocity_m_per_s"],
                    "units": "m/s",
                },
                "blade_surface_area": {
                    "val": modeling_options["costs"]["blade_surface_area"],
                    "units": "m**2",
                },
                "hub_mass": {
                    "val": modeling_options["costs"]["hub_mass"],
                    "units": "kg",
                },
                "foundation_height": {
                    "val": modeling_options["costs"]["foundation_height"],
                    "units": "m",
                },
                "trench_len_to_substation_km": {
                    "val": modeling_options["costs"]["trench_len_to_substation_km"],
                    "units": "km",
                },
                "distance_to_interconnect_mi": {
                    "val": modeling_options["costs"]["distance_to_interconnect_mi"],
                    "units": "mi",
                },
                "interconnect_voltage_kV": {
                    "val": modeling_options["costs"]["interconnect_voltage_kV"],
                    "units": "kV",
                },
            }
        )

    return variable_mapping


def ORBIT_setup_latents(modeling_options: dict) -> None:
    """
    A function to set up the ORBIT latent variables using modeling options.

    Parameters
    ----------
    prob : openmdao.api.Problem
        an OpenMDAO problem for which we want to setup the ORBIT latent
        variables
    modeling_options : dict
        a modeling options dictionary
    """

    variable_mapping = {
        "turbine_rating": {
            "val": modeling_options["windIO_plant"]["wind_farm"]["turbine"][
                "performance"
            ]["rated_power"],
            "units": "W",
        },
        "site_depth": {"val": modeling_options["site_depth"], "units": "m"},
        "number_of_turbines": {
            "val": modeling_options["layout"]["N_turbines"],
            "units": None,
        },
        "number_of_blades": {
            "val": modeling_options["costs"]["num_blades"],
            "units": None,
        },
        "hub_height": {
            "val": modeling_options["windIO_plant"]["wind_farm"]["turbine"][
                "hub_height"
            ],
            "units": "m",
        },
        "turbine_rotor_diameter": {
            "val": modeling_options["windIO_plant"]["wind_farm"]["turbine"][
                "rotor_diameter"
            ],
            "units": "m",
        },
        "tower_length": {
            "val": modeling_options["costs"]["tower_length"],
            "units": "m",
        },
        "tower_mass": {"val": modeling_options["costs"]["tower_mass"], "units": "t"},
        "nacelle_mass": {
            "val": modeling_options["costs"]["nacelle_mass"],
            "units": "t",
        },
        "blade_mass": {"val": modeling_options["costs"]["blade_mass"], "units": "t"},
        "turbine_capex": {
            "val": modeling_options["costs"]["turbine_capex"],
            "units": "USD/kW",
        },
        "site_mean_windspeed": {
            "val": modeling_options["costs"]["site_mean_windspeed"],
            "units": "m/s",
        },
        "turbine_rated_windspeed": {
            "val": modeling_options["costs"]["turbine_rated_windspeed"],
            "units": "m/s",
        },
        "commissioning_cost_kW": {
            "val": modeling_options["costs"]["commissioning_cost_kW"],
            "units": "USD/kW",
        },
        "decommissioning_cost_kW": {
            "val": modeling_options["costs"]["decommissioning_cost_kW"],
            "units": "USD/kW",
        },
        "plant_substation_distance": {
            "val": modeling_options["costs"]["plant_substation_distance"],
            "units": "km",
        },
        "interconnection_distance": {
            "val": modeling_options["costs"]["interconnection_distance"],
            "units": "km",
        },
        "site_distance": {
            "val": modeling_options["costs"]["site_distance"],
            "units": "km",
        },
        "site_distance_to_landfall": {
            "val": modeling_options["costs"]["site_distance_to_landfall"],
            "units": "km",
        },
        "port_cost_per_month": {
            "val": modeling_options["costs"]["port_cost_per_month"],
            "units": "USD/month",
        },
        "construction_insurance": {
            "val": modeling_options["costs"]["construction_insurance"],
            "units": "USD/kW",
        },
        "construction_financing": {
            "val": modeling_options["costs"]["construction_financing"],
            "units": "USD/kW",
        },
        "contingency": {
            "val": modeling_options["costs"]["contingency"],
            "units": "USD/kW",
        },
        "site_auction_price": {
            "val": modeling_options["costs"]["site_auction_price"],
            "units": "USD",
        },
        "site_assessment_cost": {
            "val": modeling_options["costs"]["site_assessment_cost"],
            "units": "USD",
        },
        "construction_plan_cost": {
            "val": modeling_options["costs"]["construction_plan_cost"],
            "units": "USD",
        },
        "installation_plan_cost": {
            "val": modeling_options["costs"]["installation_plan_cost"],
            "units": "USD",
        },
        "boem_review_cost": {
            "val": modeling_options["costs"]["boem_review_cost"],
            "units": "USD",
        },
    }

    # Add floating-foundation specific keys if applicable
    if modeling_options["floating"]:
        variable_mapping.update(
            {
                "num_mooring_lines": {
                    "val": modeling_options["costs"]["num_mooring_lines"],
                    "units": None,
                },
                "mooring_line_mass": {
                    "val": modeling_options["costs"]["mooring_line_mass"],
                    "units": "kg",
                },
                "mooring_line_diameter": {
                    "val": modeling_options["costs"]["mooring_line_diameter"],
                    "units": "m",
                },
                "mooring_line_length": {
                    "val": modeling_options["costs"]["mooring_line_length"],
                    "units": "m",
                },
                "anchor_mass": {
                    "val": modeling_options["costs"]["anchor_mass"],
                    "units": "kg",
                },
                "transition_piece_mass": {
                    "val": modeling_options["costs"]["transition_piece_mass"],
                    "units": "t",
                },
                "transition_piece_cost": {
                    "val": modeling_options["costs"]["transition_piece_cost"],
                    "units": "USD",
                },
                "floating_substructure_cost": {
                    "val": modeling_options["costs"]["floating_substructure_cost"],
                    "units": "USD",
                },
            }
        )
    # Add fixed-foundation (mooring) specific keys if applicable
    else:
        variable_mapping.update(
            {
                "monopile_mass": {
                    "val": modeling_options["costs"]["monopile_mass"],
                    "units": "t",
                },
                "monopile_cost": {
                    "val": modeling_options["costs"]["monopile_cost"],
                    "units": "USD",
                },
                "monopile_length": {
                    "val": modeling_options["costs"]["monopile_length"],
                    "units": "m",
                },
                "monopile_diameter": {
                    "val": modeling_options["costs"]["monopile_diameter"],
                    "units": "m",
                },
                "transition_piece_mass": {
                    "val": modeling_options["costs"]["transition_piece_mass"],
                    "units": "t",
                },
                "transition_piece_cost": {
                    "val": modeling_options["costs"]["transition_piece_cost"],
                    "units": "USD",
                },
            }
        )
    # TODO include jacket-type foundation
    # # if jacket
    # prob.set_val(
    #     comp2promotion_map["orbit.orbit.jacket_r_foot"],
    #     modeling_options["turbine"]["costs"]["jacket_r_foot"])
    # prob.set_val(
    #     comp2promotion_map["orbit.orbit.jacket_length"],
    #     modeling_options["turbine"]["costs"]["jacket_length"])
    # prob.set_val(
    #     comp2promotion_map["orbit.orbit.jacket_mass"],
    #     modeling_options["turbine"]["costs"]["jacket_mass"])
    # prob.set_val(
    #     comp2promotion_map["orbit.orbit.jacket_cost"],
    #     modeling_options["turbine"]["costs"]["jacket_cost"])
    # prob.set_val(
    #     comp2promotion_map["orbit.orbit.transition_piece_mass"],
    #     modeling_options["turbine"]["costs"]["transition_piece_mass"])
    # prob.set_val(
    #     comp2promotion_map["orbit.orbit.transition_piece_cost"],
    #     modeling_options["turbine"]["costs"]["transition_piece_cost"])

    return variable_mapping


def FinanceSE_setup_latents(modeling_options):
    """
    A function to set up the FinanceSE latent variables using modeling options.

    Parameters
    ----------
    prob : openmdao.api.Problem
        an OpenMDAO problem for which we want to setup the FinanceSE latent
        variables
    modeling_options : dict
        a modeling options dictionary
    """

    # Define the mapping between OpenMDAO variable names and modeling_options keys
    variable_mapping = {
        "turbine_number": {
            "val": int(modeling_options["layout"]["N_turbines"]),
            "units": None,
        },
        "machine_rating": {
            "val": modeling_options["windIO_plant"]["wind_farm"]["turbine"][
                "performance"
            ]["rated_power"],
            "units": "W",
        },
        "tcc_per_kW": {
            "val": modeling_options["costs"]["tcc_per_kW"],
            "units": "USD/kW",
        },
        "opex_per_kW": {
            "val": modeling_options["costs"]["opex_per_kW"],
            "units": "USD/kW/year",
        },
    }

    return variable_mapping
