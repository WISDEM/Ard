import warnings
import numpy as np

import openmdao.api as om
from wisdem.plant_financese.plant_finance import PlantFinance as PlantFinance_orig
from wisdem.landbosse.landbosse_omdao.landbosse import LandBOSSE as LandBOSSE_orig
from wisdem.orbit.api.wisdem import Orbit as Orbit_orig

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
            LandBOSSEArdComp(),
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


class LandBOSSEArdComp(LandBOSSE_orig):
    """
    Wrapper for WISDEM's LandBOSSE BOS calculators.

    A thin wrapper of `wisdem.landbosse.landbosse_omdao.landbosse.LandBOSSE`
    that traps warning messages that are recognized not to be issues.

    See: https://github.com/WISDEM/LandBOSSE
    """

    def setup(self):
        """Setup of OM component."""
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with warnings.catch_warnings():
            return super().setup()

    def setup_partials(self):
        """Derivative setup for OM component."""

        # finite difference WISDEM tools for gradients
        self.declare_partials(
            [
                "turbine_spacing_rotor_diameters",
                "row_spacing_rotor_diameters",
            ],
            [
                "bos_capex_kW",
                "total_capex",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """Computation for the OM compoent."""
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with warnings.catch_warnings():
            return super().compute(inputs, outputs, discrete_inputs, discrete_outputs)


class ORBIT(Orbit_orig):
    """
    Wrapper for WISDEM's ORBIT offshore BOS calculators.

    A thin wrapper of `wisdem.orbit.api.wisdem`
    that traps warning messages that are recognized not to be issues.

    See: https://github.com/WISDEM/ORBIT
    """

    def setup(self):
        """Setup of OM component."""
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with warnings.catch_warnings():
            return super().setup()

    def setup_partials(self):
        """Derivative setup for OM component."""

        # finite difference WISDEM tools for gradients
        self.declare_partials(
            [
                "turbine_spacing_rotor_diameters",
                "row_spacing_rotor_diameters",
            ],
            [
                "bos_capex_kW",
                "total_capex",
                "installation_capex",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """Computation for the OM compoent."""
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with warnings.catch_warnings():
            return super().compute(inputs, outputs, discrete_inputs, discrete_outputs)


class PlantFinance(PlantFinance_orig):
    """
    Wrapper for WISDEM's PlantFinanceSE calculators.

    A thin wrapper of `wisdem.plant_financese.plant_finance.PlantFinance` that
    traps warning messages that are recognized not to be issues.

    See: https://github.com/WISDEM/WISDEM/tree/master/wisdem/plant_financese
    """

    def setup(self):
        """Setup of OM component."""
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with warnings.catch_warnings():
            return super().setup()

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """Computation for the OM compoent."""
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with warnings.catch_warnings():
            return super().compute(inputs, outputs, discrete_inputs, discrete_outputs)


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


def LandBOSSE_setup_latents(prob, modeling_options: dict) -> None:
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

    if any(key in modeling_options["turbine"]["costs"] for key in offshore_fixed_keys):

        variable_mapping = {
            "num_turbines": modeling_options["farm"]["N_turbines"],
            "turbine_rating_MW": modeling_options["turbine"]["nameplate"]["power_rated"]
            * 1.0e3,
            "hub_height_meters": modeling_options["turbine"]["geometry"]["height_hub"],
            "rotor_diameter_m": modeling_options["turbine"]["geometry"][
                "diameter_rotor"
            ],
            "number_of_blades": modeling_options["turbine"]["geometry"]["num_blades"],
            "tower_mass": modeling_options["turbine"]["costs"]["tower_mass"],
            "nacelle_mass": modeling_options["turbine"]["costs"]["nacelle_mass"],
            "blade_mass": modeling_options["turbine"]["costs"]["blade_mass"],
            "commissioning_cost_kW": modeling_options["turbine"]["costs"][
                "commissioning_cost_kW"
            ],
            "decommissioning_cost_kW": modeling_options["turbine"]["costs"][
                "decommissioning_cost_kW"
            ],
            # Offshore fixed-specific keys
            "monopile_mass": modeling_options["turbine"]["costs"]["monopile_mass"],
            "monopile_cost": modeling_options["turbine"]["costs"]["monopile_cost"],
        }

    elif any(
        key in modeling_options["turbine"]["costs"] for key in offshore_floating_keys
    ):
        variable_mapping = {
            "num_turbines": modeling_options["farm"]["N_turbines"],
            "turbine_rating_MW": modeling_options["turbine"]["nameplate"]["power_rated"]
            * 1.0e3,
            "hub_height_meters": modeling_options["turbine"]["geometry"]["height_hub"],
            # "wind_shear_exponent": modeling_options["turbine"]["costs"]["wind_shear_exponent"],
            "rotor_diameter_m": modeling_options["turbine"]["geometry"][
                "diameter_rotor"
            ],
            "number_of_blades": modeling_options["turbine"]["geometry"]["num_blades"],
            "tower_mass": modeling_options["turbine"]["costs"]["tower_mass"],
            "nacelle_mass": modeling_options["turbine"]["costs"]["nacelle_mass"],
            "blade_mass": modeling_options["turbine"]["costs"]["blade_mass"],
            "commissioning_cost_kW": modeling_options["turbine"]["costs"][
                "commissioning_cost_kW"
            ],
            "decommissioning_cost_kW": modeling_options["turbine"]["costs"][
                "decommissioning_cost_kW"
            ],
            # Offshore floating-specific keys
            "num_mooring_lines": modeling_options["turbine"]["costs"][
                "num_mooring_lines"
            ],
            "mooring_line_mass": modeling_options["turbine"]["costs"][
                "mooring_line_mass"
            ],
            "mooring_line_diameter": modeling_options["turbine"]["costs"][
                "mooring_line_diameter"
            ],
            "mooring_line_length": modeling_options["turbine"]["costs"][
                "mooring_line_length"
            ],
            "anchor_mass": modeling_options["turbine"]["costs"]["anchor_mass"],
            "floating_substructure_cost": modeling_options["turbine"]["costs"][
                "floating_substructure_cost"
            ],
        }
    else:
        # this is the standard mapping for using LandBOSSE, since typically ORIBIT should
        # be used for BOS costs for offshore systems.
        variable_mapping = {
            "num_turbines": modeling_options["farm"]["N_turbines"],
            "turbine_rating_MW": modeling_options["turbine"]["nameplate"]["power_rated"]
            * 1.0e3,
            "hub_height_meters": modeling_options["turbine"]["geometry"]["height_hub"],
            "wind_shear_exponent": modeling_options["turbine"]["costs"][
                "wind_shear_exponent"
            ],
            "rotor_diameter_m": modeling_options["turbine"]["geometry"][
                "diameter_rotor"
            ],
            "number_of_blades": modeling_options["turbine"]["geometry"]["num_blades"],
            "rated_thrust_N": modeling_options["turbine"]["costs"]["rated_thrust_N"],
            "gust_velocity_m_per_s": modeling_options["turbine"]["costs"][
                "gust_velocity_m_per_s"
            ],
            "blade_surface_area": modeling_options["turbine"]["costs"][
                "blade_surface_area"
            ],
            "tower_mass": modeling_options["turbine"]["costs"]["tower_mass"],
            "nacelle_mass": modeling_options["turbine"]["costs"]["nacelle_mass"],
            "hub_mass": modeling_options["turbine"]["costs"]["hub_mass"],
            "blade_mass": modeling_options["turbine"]["costs"]["blade_mass"],
            "foundation_height": modeling_options["turbine"]["costs"][
                "foundation_height"
            ],
            "commissioning_cost_kW": modeling_options["turbine"]["costs"][
                "commissioning_cost_kW"
            ],
            "decommissioning_cost_kW": modeling_options["turbine"]["costs"][
                "decommissioning_cost_kW"
            ],
            "trench_len_to_substation_km": modeling_options["turbine"]["costs"][
                "trench_len_to_substation_km"
            ],
            "distance_to_interconnect_mi": modeling_options["turbine"]["costs"][
                "distance_to_interconnect_mi"
            ],
            "interconnect_voltage_kV": modeling_options["turbine"]["costs"][
                "interconnect_voltage_kV"
            ],
        }

    set_values(prob, variable_map=variable_mapping)


def ORBIT_setup_latents(prob, modeling_options: dict) -> None:
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

    # Define the mapping between OpenMDAO variable names and modeling_options keys
    variable_mapping = {
        "turbine_rating": modeling_options["turbine"]["nameplate"]["power_rated"],
        "site_depth": modeling_options["site_depth"],
        "number_of_turbines": modeling_options["farm"]["N_turbines"],
        "number_of_blades": modeling_options["turbine"]["geometry"]["num_blades"],
        "hub_height": modeling_options["turbine"]["geometry"]["height_hub"],
        "turbine_rotor_diameter": modeling_options["turbine"]["geometry"][
            "diameter_rotor"
        ],
        "tower_length": modeling_options["turbine"]["geometry"]["tower_length"],
        "tower_mass": modeling_options["turbine"]["costs"]["tower_mass"],
        "nacelle_mass": modeling_options["turbine"]["costs"]["nacelle_mass"],
        "blade_mass": modeling_options["turbine"]["costs"]["blade_mass"],
        "turbine_capex": modeling_options["turbine"]["costs"]["turbine_capex"],
        "site_mean_windspeed": modeling_options["turbine"]["costs"][
            "site_mean_windspeed"
        ],
        "turbine_rated_windspeed": modeling_options["turbine"]["costs"][
            "turbine_rated_windspeed"
        ],
        "commissioning_cost_kW": modeling_options["turbine"]["costs"][
            "commissioning_cost_kW"
        ],
        "decommissioning_cost_kW": modeling_options["turbine"]["costs"][
            "decommissioning_cost_kW"
        ],
        "plant_substation_distance": modeling_options["turbine"]["costs"][
            "plant_substation_distance"
        ],
        "interconnection_distance": modeling_options["turbine"]["costs"][
            "interconnection_distance"
        ],
        "site_distance": modeling_options["turbine"]["costs"]["site_distance"],
        "site_distance_to_landfall": modeling_options["turbine"]["costs"][
            "site_distance_to_landfall"
        ],
        "port_cost_per_month": modeling_options["turbine"]["costs"][
            "port_cost_per_month"
        ],
        "construction_insurance": modeling_options["turbine"]["costs"][
            "construction_insurance"
        ],
        "construction_financing": modeling_options["turbine"]["costs"][
            "construction_financing"
        ],
        "contingency": modeling_options["turbine"]["costs"]["contingency"],
        "site_auction_price": modeling_options["turbine"]["costs"][
            "site_auction_price"
        ],
        "site_assessment_cost": modeling_options["turbine"]["costs"][
            "site_assessment_cost"
        ],
        "construction_plan_cost": modeling_options["turbine"]["costs"][
            "construction_plan_cost"
        ],
        "installation_plan_cost": modeling_options["turbine"]["costs"][
            "installation_plan_cost"
        ],
        "boem_review_cost": modeling_options["turbine"]["costs"]["boem_review_cost"],
    }

    # Add floating-foundation specific keys if applicable
    if modeling_options["floating"]:
        variable_mapping.update(
            {
                "num_mooring_lines": modeling_options["turbine"]["costs"][
                    "num_mooring_lines"
                ],
                "mooring_line_mass": modeling_options["turbine"]["costs"][
                    "mooring_line_mass"
                ],
                "mooring_line_diameter": modeling_options["turbine"]["costs"][
                    "mooring_line_diameter"
                ],
                "mooring_line_length": modeling_options["turbine"]["costs"][
                    "mooring_line_length"
                ],
                "anchor_mass": modeling_options["turbine"]["costs"]["anchor_mass"],
                "transition_piece_mass": modeling_options["turbine"]["costs"][
                    "transition_piece_mass"
                ],
                "transition_piece_cost": modeling_options["turbine"]["costs"][
                    "transition_piece_cost"
                ],
                "floating_substructure_cost": modeling_options["turbine"]["costs"][
                    "floating_substructure_cost"
                ],
            }
        )
    # Add fixed-foundation (mooring) specific keys if applicable
    else:
        variable_mapping.update(
            {
                "monopile_mass": modeling_options["turbine"]["costs"]["monopile_mass"],
                "monopile_cost": modeling_options["turbine"]["costs"]["monopile_cost"],
                "monopile_length": modeling_options["turbine"]["geometry"][
                    "monopile_length"
                ],
                "monopile_diameter": modeling_options["turbine"]["geometry"][
                    "monopile_diameter"
                ],
                "transition_piece_mass": modeling_options["turbine"]["costs"][
                    "transition_piece_mass"
                ],
                "transition_piece_cost": modeling_options["turbine"]["costs"][
                    "transition_piece_cost"
                ],
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

    set_values(prob, variable_map=variable_mapping)


def FinanceSE_setup_latents(prob, modeling_options):
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
        "turbine_number": int(modeling_options["farm"]["N_turbines"]),
        "machine_rating": modeling_options["turbine"]["nameplate"]["power_rated"]
        * 1.0e3,
        "tcc_per_kW": modeling_options["turbine"]["costs"]["tcc_per_kW"],
        "opex_per_kW": modeling_options["turbine"]["costs"]["opex_per_kW"],
    }

    set_values(prob, variable_map=variable_mapping)


def set_values(prob, variable_map: dict) -> None:
    """
    Set values in an OpenMDAO problem based on a mapping of variable names to values.

    This function dynamically maps core variable names to their promoted names in the
    OpenMDAO problem and sets their values using the provided `variable_map`.

    Parameters
    ----------
    prob : openmdao.api.Problem
        The OpenMDAO problem instance where the variables are to be set.
    variable_map : dict
        A dictionary mapping core variable names (keys) to their corresponding values
        (values) that need to be set in the OpenMDAO problem.

    Returns
    -------
    None
    """

    # Get a map from the component variables to the promotion variables
    promotion_map = {
        v[0].split(".")[-1]: v[-1]["prom_name"]
        for v in prob.model.list_vars(val=False, out_stream=None)
    }

    # Iterate over the mapping and set values in the OpenMDAO problem
    for full_name in promotion_map:
        prom_name = promotion_map[full_name]
        core_name = prom_name.split(".")[-1]
        if core_name in promotion_map:
            try:
                prob.set_val(prom_name, variable_map[core_name])
            except:
                print(
                    f"{core_name} not provided in turbine input, using WISDEM default"
                )
