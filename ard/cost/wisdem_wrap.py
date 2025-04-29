import warnings

import openmdao.api as om
from wisdem.plant_financese.plant_finance import PlantFinance as PlantFinance_orig
from wisdem.landbosse.landbosse_omdao.landbosse import LandBOSSE as LandBOSSE_orig
from wisdem.orbit.api.wisdem import Orbit as Orbit_orig


class LandBOSSE(LandBOSSE_orig):
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


def LandBOSSE_setup_latents(prob, modeling_options):
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

    # get a map from the component variables to the promotion variables
    comp2promotion_map = {
        v[0]: v[-1]["prom_name"]
        for v in prob.model.list_vars(val=False, out_stream=None)
    }

    # set latent/non-design inputs to LandBOSSE using values in modeling_options
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.num_turbines"],
        modeling_options["farm"]["N_turbines"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.turbine_rating_MW"],
        modeling_options["turbine"]["nameplate"]["power_rated"] * 1.0e3,
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.hub_height_meters"],
        modeling_options["turbine"]["geometry"]["height_hub"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.wind_shear_exponent"],
        modeling_options["turbine"]["costs"]["wind_shear_exponent"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.rotor_diameter_m"],
        modeling_options["turbine"]["geometry"]["diameter_rotor"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.number_of_blades"],
        modeling_options["turbine"]["geometry"]["num_blades"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.rated_thrust_N"],
        modeling_options["turbine"]["costs"]["rated_thrust_N"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.gust_velocity_m_per_s"],
        modeling_options["turbine"]["costs"]["gust_velocity_m_per_s"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.blade_surface_area"],
        modeling_options["turbine"]["costs"]["blade_surface_area"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.tower_mass"],
        modeling_options["turbine"]["costs"]["tower_mass"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.nacelle_mass"],
        modeling_options["turbine"]["costs"]["nacelle_mass"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.hub_mass"],
        modeling_options["turbine"]["costs"]["hub_mass"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.blade_mass"],
        modeling_options["turbine"]["costs"]["blade_mass"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.foundation_height"],
        modeling_options["turbine"]["costs"]["foundation_height"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.commissioning_cost_kW"],
        modeling_options["turbine"]["costs"]["commissioning_cost_kW"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.decommissioning_cost_kW"],
        modeling_options["turbine"]["costs"]["decommissioning_cost_kW"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.trench_len_to_substation_km"],
        modeling_options["turbine"]["costs"]["trench_len_to_substation_km"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.distance_to_interconnect_mi"],
        modeling_options["turbine"]["costs"]["distance_to_interconnect_mi"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.interconnect_voltage_kV"],
        modeling_options["turbine"]["costs"]["interconnect_voltage_kV"],
    )


def ORBIT_setup_latents(prob, modeling_options):
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

    # get a map from the component variables to the promotion variables
    comp2promotion_map = {
        v[0]: v[-1]["prom_name"]
        for v in prob.model.list_vars(val=False, out_stream=None)
    }

    # set latent/non-design inputs to ORBIT using values in modeling_options
    prob.set_val(
        comp2promotion_map["orbit.orbit.turbine_rating"],
        modeling_options["turbine"]["nameplate"]["power_rated"],
    )
    prob.set_val(
        comp2promotion_map["orbit.orbit.site_depth"],
        modeling_options["site_depth"],
    )
    prob.set_val(
        comp2promotion_map["orbit.orbit.number_of_turbines"],
        modeling_options["farm"]["N_turbines"],
    )
    prob.set_val(
        comp2promotion_map["orbit.orbit.number_of_blades"],
        modeling_options["turbine"]["geometry"]["num_blades"],
    )
    prob.set_val(
        comp2promotion_map["orbit.orbit.hub_height"],
        modeling_options["turbine"]["geometry"]["height_hub"],
    )
    prob.set_val(
        comp2promotion_map["orbit.orbit.turbine_rotor_diameter"],
        modeling_options["turbine"]["geometry"]["diameter_rotor"],
    )
    prob.set_val(
        comp2promotion_map["orbit.orbit.tower_length"],
        modeling_options["turbine"]["geometry"]["tower_length"],
    )
    prob.set_val(
        comp2promotion_map["orbit.orbit.tower_mass"],
        modeling_options["turbine"]["costs"]["tower_mass"],
    )
    prob.set_val(
        comp2promotion_map["orbit.orbit.nacelle_mass"],
        modeling_options["turbine"]["costs"]["nacelle_mass"],
    )
    prob.set_val(
        comp2promotion_map["orbit.orbit.blade_mass"],
        modeling_options["turbine"]["costs"]["blade_mass"],
    )
    prob.set_val(
        comp2promotion_map["orbit.orbit.turbine_capex"],
        modeling_options["turbine"]["costs"]["turbine_capex"],
    )
    prob.set_val(
        comp2promotion_map["orbit.orbit.site_mean_windspeed"],
        modeling_options["turbine"]["costs"]["site_mean_windspeed"],
    )
    prob.set_val(
        comp2promotion_map["orbit.orbit.turbine_rated_windspeed"],
        modeling_options["turbine"]["costs"]["turbine_rated_windspeed"],
    )
    prob.set_val(
        comp2promotion_map["orbit.orbit.commissioning_cost_kW"],
        modeling_options["turbine"]["costs"]["commissioning_cost_kW"],
    )
    prob.set_val(
        comp2promotion_map["orbit.orbit.decommissioning_cost_kW"],
        modeling_options["turbine"]["costs"]["decommissioning_cost_kW"],
    )
    prob.set_val(
        comp2promotion_map["orbit.orbit.plant_substation_distance"],
        modeling_options["turbine"]["costs"]["plant_substation_distance"],
    )
    prob.set_val(
        comp2promotion_map["orbit.orbit.interconnection_distance"],
        modeling_options["turbine"]["costs"]["interconnection_distance"],
    )
    prob.set_val(
        comp2promotion_map["orbit.orbit.site_distance"],
        modeling_options["turbine"]["costs"]["site_distance"],
    )
    prob.set_val(
        comp2promotion_map["orbit.orbit.site_distance_to_landfall"],
        modeling_options["turbine"]["costs"]["site_distance_to_landfall"],
    )
    prob.set_val(
        comp2promotion_map["orbit.orbit.port_cost_per_month"],
        modeling_options["turbine"]["costs"]["port_cost_per_month"],
    )
    prob.set_val(
        comp2promotion_map["orbit.orbit.construction_insurance"],
        modeling_options["turbine"]["costs"]["construction_insurance"],
    )
    prob.set_val(
        comp2promotion_map["orbit.orbit.construction_financing"],
        modeling_options["turbine"]["costs"]["construction_financing"],
    )
    prob.set_val(
        comp2promotion_map["orbit.orbit.contingency"],
        modeling_options["turbine"]["costs"]["contingency"],
    )
    prob.set_val(
        comp2promotion_map["orbit.orbit.site_auction_price"],
        modeling_options["turbine"]["costs"]["site_auction_price"],
    )
    prob.set_val(
        comp2promotion_map["orbit.orbit.site_assessment_cost"],
        modeling_options["turbine"]["costs"]["site_assessment_cost"],
    )
    prob.set_val(
        comp2promotion_map["orbit.orbit.construction_plan_cost"],
        modeling_options["turbine"]["costs"]["construction_plan_cost"],
    )
    prob.set_val(
        comp2promotion_map["orbit.orbit.installation_plan_cost"],
        modeling_options["turbine"]["costs"]["installation_plan_cost"],
    )
    prob.set_val(
        comp2promotion_map["orbit.orbit.boem_review_cost"],
        modeling_options["turbine"]["costs"]["boem_review_cost"],
    )

    if modeling_options["floating"]:
        prob.set_val(
            comp2promotion_map["orbit.orbit.num_mooring_lines"],
            modeling_options["turbine"]["costs"]["num_mooring_lines"],
        )
        prob.set_val(
            comp2promotion_map["orbit.orbit.mooring_line_mass"],
            modeling_options["turbine"]["costs"]["mooring_line_mass"],
        )
        prob.set_val(
            comp2promotion_map["orbit.orbit.mooring_line_diameter"],
            modeling_options["turbine"]["costs"]["mooring_line_diameter"],
        )
        prob.set_val(
            comp2promotion_map["orbit.orbit.mooring_line_length"],
            modeling_options["turbine"]["costs"]["mooring_line_length"],
        )
        prob.set_val(
            comp2promotion_map["orbit.orbit.anchor_mass"],
            modeling_options["turbine"]["costs"]["anchor_mass"],
        )
        prob.set_val(
            comp2promotion_map["orbit.orbit.transition_piece_mass"],
            modeling_options["turbine"]["costs"]["transition_piece_mass"],
        )
        prob.set_val(
            comp2promotion_map["orbit.orbit.transition_piece_cost"],
            modeling_options["turbine"]["costs"]["transition_piece_cost"],
        )
        prob.set_val(
            comp2promotion_map["orbit.orbit.floating_substructure_cost"],
            modeling_options["turbine"]["costs"]["floating_substructure_cost"],
        )
    else:
        prob.set_val(
            comp2promotion_map["orbit.orbit.monopile_mass"],
            modeling_options["turbine"]["costs"]["monopile_mass"],
        )
        prob.set_val(
            comp2promotion_map["orbit.orbit.monopile_cost"],
            modeling_options["turbine"]["costs"]["monopile_cost"],
        )
        prob.set_val(
            comp2promotion_map["orbit.orbit.monopile_length"],
            modeling_options["turbine"]["geometry"]["monopile_length"],
        )
        prob.set_val(
            comp2promotion_map["orbit.orbit.monopile_diameter"],
            modeling_options["turbine"]["geometry"]["monopile_diameter"],
        )
        prob.set_val(
            comp2promotion_map["orbit.orbit.transition_piece_mass"],
            modeling_options["turbine"]["costs"]["transition_piece_mass"],
        )
        prob.set_val(
            comp2promotion_map["orbit.orbit.transition_piece_cost"],
            modeling_options["turbine"]["costs"]["transition_piece_cost"],
        )
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

    # get a map from the component variables to the promotion variables
    comp2promotion_map = {
        v[0]: v[-1]["prom_name"]
        for v in prob.model.list_vars(val=False, out_stream=None)
    }

    # inputs to PlantFinanceSE
    prob.set_val(
        comp2promotion_map["financese.turbine_number"],
        int(modeling_options["farm"]["N_turbines"]),
    )
    prob.set_val(
        comp2promotion_map["financese.machine_rating"],
        modeling_options["turbine"]["nameplate"]["power_rated"] * 1.0e3,
    )
    prob.set_val(
        comp2promotion_map["financese.tcc_per_kW"],
        modeling_options["turbine"]["costs"]["tcc_per_kW"],
    )
    prob.set_val(
        comp2promotion_map["financese.opex_per_kW"],
        modeling_options["turbine"]["costs"]["opex_per_kW"],
    )
