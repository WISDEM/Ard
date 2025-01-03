import warnings

import openmdao.api as om
from wisdem.plant_financese.plant_finance import PlantFinance as PlantFinance_orig
from wisdem.landbosse.landbosse_omdao.landbosse import LandBOSSE as LandBOSSE_orig


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
        self.add_discrete_input("turbine_number", 1)
        self.add_output("tcc", 0.0, units="USD")

    def setup_partials(self):
        """Derivative setup for OM component."""
        # complex step for simple gradients
        self.declare_partials(
            ["tcc"],
            ["machine_rating", "tcc_per_kW", "offset_tcc_per_kW"],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """Computation for the OM compoent."""
        # Unpack parameters
        t_rating = inputs["machine_rating"]
        n_turbine = discrete_inputs["turbine_number"]
        tcc_per_kW = inputs["tcc_per_kW"] + inputs["offset_tcc_per_kW"]
        outputs["tcc"] = n_turbine * tcc_per_kW * t_rating

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["tcc", "machine_rating"] = discrete_inputs["turbine_number"] * (
            inputs["tcc_per_kW"] + inputs["offset_tcc_per_kW"]
        )
        partials["tcc", "tcc_per_kW"] = (
            discrete_inputs["turbine_number"] * inputs["machine_rating"]
        )
        partials["tcc", "offset_tcc_per_kW"] = (
            discrete_inputs["turbine_number"] * inputs["machine_rating"]
        )


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
        self.add_discrete_input("turbine_number", 1)
        self.add_output("opex", 0.0, units="USD/yr")

    def setup_partials(self):
        """Derivative setup for OM component."""
        # complex step for simple gradients
        self.declare_partials("opex", ["machine_rating", "opex_per_kW"], method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """Computation for the OM compoent."""
        # Unpack parameters
        t_rating = inputs["machine_rating"]
        n_turbine = discrete_inputs["turbine_number"]
        opex_per_kW = inputs["opex_per_kW"]
        outputs["opex"] = n_turbine * opex_per_kW * t_rating

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        # Unpack parameters
        partials["opex", "machine_rating"] = (
            discrete_inputs["turbine_number"] * inputs["opex_per_kW"]
        )
        partials["opex", "opex_per_kW"] = (
            discrete_inputs["turbine_number"] * inputs["machine_rating"]
        )


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
        comp2promotion_map["landbosse.landbosse.commissioning_pct"],
        modeling_options["turbine"]["costs"]["commissioning_pct"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.decommissioning_pct"],
        modeling_options["turbine"]["costs"]["decommissioning_pct"],
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


def FinanceSE_setup_latents(prob, modeling_options):
    """
    A function to set up the FinanceSE latent variables using modeling options.

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
