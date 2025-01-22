import numpy as np

import openmdao.api as om


class FarmAeroTemplate(om.ExplicitComponent):
    """
    Template component for using a farm aerodynamics model.

    A farm aerodynamics component, based on this template, will compute the
    aerodynamics for a farm with some layout and yaw configuration.

    Options
    -------
    modeling_options : dict
        a modeling options dictionary

    Inputs
    ------
    x_turbines : np.ndarray
        a 1D numpy array indicating the x-dimension locations of the turbines,
        with length `N_turbines`
    y_turbines : np.ndarray
        a 1D numpy array indicating the y-dimension locations of the turbines,
        with length `N_turbines`
    yaw_turbines : np.ndarray
        a numpy array indicating the yaw angle to drive each turbine to with
        respect to the ambient wind direction, with length `N_turbines`

    Outputs
    -------
    None
    """

    def initialize(self):
        """Initialization of OM component."""
        self.options.declare("modeling_options")

    def setup(self):
        """Setup of OM component."""
        # load modeling options
        self.modeling_options = self.options["modeling_options"]
        self.N_turbines = self.modeling_options["farm"]["N_turbines"]

        # grab rotor diameter
        D_rotor = self.modeling_options["turbine"]["geometry"]["diameter_rotor"]

        # set up inputs and outputs for farm layout
        self.add_input(
            "x_turbines",
            np.zeros((self.N_turbines,)),
            units="km",
        )
        self.add_input(
            "y_turbines",
            np.zeros((self.N_turbines,)),
            units="km",
        )
        self.add_input(
            "yaw_turbines",
            np.zeros((self.N_turbines,)),
            units="deg",
        )

    def compute(self, inputs, outputs):
        """
        Computation for the OM component.

        For a template class this is not implemented and raises an error!
        """

        #############################################
        #                                           #
        # IMPLEMENT THE AERODYNAMICS COMPONENT HERE #
        #                                           #
        #############################################

        raise NotImplementedError(
            "This is an abstract class for a derived class to implement!"
        )


class BatchFarmPowerTemplate(FarmAeroTemplate):
    """
    Template component for computing power using a farm aerodynamics model.

    A farm power component, based on this template, will compute the power and
    thrust for a farm composed of a given rotor type.

    Options
    -------
    modeling_options : dict
        a modeling options dictionary (inherited from `FarmAeroTemplate`)
    wind_query : floris.wind_data.WindRose
        a WindQuery objects that specifies the wind conditions that are to be
        computed

    Inputs
    ------
    x_turbines : np.ndarray
        a 1D numpy array indicating the x-dimension locations of the turbines,
        with length `N_turbines` (inherited from `FarmAeroTemplate`)
    y_turbines : np.ndarray
        a 1D numpy array indicating the y-dimension locations of the turbines,
        with length `N_turbines` (inherited from `FarmAeroTemplate`)
    yaw_turbines : np.ndarray
        a numpy array indicating the yaw angle to drive each turbine to with
        respect to the ambient wind direction, with length `N_turbines`
        (inherited from `FarmAeroTemplate`)

    Outputs
    -------
    power_farm : np.ndarray
        an array of the farm power for each of the wind conditions that have
        been queried
    power_turbines : np.ndarray
        an array of the farm power for each of the turbines in the farm across
        all of the conditions that have been queried on the wind rose
        (`N_turbines`, `N_wind_conditions`)
    thrust_turbines : np.ndarray
        an array of the wind turbine thrust for each of the turbines in the farm
        across all of the conditions that have been queried on the wind rose
        (`N_turbines`, `N_wind_conditions`)
    """

    def initialize(self):
        """Initialization of OM component."""
        super().initialize()

        # farm power wind conditions query (not necessarily a full wind rose)
        self.options.declare("wind_query")

    def setup(self):
        """Setup of OM component."""
        super().setup()

        # unpack wind query object
        self.wind_query = self.options["wind_query"]
        self.directions_wind = self.options["wind_query"].get_directions()
        self.speeds_wind = self.options["wind_query"].get_speeds()
        if self.options["wind_query"].get_TIs() is None:
            self.options["wind_query"].set_TI_using_IEC_method()
        self.TIs_wind = self.options["wind_query"].get_TIs()
        self.N_wind_conditions = self.options["wind_query"].N_conditions

        # add the outputs we want for a batched power analysis:
        #   - farm and turbine powers
        #   - turbine thrusts
        self.add_output(
            "power_farm",
            np.zeros((self.N_wind_conditions,)),
            units="MW",
            ref=5.0 * self.N_turbines,  # normalize to 5 MW
        )
        self.add_output(
            "power_turbines",
            np.zeros((self.N_turbines, self.N_wind_conditions)),
            units="MW",
            ref=5.0,  # normalize to 5 MW
        )
        self.add_output(
            "thrust_turbines",
            np.zeros((self.N_turbines, self.N_wind_conditions)),
            units="kN",
            ref=500.0,  # normalize to 500 kN
        )

    def setup_partials(self):
        """Derivative setup for OM component."""
        # the default (but not preferred!) derivatives are FDM
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        """
        Computation for the OM component.

        For a template class this is not implemented and raises an error!
        """
        super().compute(inputs, outputs)

        raise NotImplementedError(
            "This is an abstract class for a derived class to implement!"
        )

        # the following should be set
        outputs["power_farm"] = np.zeros((self.N_wind_conditions,))
        outputs["power_turbines"] = np.zeros((self.N_turbines, self.N_wind_conditions))
        outputs["thrust_turbines"] = np.zeros((self.N_turbines, self.N_wind_conditions))


class FarmAEPTemplate(FarmAeroTemplate):
    """
    A template component for computing power using a farm aerodynamics model.

    A farm power component, based on this template, will compute the power and
    thrust for a farm composed of a given rotor type.

    Options
    -------
    modeling_options : dict
        a modeling options dictionary (inherited from FarmAeroTemplate)
    wind_rose : floris.wind_data.WindRose
        a FLORIS WindRose object that fully specifies the wind conditions on
        which a farm is to be evaluated

    Inputs
    ------
    x_turbines : np.ndarray
        a 1D numpy array indicating the x-dimension locations of the turbines,
        with length `N_turbines` (inherited from `FarmAeroTemplate`)
    y_turbines : np.ndarray
        a 1D numpy array indicating the y-dimension locations of the turbines,
        with length `N_turbines` (inherited from `FarmAeroTemplate`)
    yaw_turbines : np.ndarray
        a numpy array indicating the yaw angle to drive each turbine to with
        respect to the ambient wind direction, with length `N_turbines`
        (inherited from `FarmAeroTemplate`)

    Outputs
    -------
    AEP_farm : float
        the AEP of the farm given by the analysis
    power_farm : np.ndarray
        an array of the farm power for each of the wind conditions that have
        been queried
    power_turbines : np.ndarray
        an array of the farm power for each of the turbines in the farm across
        all of the conditions that have been queried on the wind rose
        (`N_turbines`, `N_wind_conditions`)
    thrust_turbines : np.ndarray
        an array of the wind turbine thrust for each of the turbines in the farm
        across all of the conditions that have been queried on the wind rose
        (`N_turbines`, `N_wind_conditions`)
    """

    def initialize(self):
        """Initialization of OM component."""
        super().initialize()

        # wind conditions for AEP analysis are a FLORIS WindRose
        self.options.declare("wind_rose")  # FLORIS WindRose object

    def setup(self):
        """Setup of OM component."""
        super().setup()

        # unpack FLORIS wind data object
        self.wind_rose = self.options["wind_rose"]
        self.directions_wind, self.speeds_wind, self.TIs_wind, self.pmf_wind, _, _ = (
            self.wind_rose.unpack()
        )
        self.N_wind_conditions = len(self.pmf_wind)

        # add the outputs we want for an AEP analysis:
        #   - AEP estimate
        #   - farm and turbine powers
        #   - turbine thrusts
        self.add_output(
            "AEP_farm",
            0.0,
            units="GW*h",
            ref=0.30 * 8760 * 5.0 * self.N_turbines,  # normalize to 30%CF @ 5 MW
        )
        self.add_output(
            "power_farm",
            np.zeros((self.N_wind_conditions,)),
            units="MW",
            ref=5.0 * self.N_turbines,  # normalize to 5 MW
        )
        self.add_output(
            "power_turbines",
            np.zeros((self.N_turbines, self.N_wind_conditions)),
            units="MW",
            ref=5.0,  # normalize to 5 MW
        )
        self.add_output(
            "thrust_turbines",
            np.zeros((self.N_turbines, self.N_wind_conditions)),
            units="kN",
            ref=500.0,  # normalize to 500 kN
        )
        # ... more outputs can be added here

    def setup_partials(self):
        """Derivative setup for OM component."""
        # the default (but not preferred!) derivatives are FDM
        self.declare_partials("*", "*")

    def compute(self, inputs, outputs):
        """
        Computation for the OM component.

        For a template class this is not implemented and raises an error!
        """

        #############################################
        #                                           #
        # IMPLEMENT THE AERODYNAMICS COMPONENT HERE #
        #                                           #
        #############################################

        raise NotImplementedError(
            "This is an abstract class for a derived class to implement!"
        )

        # the following should be set
        outputs["AEP_farm"] = 0.0
        outputs["power_farm"] = np.zeros((self.N_wind_conditions,))
        outputs["power_turbines"] = np.zeros((self.N_turbines, self.N_wind_conditions))
        outputs["thrust_turbines"] = np.zeros((self.N_turbines, self.N_wind_conditions))
