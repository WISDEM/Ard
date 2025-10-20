import numpy as np

import ard.farm_aero.templates as templates


class PlaceholderBatchPower(templates.BatchFarmPowerTemplate):
    """
    Placeholder component for computing power assuming nameplate power at all
    times.

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
        super().initialize()  # run super class script first!

    def setup(self):
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
            units="W",
        )
        if self.return_turbine_output:
            self.add_output(
                "power_turbines",
                np.zeros((self.N_turbines, self.N_wind_conditions)),
                units="W",
            )
            self.add_output(
                "thrust_turbines",
                np.zeros((self.N_turbines, self.N_wind_conditions)),
                units="N",
            )

    def setup_partials(self):
        """Derivative setup for OM component."""
        # the default (but not preferred!) derivatives are FDM
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        """
        Computation for the OM component.
        """

        # the following should be set
        outputs["power_farm"] = (
            self.N_turbines
            * self.modeling_options["turbine"]["nameplate"]["power_rated"]
            * np.ones((self.N_wind_conditions,))
        )
        outputs["power_turbines"] = self.modeling_options["turbine"]["nameplate"][
            "power_rated"
        ] * np.ones((self.N_turbines, self.N_wind_conditions))
        outputs["thrust_turbines"] = np.zeros((self.N_turbines, self.N_wind_conditions))


class PlaceholderAEP(templates.FarmAEPTemplate):
    """
    Placeholder component for computing AEP assuming nameplate power at all
    times.

    Options
    -------
    modeling_options : dict
        a modeling options dictionary (inherited via
        `templates.FarmAEPTemplate`)
    wind_query : floris.wind_data.WindRose
        a WindQuery objects that specifies the wind conditions that are to be
        computed (inherited from `templates.FarmAEPTemplate`)

    Inputs
    ------
    x_turbines : np.ndarray
        a 1D numpy array indicating the x-dimension locations of the turbines,
        with length `N_turbines` (inherited via `templates.FarmAEPTemplate`)
    y_turbines : np.ndarray
        a 1D numpy array indicating the y-dimension locations of the turbines,
        with length `N_turbines` (inherited via `templates.FarmAEPTemplate`)
    yaw_turbines : np.ndarray
        a numpy array indicating the yaw angle to drive each turbine to with
        respect to the ambient wind direction, with length `N_turbines`
        (inherited via `templates.FarmAEPTemplate`)

    Outputs
    -------
    AEP_farm : float
        the AEP of the farm given by the analysis (inherited from
        `templates.FarmAEPTemplate`)
    power_farm : np.ndarray
        an array of the farm power for each of the wind conditions that have
        been queried (inherited from `templates.FarmAEPTemplate`)
    power_turbines : np.ndarray
        an array of the farm power for each of the turbines in the farm across
        all of the conditions that have been queried on the wind rose
        (`N_turbines`, `N_wind_conditions`) (inherited from
        `templates.FarmAEPTemplate`)
    thrust_turbines : np.ndarray
        an array of the wind turbine thrust for each of the turbines in the farm
        across all of the conditions that have been queried on the wind rose
        (`N_turbines`, `N_wind_conditions`) (inherited from
        `templates.FarmAEPTemplate`)
    """

    def initialize(self):
        super().initialize()  # run super class script first!

    def setup(self):
        super().setup()  # run super class script first!

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):

        # the following should be set
        outputs["AEP_farm"] = (
            self.N_turbines
            * self.modeling_options["turbine"]["nameplate"]["power_rated"]
            * 1.0e6
            * 8760.0
        )
        outputs["power_farm"] = (
            self.N_turbines
            * self.modeling_options["turbine"]["nameplate"]["power_rated"]
            * 1.0e6
            * np.ones((self.N_wind_conditions,))
        )
        outputs["power_turbines"] = (
            self.modeling_options["turbine"]["nameplate"]["power_rated"]
            * 1.0e6
            * np.ones((self.N_turbines, self.N_wind_conditions))
        )
        outputs["thrust_turbines"] = np.zeros((self.N_turbines, self.N_wind_conditions))
