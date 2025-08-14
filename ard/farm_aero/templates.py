from pathlib import Path

import numpy as np

import openmdao.api as om

import floris


def create_windresource_from_windIO(
    windIOdict: dict,
    resource_type: str = None,  # ["probability", "timeseries", "weibull_sector"]
):
    """
    break out the windIO wind resource specification
    """

    assert "site" in windIOdict  # make sure the site is specified
    assert "energy_resource" in windIOdict["site"]
    assert "wind_resource" in windIOdict["site"]["energy_resource"]

    # get the wind resource specification out of the dictionary
    wind_resource = windIOdict["site"]["energy_resource"]["wind_resource"]

    # figure out the case in play
    fields_wind_resource = wind_resource.keys()
    case_probability_based = "probability" in fields_wind_resource
    case_weibull_based = all(
        val in fields_wind_resource
        for val in ["weibull_a", "weibull_k", "weibull_probability"]
    )
    case_timeseries_based = all(
        val in fields_wind_resource for val in ["time", "wind_speed", "wind_direction"]
    )

    if case_weibull_based and not (case_probability_based or case_timeseries_based):
        if resource_type is not None and resource_type != "weibull_sector":
            raise ValueError(
                f"Attempted to load {resource_type}-type wind resource and "
                "only weibull_sector was found."
            )
        raise NotImplementedError(
            "Sector Weibull wind resource has not yet been implemented for FLORIS."
        )

    wind_resource_representation = None

    if case_probability_based and case_timeseries_based:
        raise ValueError(
            "Both probability-based and time-series-based wind resource "
            "specifications have been found; this is ambiguous."
        )
    elif case_probability_based:
        if resource_type is not None and resource_type != "probability":
            raise ValueError(
                f"Attempted to load {resource_type}-type wind resource and "
                "only probability was found."
            )

        # extract key variables
        wind_directions = np.array(
            wind_resource["wind_direction"]["data"]
            if "data" in wind_resource["wind_direction"]
            else wind_resource["wind_direction"]
        )
        wind_speeds = np.array(
            wind_resource["wind_speed"]["data"]
            if "data" in wind_resource["wind_speed"]
            else wind_resource["wind_speed"]
        )
        probabilities = np.array(wind_resource["probability"]["data"])
        turbulence_intensities = (
            np.array(wind_resource["turbulence_instensity"]["data"])
            if "turbulence_instensity" in wind_resource
            else 0.06
        )

        # create FLORIS representation
        wind_resource_representation = floris.WindRose(
            wind_directions=wind_directions,
            wind_speeds=wind_speeds,
            freq_table=probabilities,
            ti_table=turbulence_intensities,
        )
        # stash some metadata for the wind resource
        wind_resource_representation.reference_height = (
            wind_resource["reference_height"]
            if "reference_height" in wind_resource
            else None
        )

        return wind_resource_representation

    elif case_timeseries_based:
        if resource_type is not None and resource_type != "time-series":
            raise ValueError(
                f"Attempted to load {resource_type}-type wind resource and "
                "only time-series was found."
            )

        wind_directions = np.array(wind_resource["wind_direction"]["data"])
        wind_speeds = np.array(wind_resource["wind_speed"]["data"])
        turbulence_intensities = (
            (
                np.array(wind_resource["turbulence_instensity"]["data"])
                if "turbulence_instensity" in wind_resource
                else None
            ),
        )

        wind_resource_representation = floris.TimeSeries(
            wind_directions=wind_directions,
            wind_speeds=wind_speeds,
            turbulence_intensities=turbulence_intensities,
        )
        # stash some metadata for the wind resource
        wind_resource_representation.reference_height = (
            wind_resource["reference_height"]
            if "reference_height" in wind_resource
            else None
        )
        wind_resource_representation.time = (
            wind_resource["time"] if "time" in wind_resource else None
        )

        raise NotImplementedError(
            "Time-series-based wind resource implementation is in progress..."
        )

        return wind_resource_representation

    else:
        raise ValueError(
            "You shouldn't have ended here, try validating the windIO yaml file."
        )


class FarmAeroTemplate(om.ExplicitComponent):
    """
    Template component for using a farm aerodynamics model.

    A farm aerodynamics component, based on this template, will compute the
    aerodynamics for a farm with some layout and yaw configuration.

    Options
    -------
    modeling_options : dict
        a modeling options dictionary
    windIO : dict
        a windIO file dictionary

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
        self.options.declare("windIO_plant")
        self.options.declare("data_path")

    def setup(self):
        """Setup of OM component."""

        # load modeling options
        self.modeling_options = self.options["modeling_options"]
        self.windIO = self.options["windIO_plant"]
        self.N_turbines = self.modeling_options["layout"]["N_turbines"]

        # set up inputs and outputs for farm layout
        self.add_input("x_turbines", np.zeros((self.N_turbines,)), units="m")
        self.add_input("y_turbines", np.zeros((self.N_turbines,)), units="m")
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
    windIO : dict
        a windIO file dictionary

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

    def setup(self):
        """Setup of OM component."""
        super().setup()

        # unpack wind query object
        self.wind_query = create_windresource_from_windIO(
            self.windIO,
            "timeseries",
        )
        self.directions_wind = self.wind_query.get_directions()
        self.speeds_wind = self.wind_query.get_speeds()
        if self.wind_query.get_TIs() is None:
            self.wind_query.set_TI_using_IEC_method()
        self.TIs_wind = self.wind_query.get_TIs()
        self.N_wind_conditions = self.wind_query.N_conditions

        # add the outputs we want for a batched power analysis:
        #   - farm and turbine powers
        #   - turbine thrusts
        self.add_output(
            "power_farm",
            np.zeros((self.N_wind_conditions,)),
            units="W",
        )
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
        self.declare_partials("*", "*", method="fd")

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
    windIO : dict
        a windIO file dictionary (inherited from FarmAEPTemplate)
    data_path: str
        absolute path to data directory

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
        self.options.declare("data_path", default=None)

    def setup(self):
        """Setup of OM component."""
        super().setup()
        data_path = str(self.options["data_path"])

        self.wind_query = create_windresource_from_windIO(
            self.windIO,
            "probability",
        )

        if data_path is None:
            data_path = ""

        self.directions_wind, self.speeds_wind, self.TIs_wind, self.pmf_wind, _, _ = (
            self.wind_query.unpack()
        )
        self.N_wind_conditions = len(self.pmf_wind)

        # add the outputs we want for an AEP analysis:
        #   - AEP estimate
        #   - farm and turbine powers
        #   - turbine thrusts
        self.add_output(
            "AEP_farm",
            0.0,
            units="W*h",
        )
        self.add_output(
            "power_farm",
            np.zeros((self.N_wind_conditions,)),
            units="W",
        )
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
        # ... more outputs can be added here

    def setup_partials(self):
        """Derivative setup for OM component."""
        # the default (but not preferred!) derivatives are FDM
        self.declare_partials("*", "*", method="fd")

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
