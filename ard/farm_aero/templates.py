import numpy as np

import openmdao.api as om

from pathlib import Path

import floris


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
        self.options.declare("data_path")

    def setup(self):
        """Setup of OM component."""
        # load modeling options
        self.modeling_options = self.options["modeling_options"]
        self.N_turbines = self.modeling_options["farm"]["N_turbines"]

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
        self.wind_query = self.options["modeling_options"]["wind_rose"]
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
        wind_spec = self.options["modeling_options"]["wind_rose"]
        data_path = str(self.options["data_path"])

        if data_path is None:
            data_path = ""

        if isinstance(wind_spec, dict):
            # generate FLORIS wind data object from dictionary
            wind_rose_wrg_file = Path(data_path + "/" + wind_spec["file"]).resolve()
            wind_rose_wrg = floris.wind_data.WindRoseWRG(Path(wind_rose_wrg_file))
            wind_rose_wrg.set_wd_step(wind_spec["wd_step"])
            wind_rose_wrg.set_wind_speeds(np.array(wind_spec["wind_speeds"]))
            self.wind_rose = wind_rose_wrg.get_wind_rose_at_point(*wind_spec["point"])
        elif isinstance(wind_spec, floris.wind_data.WindRose):
            # unpack FLORIS wind data object
            self.wind_rose = wind_spec
        else:
            raise (
                TypeError(
                    f"wind rose was given as type {type(wind_spec)}, but must \
                            be one of [dict, floris.wind_data.WindRose]"
                )
            )
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
