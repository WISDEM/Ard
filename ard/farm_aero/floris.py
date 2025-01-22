import os

import numpy as np
import floris

import ard.utils
import ard.farm_aero.templates as templates


class FLORISFarmComponent:
    """
    Secondary-inherit component for managing FLORIS for farm simulations.

    This is a base class for farm aerodynamics simulations using FLORIS, which
    should cover all the necessary configuration, reproducibility config file
    saving, and output directory management.

    It is not a child class of an OpenMDAO components, but it is designed to
    mirror the form of the OM component, so that FLORIS activities are separated
    to have run times that correspond to the similarly-named OM component
    methods. It is intended to be a second-inherit base class for FLORIS-based
    OpenMDAO components, and will not work unless the calling object is a
    specialized class that _also_ specializes `openmdao.api.Component`.

    Options
    -------
    case_title : str
        a "title" for the case, used to disambiguate runs in practice
    """

    eval_count = 0  # an evaluation counter for FLORIS calls

    def initialize(self):
        """Initialization-time FLORIS management."""
        self.options.declare("case_title")

    def setup(self):
        """Setup-time FLORIS management."""

        # set up FLORIS
        self.fmodel = floris.FlorisModel("defaults")
        self.fmodel.set(
            wind_shear=self.modeling_options.get("wind_shear", 0.585),
            turbine_type=[
                ard.utils.create_FLORIS_turbine(self.modeling_options["turbine"])
            ],
        )
        self.fmodel.assign_hub_height_to_ref_height()

        self.case_title = self.options["case_title"]
        self.dir_floris = os.path.join("case_files", self.case_title, "floris_inputs")
        os.makedirs(self.dir_floris, exist_ok=True)

    def compute(self, inputs):
        """
        Compute-time FLORIS management.

        Compute-time FLORIS management should be specialized based on use case.
        If the base class is not specialized, an error will be raised.
        """

        raise NotImplementedError("compute must be specialized,")

    def setup_partials(self):
        """Derivative setup for OM component."""
        # for FLORIS, no derivatives. use FD because FLORIS is cheap
        self.declare_partials("*", "*")
        self.declare_partials(
            of="*",
            wrt=["x_turbines", "y_turbines"],
            method="fd",
            form="forward",
            step=1.0e-1,  # step by 100m for FD
        )

    def get_AEP_farm(self):
        """Get the AEP of a FLORIS farm."""
        return self.fmodel.get_farm_AEP()

    def get_power_farm(self):
        """Get the farm power of a FLORIS farm at each wind condition."""
        return self.fmodel.get_farm_power()

    def get_power_turbines(self):
        """Get the turbine powers of a FLORIS farm at each wind condition."""
        return self.fmodel.get_turbine_powers().T

    def get_thrust_turbines(self):
        """Get the turbine thrusts of a FLORIS farm at each wind condition."""
        # FLORIS computes the thrust precursors, compute and return thrust
        # use pure FLORIS to get these values for consistency
        CT_turbines = self.fmodel.get_turbine_thrust_coefficients()
        V_turbines = self.fmodel.turbine_average_velocities
        rho_floris = self.fmodel.core.flow_field.air_density
        A_floris = np.pi * self.fmodel.core.farm.rotor_diameters**2 / 4

        thrust_turbines = CT_turbines * (0.5 * rho_floris * A_floris * V_turbines**2)
        return thrust_turbines.T

    def dump_floris_yamlfile(self, dir_output=None):
        """
        Export the current FLORIS inputs to a YAML file file for reproducibility of the analysis.
        The file will be saved in the `dir_output` directory, or in the current working directory
        if `dir_output` is None.
        """
        if dir_output is None:
            dir_output = self.dir_floris
        self.fmodel.core.to_file(os.path.join(dir_output, "batch.yaml"))


class FLORISBatchPower(templates.BatchFarmPowerTemplate, FLORISFarmComponent):
    """
    Component class for computing a batch power analysis using FLORIS.

    A component class that evaluates a series of farm power and associated
    quantities using FLORIS. Inherits the interface from
    `templates.BatchFarmPowerTemplate` and the computational guts from
    `FLORISFarmComponent`.

    Options
    -------
    case_title : str
        a "title" for the case, used to disambiguate runs in practice (inherited
        from `FLORISFarmComponent`)
    modeling_options : dict
        a modeling options dictionary (inherited via
        `templates.BatchFarmPowerTemplate`)
    wind_query : floris.wind_data.WindRose
        a WindQuery objects that specifies the wind conditions that are to be
        computed (inherited from `templates.BatchFarmPowerTemplate`)

    Inputs
    ------
    x_turbines : np.ndarray
        a 1D numpy array indicating the x-dimension locations of the turbines,
        with length `N_turbines` (inherited via
        `templates.BatchFarmPowerTemplate`)
    y_turbines : np.ndarray
        a 1D numpy array indicating the y-dimension locations of the turbines,
        with length `N_turbines` (inherited via
        `templates.BatchFarmPowerTemplate`)
    yaw_turbines : np.ndarray
        a numpy array indicating the yaw angle to drive each turbine to with
        respect to the ambient wind direction, with length `N_turbines`
        (inherited via `templates.BatchFarmPowerTemplate`)

    Outputs
    -------
    power_farm : np.ndarray
        an array of the farm power for each of the wind conditions that have
        been queried (inherited from `templates.BatchFarmPowerTemplate`)
    power_turbines : np.ndarray
        an array of the farm power for each of the turbines in the farm across
        all of the conditions that have been queried on the wind rose
        (`N_turbines`, `N_wind_conditions`) (inherited from
        `templates.BatchFarmPowerTemplate`)
    thrust_turbines : np.ndarray
        an array of the wind turbine thrust for each of the turbines in the farm
        across all of the conditions that have been queried on the wind rose
        (`N_turbines`, `N_wind_conditions`) (inherited from
        `templates.BatchFarmPowerTemplate`)
    """

    def initialize(self):
        super().initialize()  # run super class script first!
        FLORISFarmComponent.initialize(self)  # FLORIS superclass

    def setup(self):
        super().setup()  # run super class script first!
        FLORISFarmComponent.setup(self)  # setup a FLORIS run

    def setup_partials(self):
        FLORISFarmComponent.setup_partials(self)

    def compute(self, inputs, outputs):

        # generate the list of conditions for evaluation
        self.time_series = floris.TimeSeries(
            wind_directions=np.degrees(np.array(self.wind_query.get_directions())),
            wind_speeds=np.array(self.wind_query.get_speeds()),
            turbulence_intensities=np.array(self.wind_query.get_TIs()),
        )

        # set up and run the floris model
        self.fmodel.set(
            layout_x=inputs["x_turbines"] * 1e3,  # convert to meters for FLORIS
            layout_y=inputs["y_turbines"] * 1e3,  # convert to meters for FLORIS
            wind_data=self.time_series,
            yaw_angles=np.array([inputs["yaw_turbines"]]),
        )
        self.fmodel.set_operation_model("peak-shaving")

        self.fmodel.run()
        self.eval_count += 1

        # dump the yaml to re-run this case on demand
        FLORISFarmComponent.dump_floris_yamlfile(self, self.dir_floris)

        # FLORIS computes the powers
        outputs["power_farm"] = (
            FLORISFarmComponent.get_power_farm(self) / 1e6
        )  # convert from FLORIS W to output MW
        outputs["power_turbines"] = (
            FLORISFarmComponent.get_power_turbines(self) / 1e6
        )  # convert from FLORIS W to output MW
        outputs["thrust_turbines"] = (
            FLORISFarmComponent.get_thrust_turbines(self) / 1e3
        )  # convert from FLORIS N to output kN


class FLORISAEP(templates.FarmAEPTemplate, FLORISFarmComponent):
    """
    Component class for computing an AEP analysis using FLORIS.

    A component class that evaluates a series of farm power and associated
    quantities using FLORIS with a wind rose to make an AEP estimate. Inherits
    the interface from `templates.FarmAEPTemplate` and the computational guts
    from `FLORISFarmComponent`.

    Options
    -------
    case_title : str
        a "title" for the case, used to disambiguate runs in practice (inherited
        from `FLORISFarmComponent`)
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
        FLORISFarmComponent.initialize(self)  # add on FLORIS superclass

    def setup(self):
        super().setup()  # run super class script first!
        FLORISFarmComponent.setup(self)  # setup a FLORIS run

    def setup_partials(self):
        FLORISFarmComponent().setup_partials()

    def compute(self, inputs, outputs):

        # set up and run the floris model
        self.fmodel.set(
            layout_x=inputs["x_turbines"] * 1e3,
            layout_y=inputs["y_turbines"] * 1e3,
            wind_data=self.wind_rose,
            yaw_angles=np.array([inputs["yaw_turbines"]]),
        )
        self.fmodel.set_operation_model("peak-shaving")

        print("RUNNING A FLORIS! ...", end="", flush=True)  # DEBUG!!!!!
        self.fmodel.run()
        self.eval_count += 1
        print(" DONE!", flush=True)  # DEBUG!!!!!

        # dump the yaml to re-run this case on demand
        FLORISFarmComponent.dump_floris_yamlfile(self, self.dir_floris)

        # FLORIS computes the powers
        outputs["AEP_farm"] = (
            FLORISFarmComponent.get_AEP_farm(self) / 1e9
        )  # convert from FLORIS W*h to output GW*h
        outputs["power_farm"] = (
            FLORISFarmComponent.get_power_farm(self) / 1e6
        )  # convert from FLORIS W to output MW
        outputs["power_turbines"] = (
            FLORISFarmComponent.get_power_turbines(self) / 1e6
        )  # convert from FLORIS W to output MW
        outputs["thrust_turbines"] = (
            FLORISFarmComponent.get_thrust_turbines(self) / 1e3
        )  # convert from FLORIS N to output kN

    def setup_partials(self):
        FLORISFarmComponent.setup_partials(self)
