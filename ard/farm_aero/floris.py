from pathlib import Path
import copy

import numpy as np

import floris
import floris.turbine_library.turbine_utilities

import ard.farm_aero.templates as templates


def create_FLORIS_turbine_from_windIO(
    windIOplant: dict,
    modeling_options: dict = {},
) -> dict:

    # extract the turbine... assuming a single one for now
    windIOturbine = windIOplant["wind_farm"]["turbine"]

    # generate dictionary for FLORIS utility
    tdd = {}
    wind_speeds_Ct_curve = windIOturbine["performance"]["Ct_curve"]["Ct_wind_speeds"]
    values_Ct_curve = windIOturbine["performance"]["Ct_curve"]["Ct_values"]
    if "Cp_curve" in windIOturbine["performance"]:
        wind_speeds_Cp_curve = windIOturbine["performance"]["Cp_curve"][
            "Cp_wind_speeds"
        ]
        values_Cp_curve = windIOturbine["performance"]["Cp_curve"]["Cp_values"]
        if not np.allclose(wind_speeds_Cp_curve, wind_speeds_Ct_curve):
            raise NotImplementedError(
                "Ct and Cp curves are specified with different indep. variables. "
                "windIO allows seperate indep. variables for power/thrust curve "
                "specification, while FLORIS requires common indep. variables. "
                "Reconciliation has not been implemented yet."
            )
        tdd["wind_speed"] = wind_speeds_Cp_curve
        tdd["power_coefficient"] = values_Cp_curve
        tdd["thrust_coefficient"] = values_Ct_curve
    elif "power_curve" in windIOturbine["performance"]:
        wind_speeds_power_curve = windIOturbine["performance"]["power_curve"][
            "power_wind_speeds"
        ]
        values_power_curve = windIOturbine["performance"]["power_curve"]["power_values"]
        if not np.allclose(wind_speeds_power_curve, wind_speeds_Ct_curve):
            raise NotImplementedError(
                "Ct and power curves are specified with different indep. variables. "
                "windIO allows seperate indep. variables for power/thrust curve "
                "specification, while FLORIS requires common indep. variables. "
                "Reconciliation has not been implemented yet."
            )
        tdd["wind_speed"] = wind_speeds_power_curve
        tdd["power"] = values_power_curve / 1e3
        tdd["thrust_coefficient"] = values_Ct_curve
    elif all(
        val in windIOturbine["performance"]
        for val in [
            "rated_power",
            "rated_wind_speed",
            "cutin_wind_speed",
            "cutout_wind_speed",
        ]
    ):
        # extract key values
        rated_power = windIOturbine["rated_power"]
        rated_wind_speed = windIOturbine["rated_wind_speed"]
        cutin_wind_speed = windIOturbine["cutin_wind_speed"]
        cutout_wind_speed = windIOturbine["cutout_wind_speed"]

        # compute based on reg. I, II, III, IV textbook behaviors
        values_power_curve = (
            wind_speeds_Ct_curve**3 / rated_wind_speed**3 * rated_power
        )  # scales proportionally in reg. II
        values_power_curve[wind_speeds_Ct_curve >= rated_wind_speed] = (
            rated_power  # flat in reg. III
        )
        values_power_curve[wind_speeds_Ct_curve <= cutin_wind_speed] = (
            0.0  # zero in reg. I
        )
        values_power_curve[wind_speeds_Ct_curve >= cutout_wind_speed] = (
            0.0  # zero in reg. IV
        )

        # pack and ship
        tdd["wind_speed"] = wind_speeds_Ct_curve
        tdd["power"] = values_power_curve / 1e3
        tdd["thrust_coefficient"] = values_Ct_curve
    else:
        raise IndexError(
            "The windIO file appears to be invalid. Try validating and re-running."
        )

    # create a FLORIS turbine using the utility
    turbine_FLORIS = (
        floris.turbine_library.turbine_utilities.build_cosine_loss_turbine_dict(
            turbine_data_dict=tdd,
            turbine_name=windIOturbine["name"],
            hub_height=windIOturbine["hub_height"],
            rotor_diameter=windIOturbine["rotor_diameter"],
            TSR=windIOturbine.get("TSR"),
            generator_efficiency=windIOturbine.get("generator_efficiency", 1.0),
        )
    )

    # append peak shaving reduction fraction and TI threshhold they exist
    psf_val = modeling_options.get("floris", {}).get("peak_shaving_fraction")
    psf_thresh = modeling_options.get("floris", {}).get(
        "peak_shaving_TI_threshold", 0.0
    )
    if psf_val is not None:
        # if they exist, set them
        print(f"DEBUG!!!!! INSTALLING PEAK-SHAVING!")
        turbine_FLORIS["power_thrust_table"]["peak_shaving_fraction"] = psf_val
        turbine_FLORIS["power_thrust_table"]["peak_shaving_TI_threshold"] = psf_thresh

    # # If an export filename is given, write it out
    # if filename_turbine_FLORIS is not None:
    #     with open(filename_turbine_FLORIS, "w") as file_turbine_FLORIS:
    #         yaml.safe_dump(turbine_FLORIS, file_turbine_FLORIS)

    return copy.deepcopy(turbine_FLORIS)


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

    def initialize(self):
        """Initialization-time FLORIS management."""
        self.options.declare("case_title")

    def setup(self):
        """Setup-time FLORIS management."""

        # set up FLORIS
        self.fmodel = floris.FlorisModel("defaults")
        data_path = self.options["data_path"]
        self.fmodel.set(
            turbine_type=[
                create_FLORIS_turbine_from_windIO(self.windIO, self.modeling_options),
            ],
            wind_shear=self.windIO["site"]["energy_resource"]["wind_resource"].get(
                "shear"
            ),
        )
        self.fmodel.assign_hub_height_to_ref_height()

        self.case_title = self.options["case_title"]
        self.dir_floris = Path("case_files", self.case_title, "floris_inputs")
        self.dir_floris.mkdir(parents=True, exist_ok=True)

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
        self.declare_partials("*", "*", method="fd")

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
        self.fmodel.core.to_file(Path(dir_output, "batch.yaml"))


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
    # modeling_options : dict
    #     a modeling options dictionary (inherited via
    #     `templates.BatchFarmPowerTemplate`)
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
            wind_directions=np.degrees(np.array(self.wind_query.wind_directions)),
            wind_speeds=np.array(self.wind_query.wind_speeds),
            turbulence_intensities=np.array(self.wind_query.turbulence_intensities),
        )

        # set up and run the floris model
        self.fmodel.set(
            layout_x=inputs["x_turbines"],
            layout_y=inputs["y_turbines"],
            wind_data=self.time_series,
            yaw_angles=np.array([inputs["yaw_turbines"]]),
            reference_wind_height=(
                self.wind_query.reference_height
                if hasattr(self.wind_query, "reference_height")
                else None
            ),
        )
        print(f"DEBUG!!!!! self.modeling_options: {self.modeling_options['floris']}")
        if "peak_shaving_fraction" in self.modeling_options.get("floris", {}):
            print(f"DEBUG!!!!! ENABLING PEAK-SHAVING!")
            self.fmodel.set_operation_model("peak-shaving")

        self.fmodel.run()

        # dump the yaml to re-run this case on demand
        FLORISFarmComponent.dump_floris_yamlfile(self, self.dir_floris)

        # FLORIS computes the powers
        outputs["power_farm"] = FLORISFarmComponent.get_power_farm(self)
        outputs["power_turbines"] = FLORISFarmComponent.get_power_turbines(self)
        outputs["thrust_turbines"] = FLORISFarmComponent.get_thrust_turbines(self)


class FLORISAEP(templates.FarmAEPTemplate):
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
    # modeling_options : dict
    #     a modeling options dictionary (inherited via
    #     `templates.FarmAEPTemplate`)
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
        super().setup_partials()

    def compute(self, inputs, outputs):

        # set up and run the floris model
        self.fmodel.set(
            layout_x=inputs["x_turbines"],
            layout_y=inputs["y_turbines"],
            wind_data=self.wind_query,
            yaw_angles=np.array([inputs["yaw_turbines"]]),
            reference_wind_height=(
                self.wind_query.reference_height
                if hasattr(self.wind_query, "reference_height")
                else None
            ),
        )
        print(f"DEBUG!!!!! self.modeling_options: {self.modeling_options['floris']}")
        if "peak_shaving_fraction" in self.modeling_options.get("floris", {}):
            print(f"DEBUG!!!!! ENABLING PEAK-SHAVING!")
            self.fmodel.set_operation_model("peak-shaving")

        self.fmodel.run()

        # dump the yaml to re-run this case on demand
        FLORISFarmComponent.dump_floris_yamlfile(self, self.dir_floris)

        # FLORIS computes the powers
        outputs["AEP_farm"] = FLORISFarmComponent.get_AEP_farm(self)
        outputs["power_farm"] = FLORISFarmComponent.get_power_farm(self)
        outputs["power_turbines"] = FLORISFarmComponent.get_power_turbines(self)
        outputs["thrust_turbines"] = FLORISFarmComponent.get_thrust_turbines(self)

    def setup_partials(self):
        FLORISFarmComponent.setup_partials(self)
