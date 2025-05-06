from pathlib import Path
from os import PathLike
import numpy as np
import yaml
import copy

import floris

import ard.utils.io
import ard.farm_aero.templates as templates


def create_FLORIS_turbine(
    input_turbine_spec: dict | PathLike,
    filename_turbine_FLORIS: PathLike = None,
) -> dict:
    """
    Create a FLORIS turbine from a generic Ard turbine specification.

    Parameters
    ----------
    input_turbine_spec : dict | PathLike
        a turbine specification from which to extract a FLORIS turbine
    filename_turbine_FLORIS : PathLike, optional
        a path to save a FLORIS turbine configuration yaml file, optionally

    Returns
    -------
    dict
        a FLORIS turbine configuration in dictionary form

    Raises
    ------
    TypeError
        if the turbine specification input is not the correct type
    """

    if isinstance(input_turbine_spec, PathLike):
        with open(input_turbine_spec, "r") as file_turbine_spec:
            turbine_spec = ard.utils.io.load_turbine_spec(file_turbine_spec)
    elif type(input_turbine_spec) == dict:
        turbine_spec = input_turbine_spec
    else:
        raise TypeError(
            "create_FLORIS_yamlfile requires either a dict input or a filename input.\n"
            + f"received a {type(input_turbine_spec)}"
        )

    # load speed/power/thrust file
    filename_power_thrust = turbine_spec["performance_data_ccblade"]["power_thrust_csv"]
    pt_raw = np.genfromtxt(filename_power_thrust, delimiter=",").T.tolist()

    # create FLORIS config dict
    turbine_FLORIS = dict()
    turbine_FLORIS["turbine_type"] = turbine_spec["description"]["name"]
    turbine_FLORIS["hub_height"] = turbine_spec["geometry"]["height_hub"]
    turbine_FLORIS["rotor_diameter"] = turbine_spec["geometry"]["diameter_rotor"]
    turbine_FLORIS["TSR"] = turbine_spec["nameplate"]["TSR"]
    # turbine_FLORIS["multi_dimensional_cp_ct"] = True
    # turbine_FLORIS["power_thrust_data_file"] = filename_power_thrust
    turbine_FLORIS["power_thrust_table"] = {
        "cosine_loss_exponent_yaw": turbine_spec["model_specifications"]["FLORIS"][
            "exponent_penalty_yaw"
        ],
        "cosine_loss_exponent_tilt": turbine_spec["model_specifications"]["FLORIS"][
            "exponent_penalty_tilt"
        ],
        "peak_shaving_fraction": turbine_spec["model_specifications"]["FLORIS"][
            "fraction_peak_shaving"
        ],
        "peak_shaving_TI_threshold": 0.0,
        "ref_air_density": turbine_spec["performance_data_ccblade"][
            "density_ref_cp_ct"
        ],
        "ref_tilt": turbine_spec["performance_data_ccblade"]["tilt_ref_cp_ct"],
        "wind_speed": pt_raw[0],
        "power": (
            0.5
            * turbine_spec["performance_data_ccblade"]["density_ref_cp_ct"]
            * (np.pi / 4.0 * turbine_spec["geometry"]["diameter_rotor"] ** 2)
            * np.array(pt_raw[0]) ** 3
            * pt_raw[1]
            / 1e3
        ).tolist(),
        "thrust_coefficient": pt_raw[2],
    }

    # If an export filename is given, write it out
    if filename_turbine_FLORIS is not None:
        with open(filename_turbine_FLORIS, "w") as file_turbine_FLORIS:
            yaml.safe_dump(turbine_FLORIS, file_turbine_FLORIS)

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
        self.fmodel.set(
            wind_shear=self.modeling_options.get("wind_shear", 0.585),
            turbine_type=[create_FLORIS_turbine(self.modeling_options["turbine"])],
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
            layout_x=inputs["x_turbines"],
            layout_y=inputs["y_turbines"],
            wind_data=self.time_series,
            yaw_angles=np.array([inputs["yaw_turbines"]]),
        )
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
        super().setup_partials()

    def compute(self, inputs, outputs):

        # set up and run the floris model
        self.fmodel.set(
            layout_x=inputs["x_turbines"],
            layout_y=inputs["y_turbines"],
            wind_data=self.wind_rose,
            yaw_angles=np.array([inputs["yaw_turbines"]]),
        )
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
