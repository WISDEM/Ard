import os

import numpy as np
from flowers import FlowersModel

import ard.utils
import ard.farm_aero.templates as templates


class FLOWERSFarmComponent:
    """
    Secondary-inherit component for managing FLOWERS for farm simulations.

    This is a base class for farm aerodynamics simulations using FLOWERS, which
    should cover all the necessary configuration, reproducibility config file
    saving, and output directory management.

    It is not a child class of an OpenMDAO components, but it is designed to
    mirror the form of the OM component, so that FLOWERS activities are separated
    to have run times that correspond to the similarly-named OM component
    methods. It is intended to be a second-inherit base class for FLOWERS-based
    OpenMDAO components, and will not work unless the calling object is a
    specialized class that _also_ specializes `openmdao.api.Component`.

    Options
    -------
    case_title : str
        a "title" for the case, used to disambiguate runs in practice
    """

    def initialize(self):
        """Initialization-time FLOWERS management."""
        self.options.declare("case_title")

    def setup(self):
        """Setup-time FLOWERS management."""

        # set up FLOWERS
        self.flowers_model = FlowersModel("defaults")
        self.flowers_model.set(
            turbine_type=[
                ard.utils.create_FLORIS_turbine(self.modeling_options["turbine"])
            ],
        )

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
        self.declare_partials("farm_AEP", ["x_turbines", ], method="fd")

    def get_AEP_farm(self):
        """Get the AEP of a FLORIS farm."""
        return self.flowers_model.calculate_aep()

    def get_power_farm(self):
        """Get the farm power of a FLOWERS farm at each wind condition."""
        raise NotImplementedError("Not implemented for the FLOWERS model.")

    def get_power_turbines(self):
        """Get the turbine powers of a FLOWERS farm at each wind condition."""
        raise NotImplementedError("Not implemented for the FLOWERS model.")

    def get_thrust_turbines(self):
        """Get the turbine thrusts of a FLOWERS farm at each wind condition."""
        raise NotImplementedError("Not implemented for the FLOWERS model.")

    # def dump_floris_yamlfile(self, dir_output=None):
    #     """
    #     Export the current FLORIS inputs to a YAML file file for reproducibility of the analysis.
    #     The file will be saved in the `dir_output` directory, or in the current working directory
    #     if `dir_output` is None.
    #     """
    #     if dir_output is None:
    #         dir_output = self.dir_floris
    #     self.fmodel.core.to_file(os.path.join(dir_output, "batch.yaml"))


class FLOWERSAEP(templates.FarmAEPTemplate, FLOWERSFarmComponent):
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
        FLOWERSFarmComponent.initialize(self)  # add on FLORIS superclass

    def setup(self):
        super().setup()  # run super class script first!
        FLOWERSFarmComponent.setup(self)  # setup a FLORIS run

    def setup_partials(self):
        super().setup_partials()

    def compute(self, inputs, outputs):

        # set up and run the floris model
        self.flowers_model.set(
            layout_x=inputs["x_turbines"],
            layout_y=inputs["y_turbines"],
            wind_data=self.wind_data,
        )

        self.flowers_model.run()

        # dump the yaml to re-run this case on demand
        # FLOWERSFarmComponent.dump_flowers_yamlfile(self, self.dir_flowers)

        # FLORIS computes the powers
        outputs["AEP_farm"] = FLOWERSFarmComponent.get_AEP_farm(self)
        # outputs["power_farm"] = FLOWERSFarmComponent.get_power_farm(self)
        # outputs["power_turbines"] = FLOWERSFarmComponent.get_power_turbines(self)
        # outputs["thrust_turbines"] = FLOWERSFarmComponent.get_thrust_turbines(self)

    # def compute_partials(self, inputs, partials):

    #     partials # modify in place

    def setup_partials(self):
        FLOWERSFarmComponent.setup_partials(self)
