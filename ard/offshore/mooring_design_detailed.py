from pathlib import Path

import numpy as np

import openmdao.api as om

import ard
from ard.geographic.geomorphology import BathymetryGridData


class DetailedMooringDesign(om.ExplicitComponent):
    """
    A class to create a detailed mooring design for a floating offshore wind farm.

    This is a class that should be used to generate a floating offshore wind
    farm's collective mooring system.

    Options
    -------
    modeling_options : dict
        a modeling options dictionary (inherited from `FarmAeroTemplate`)
    wind_query : floris.wind_data.WindRose
        a WindQuery objects that specifies the wind conditions that are to be
        computed
    bathymetry_data : ard.geographic.BathymetryData
        a BathymetryData object to specify the bathymetry mesh/sampling

    Inputs
    ------
    phi_platform : np.ndarray
        a 1D numpy array indicating the cardinal direction angle of the mooring
        orientation, with length `N_turbines`
    x_turbines : np.ndarray
        a 1D numpy array indicating the x-dimension locations of the turbines,
        with length `N_turbines` (mirrored w.r.t. `FarmAeroTemplate`)
    y_turbines : np.ndarray
        a 1D numpy array indicating the y-dimension locations of the turbines,
        with length `N_turbines` (mirrored w.r.t. `FarmAeroTemplate`)
    thrust_turbines : np.ndarray
        an array of the wind turbine thrust for each of the turbines in the farm
        across all of the conditions that have been queried on the wind rose
        (`N_turbines`, `N_wind_conditions`)

    Outputs
    -------
    x_anchors : np.ndarray
        a 1D numpy array indicating the x-dimension locations of the mooring
        system anchors, with shape `N_turbines` x `N_anchors`
    y_anchors : np.ndarray
        a 1D numpy array indicating the y-dimension locations of the mooring
        system anchors, with shape `N_turbines` x `N_anchors`

    """

    def initialize(self):
        """Initialization of the OpenMDAO component."""
        self.options.declare("modeling_options")

        # farm power wind conditions query (not necessarily a full wind rose)
        self.options.declare("wind_query")

        # currently I'm thinking of sea bed conditions as a class, see above
        self.options.declare("bathymetry_data")  # BatyhmetryData object

    def setup(self):
        """Setup of the OpenMDAO component."""

        # load modeling options
        self.modeling_options = self.options["modeling_options"]
        self.N_turbines = self.modeling_options["farm"]["N_turbines"]
        self.N_anchors = self.modeling_options["platform"]["N_anchors"]
        self.min_mooring_line_length_m = self.modeling_options["platform"][
            "min_mooring_line_length_m"
        ]

        # get the number of wind conditions (for thrust measurements)
        if self.options["wind_query"] is not None:
            self.N_wind_conditions = self.options["wind_query"].N_conditions
        # MANAGE ADDITIONAL LATENT VARIABLES HERE!!!!!

        # BEGIN: VARIABLES TO BE INCORPORATED PROPERLY

        class Placeholder: pass  # DEBUG!!!!!
        self.temporary_variables = Placeholder()  # DEBUG!!!!!
        self.temporary_variables.phi_mooring = np.zeros((self.N_turbines,))  # the mooring headings

        self.temporary_variables.path_to_bathy_moorpy = (
            Path(ard.__file__).parents[1]
            / "examples"
            / "data"
            / "offshore"
            / "GulfOfMaine_bathymetry_100x99.txt"
        )
        self.temporary_variables.bathymetry_data = BathymetryGridData()
        self.temporary_variables.bathymetry_data.load_moorpy_bathymetry(
            self.temporary_variables.path_to_bathy_moorpy
        )
        self.temporary_variables.soil_data = None  # TODO
        self.temporary_variables.radius_fairlead = 0.5  # m? idk, replace with a good value
        self.temporary_variables.depth_fairlead = 5.0  # m? idk, replace with a good value
        self.temporary_variables.type_anchor = "driven_pile"  # random choice
        # load anchor geometry yaml file based on ard package location
        self.temporary_variables.path_to_anchor_yaml = (
            Path(ard.__file__).parent
            / "examples"
            / "data"
            / "offshore"
            / "geometry_anchor.yaml"
        )
        self.temporary_variables.id_mooring_system = [
            f"m{v}:03d" for v in list(range(len(self.temporary_variables.phi_mooring)))
        ]  # just borrow turbine IDs for now: 3-digit, zero padded integer prefixed by m

        # END VARIABLES TO BE INCORPORATED PROPERLY

        # set up inputs and outputs for mooring system
        self.add_input(
            "phi_platform", np.zeros((self.N_turbines,)), units="deg"
        )  # cardinal direction of the mooring platform orientation
        self.add_input(
            "x_turbines", np.zeros((self.N_turbines,)), units="km"
        )  # x location of the mooring platform in km w.r.t. reference coordinates
        self.add_input(
            "y_turbines", np.zeros((self.N_turbines,)), units="km"
        )  # y location of the mooring platform in km w.r.t. reference coordinates
        if self.options["wind_query"] is not None:
            self.add_input(
                "thrust_turbines",
                np.zeros((self.N_turbines, self.N_wind_conditions)),
                units="kN",
            )  # turbine thrust coming from each wind direction
        # ADD ADDITIONAL (DESIGN VARIABLE) INPUTS HERE!!!!!

        self.add_output(
            "x_anchors",
            np.zeros((self.N_turbines, self.N_anchors)),
            units="km",
        )  # x location of the mooring platform in km w.r.t. reference coordinates
        self.add_output(
            "y_anchors",
            np.zeros((self.N_turbines, self.N_anchors)),
            units="km",
        )  # y location of the mooring platform in km w.r.t. reference coordinates

    def setup_partials(self):
        """Derivative setup for the OpenMDAO component."""
        # the default (but not preferred!) derivatives are FDM
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """Computation for the OpenMDAO component."""

        # unpack the working variables
        phi_platform = inputs["phi_platform"]
        x_turbines = inputs["x_turbines"]
        y_turbines = inputs["y_turbines"]
        # thrust_turbines = inputs["thrust_turbines"]  # future-proofing


        # BEGIN: ALIASES FOR SOME USEFUL VARIABLES

        phi_mooring = self.temporary_variables.phi_mooring
        path_to_bathy_moorpy = self.temporary_variables.path_to_bathy_moorpy
        bathymetry_data = self.temporary_variables.bathymetry_data
        soil_data = self.temporary_variables.soil_data
        radius_fairlead = self.temporary_variables.radius_fairlead
        depth_fairlead = self.temporary_variables.depth_fairlead
        type_anchor = self.temporary_variables.type_anchor
        path_to_anchor_yaml = self.temporary_variables.path_to_anchor_yaml
        id_mooring_system = self.temporary_variables.id_mooring_system

        # END ALIASES FOR SOME USEFUL VARIABLES


        # BEGIN: REPLACE ME WITH OPERATING CODE

        print("\n\nARRIVED AT COMPUTE FUNCTION\n\n")

        raise NotImplementedError("HELLO FRIENDS, IMPLEMENT HERE!")

        # END REPLACE ME WITH OPERATING CODE


        # replace the below with the final anchor locations...
        outputs["x_anchors"] = x_anchors
        outputs["y_anchors"] = y_anchors
