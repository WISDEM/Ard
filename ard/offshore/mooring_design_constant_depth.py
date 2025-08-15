import numpy as np

import openmdao.api as om

import math

import ard.geographic


def generate_anchor_points(
    center: np.ndarray, length: float, rotation_deg: float, N: int
) -> np.ndarray:
    """Generates anchor points equally spaced around the platform

    Args:
        center (np.ndarray): x and y of platform in km
        length (float): desired horizontal anchor length in km
        rotation_deg (float): rotation in deg. counter-clockwise from east
        N (int): number of anchors

    Returns:
        np.ndarray: array of size N by 2 containing the x and y positions of each anchor
    """

    cx, cy = center
    angle_step = 360 / N
    lines = np.zeros([N, 2])

    for i in range(N):
        angle_deg = rotation_deg + i * angle_step
        angle_rad = math.radians(angle_deg)
        x = cx + length * math.cos(angle_rad)
        y = cy + length * math.sin(angle_rad)
        lines[i, 0] = x
        lines[i, 1] = y

    return lines


def simple_mooring_design(
    phi_platform: np.ndarray,
    x_turbines: np.ndarray,
    y_turbines: np.ndarray,
    length: float,
    N_turbines: int,
    N_anchors: int,
) -> tuple[np.ndarray]:
    """_summary_

    Args:
        phi_platform (np.ndarray): counterclockwise rotation from east in deg. for each platform
        x_turbines (np.ndarray): list of platform/turbine easting in km
        y_turbines (np.ndarray): list of platform/turbine northing in km
        length (float): desired horizontal anchor length in km
        N_turbines (int): number of wind turbines in the farm
        N_anchors (int): number of anchors per turbine/platform

    Returns:
        tuple[np.ndarray]: x locations of anchors, y locations of anchors, each array of shape N_turbines by N_anchors
    """

    x_anchors = np.zeros([N_turbines, N_anchors])
    y_anchors = np.zeros_like(x_anchors)

    for i, (x, y) in enumerate(zip(x_turbines, y_turbines)):

        center = (x, y)

        anchors = generate_anchor_points(
            center=center, length=length, rotation_deg=phi_platform[i], N=N_anchors
        )

        for j in range(N_anchors):
            x_anchors[i, j] = anchors[j, 0]
            y_anchors[i, j] = anchors[j, 1]

    return x_anchors, y_anchors


class ConstantDepthMooringDesign(om.ExplicitComponent):
    """
    A class to create a constant-depth simplified mooring design for a floating
    offshore wind farm.

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
        self.N_turbines = self.modeling_options["layout"]["N_turbines"]
        self.N_anchors = self.modeling_options["platform"]["N_anchors"]
        self.min_mooring_line_length_m = self.modeling_options["platform"][
            "min_mooring_line_length_m"
        ]

        # get the number of wind conditions (for thrust measurements)
        if self.options["wind_query"] is not None:
            self.N_wind_conditions = self.options["wind_query"].N_conditions
        # MANAGE ADDITIONAL LATENT VARIABLES HERE!!!!!

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
        # thrust_turbines = inputs["thrust_turbines"]  #

        # BEGIN: REPLACE ME

        x_anchors, y_anchors = simple_mooring_design(
            phi_platform=phi_platform,
            x_turbines=x_turbines,
            y_turbines=y_turbines,
            length=self.min_mooring_line_length_m * 1e-3,  # convert to km
            N_turbines=self.N_turbines,
            N_anchors=self.N_anchors,
        )

        # END REPLACE ME

        # replace the below with the final anchor locations...
        outputs["x_anchors"] = x_anchors
        outputs["y_anchors"] = y_anchors
