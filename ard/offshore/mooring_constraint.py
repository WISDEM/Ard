import numpy as np
import jax
import jax.numpy as jnp

import openmdao.api as om

import ard.utils.geometry
import ard.utils.mathematics


class MooringConstraint(om.ExplicitComponent):
    """
    A class to calculate the mooring line spacing distance for use in optimization
    constraints. Mooring lines may be defined in 2D or 3D, but the turbine positions
    are always assumed to be at sea level (z=0).

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
    x_turbines : np.ndarray
        a 1D numpy array indicating the x-dimension locations of the turbines,
        with length `N_turbines` (mirrored w.r.t. `FarmAeroTemplate`)
    y_turbines : np.ndarray
        a 1D numpy array indicating the y-dimension locations of the turbines,
        with length `N_turbines` (mirrored w.r.t. `FarmAeroTemplate`)
    x_anchors : np.ndarray
        a 1D numpy array indicating the x-dimension locations of the mooring
        system anchors, with shape `N_turbines` x `N_anchors`
    y_anchors : np.ndarray
        a 1D numpy array indicating the y-dimension locations of the mooring
        system anchors, with shape `N_turbines` x `N_anchors`
    z_anchors (optional) : np.ndarray
        a 1D numpy array indicating the z-dimension locations of the mooring
        system anchors, with shape `N_turbines` x `N_anchors`
    """

    def initialize(self):
        """Initialization of the OpenMDAO component."""
        self.options.declare("modeling_options")

    def setup(self):
        """Setup of the OpenMDAO component."""

        # load modeling options
        self.modeling_options = self.options["modeling_options"]
        self.N_turbines = int(self.modeling_options["farm"]["N_turbines"])
        self.N_anchor_dimensions = int(
            self.modeling_options["platform"]["N_anchor_dimensions"]
        )
        self.N_anchors = int(self.modeling_options["platform"]["N_anchors"])
        self.N_distances = int((self.N_turbines - 1) * self.N_turbines / 2)

        # set up inputs and outputs for mooring system
        self.add_input(
            "x_turbines", jnp.zeros((self.N_turbines,)), units="km"
        )  # x location of the mooring platform in km w.r.t. reference coordinates
        self.add_input(
            "y_turbines", jnp.zeros((self.N_turbines,)), units="km"
        )  # y location of the mooring platform in km w.r.t. reference coordinates
        self.add_input(
            "x_anchors",
            jnp.zeros((self.N_turbines, self.N_anchors)),
            units="km",
        )  # x location of the mooring anchors in km w.r.t. reference coordinates
        self.add_input(
            "y_anchors",
            jnp.zeros((self.N_turbines, self.N_anchors)),
            units="km",
        )  # y location of the mooring anchors in km w.r.t. reference coordinates
        if self.N_anchor_dimensions == 3:
            self.add_input(
                "z_anchors",
                jnp.zeros((self.N_turbines, self.N_anchors)),
                units="km",
            )  # z location of the mooring anchors in km w.r.t. reference coordinates

        self.add_output(
            "mooring_spacing",
            jnp.zeros(self.N_distances),
            units="km",
        )  # consolidated violation length

    def setup_partials(self):
        """Derivative setup for the OpenMDAO component."""
        # the default (but not preferred!) derivatives are FDM
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """Computation for the OpenMDAO component."""

        # unpack the working variables
        x_turbines = inputs["x_turbines"]
        y_turbines = inputs["y_turbines"]
        x_anchors = inputs["x_anchors"]
        y_anchors = inputs["y_anchors"]
        if self.N_anchor_dimensions == 3:
            z_anchors = inputs["z_anchors"]

        if self.N_anchor_dimensions == 2:
            distances = mooring_constraint_xy(
                x_turbines, y_turbines, x_anchors, y_anchors
            )
        elif self.N_anchor_dimensions == 3:
            distances = mooring_constraint_xyz(
                x_turbines, y_turbines, x_anchors, y_anchors, z_anchors
            )
        else:
            raise (ValueError("modeling_options['farm'][']"))

        outputs["mooring_spacing"] = distances

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        # unpack the working variables
        x_turbines = inputs["x_turbines"]
        y_turbines = inputs["y_turbines"]
        x_anchors = inputs["x_anchors"]
        y_anchors = inputs["y_anchors"]
        if self.N_anchor_dimensions == 3:
            z_anchors = inputs["z_anchors"]

        if self.N_anchor_dimensions == 2:
            jacobian = mooring_constraint_xy_jac(
                x_turbines, y_turbines, x_anchors, y_anchors
            )
        elif self.N_anchor_dimensions == 3:
            jacobian = mooring_constraint_xyz_jac(
                x_turbines, y_turbines, x_anchors, y_anchors, z_anchors
            )
        else:
            raise (ValueError("modeling_options['farm'][']"))

        partials["mooring_spacing", "x_turbines"] = jacobian[0]
        partials["mooring_spacing", "y_turbines"] = jacobian[1]
        partials["mooring_spacing", "x_anchors"] = jacobian[2]
        partials["mooring_spacing", "y_anchors"] = jacobian[3]
        if self.N_anchor_dimensions == 3:
            partials["mooring_spacing", "z_anchors"] = jacobian[4]


def mooring_constraint_xy(
    x_turbines: np.ndarray,
    y_turbines: np.ndarray,
    x_anchors: np.ndarray,
    y_anchors: np.ndarray,
):
    """Mooring distance calculation in 2 dimensions

    Args:
        x_turbines (np.ndarray): array of turbine x positions
        y_turbines (np.ndarray): array of turbine y positions
        x_anchors (np.ndarray): array of anchor x positions
        y_anchors (np.ndarray): array of anchor y positions

    Returns:
        np.ndarray: 1D array of distances with length (n_turbines - 1)*n_turbines/2
    """

    # convert inputs
    mooring_points = convert_inputs_x_y_to_xy(
        x_turbines, y_turbines, x_anchors, y_anchors
    )
    # calculate minimum distances between each set of moorings
    distances = calc_mooring_distances(mooring_points)

    return distances


mooring_constraint_xy = jax.jit(mooring_constraint_xy)
mooring_constraint_xy_jac = jax.jacrev(mooring_constraint_xy, argnums=[0, 1, 2, 3])


def mooring_constraint_xyz(
    x_turbines: np.ndarray,
    y_turbines: np.ndarray,
    x_anchors: np.ndarray,
    y_anchors: np.ndarray,
    z_anchors: np.ndarray,
):
    """Mooring distance calculation in 3 dimensions. The third dimension
    is only required for the anchors since the turbine platforms are
    all assumed to be at sea level.

    Args:
        x_turbines (np.ndarray): array of turbine x positions
        y_turbines (np.ndarray): array of turbine y positions
        x_anchors (np.ndarray): array of anchor x positions
        y_anchors (np.ndarray): array of anchor y positions
        z_anchors (np.ndarray): array of anchor z positions

    Returns:
        np.ndarray: 1D array of distances with length (n_turbines - 1)*n_turbines/2
    """

    # convert inputs
    mooring_points = convert_inputs_x_y_z_to_xyz(
        x_turbines,
        y_turbines,
        np.zeros(len(x_turbines)),
        x_anchors,
        y_anchors,
        z_anchors,
    )
    # calculate minimum distances between each set of moorings
    distances = calc_mooring_distances(mooring_points)

    return distances


mooring_constraint_xyz = jax.jit(mooring_constraint_xyz)
mooring_constraint_xyz_jac = jax.jacrev(mooring_constraint_xyz, argnums=[0, 1, 2, 3, 4])


def calc_mooring_distances(mooring_points: np.ndarray) -> np.ndarray:
    """Calculate the minimum distances between each set of mooring lines

    Args:
        mooring_points (np.ndarray): array of mooring points of shape
        (n_turbines, n_anchors+1, n_dimensions) where n_dimensions may be 2 or 3

    Returns:
        np.ndarray: 1D array of distances with length (n_turbines - 1)*n_turbines/2
    """

    n_turbines = mooring_points.shape[0]

    # Create pairwise indices for all unique turbine pairs
    i_indices, j_indices = jnp.triu_indices(n_turbines, k=1)

    # Extract the corresponding mooring points for each pair
    mooring_A = mooring_points[i_indices]
    mooring_B = mooring_points[j_indices]

    # Compute distances for all pairs using `distance_mooring_to_mooring`
    distances = jax.vmap(distance_mooring_to_mooring)(mooring_A, mooring_B)

    return distances


def convert_inputs_x_y_to_xy(
    x_turbines: np.ndarray,
    y_turbines: np.ndarray,
    x_anchors: np.ndarray,
    y_anchors: np.ndarray,
) -> np.ndarray:
    """Convert from inputs of x for turbines, y for turbines, x for anchors, and y for
    anchors to single array for mooring specification that is of shape
    (n_turbines, n_anchors+1, 2). for each set of points, the turbine position is given
    first followed by the anchor positions.

    Args:
        x_turbines (np.ndarray): array of turbine x positions
        y_turbines (np.ndarray): array of turbine y positions
        x_anchors (np.ndarray): array of anchor x positions
        y_anchors (np.ndarray): array of anchor y positions

    Returns:
        np.ndarray: all turbine and anchor location information combined into a single
            array of shape (n_turbines, n_anchors+1, 2)
    """

    # Stack turbine positions and anchor positions directly
    turbine_positions = jnp.stack([x_turbines, y_turbines], axis=-1)[:, None, :]
    anchor_positions = jnp.stack([x_anchors, y_anchors], axis=-1)

    # Concatenate turbine positions with anchor positions
    xy = jnp.concatenate([turbine_positions, anchor_positions], axis=1)

    return xy


def convert_inputs_x_y_z_to_xyz(
    x_turbines: np.ndarray,
    y_turbines: np.ndarray,
    z_turbines: np.ndarray,
    x_anchors: np.ndarray,
    y_anchors: np.ndarray,
    z_anchors: np.ndarray,
) -> np.ndarray:
    """Convert from inputs of x for turbines, y for turbines, z for turbines, x for anchors,
    y for anchors, and z for anchors to single array for mooring specification that is of
    shape (n_turbines, n_anchors+1, 3). for each set of points, the turbine position is given
    first followed by the anchor positions.

    Args:
        x_turbines (np.ndarray): array of turbine x positions
        y_turbines (np.ndarray): array of turbine y positions
        z_turbines (np.ndarray): array of turbine z positions
        x_anchors (np.ndarray): array of anchor x positions
        y_anchors (np.ndarray): array of anchor y positions
        z_anchors (np.ndarray): array of anchor z positions

    Returns:
        np.ndarray: all input information combined into a single array of shape (n_turbines, n_anchors+1, 3)
    """

    # Stack turbine positions and anchor positions directly
    turbine_positions = jnp.stack([x_turbines, y_turbines, z_turbines], axis=-1)[
        :, None, :
    ]
    anchor_positions = jnp.stack([x_anchors, y_anchors, z_anchors], axis=-1)

    # Concatenate turbine positions with anchor positions
    xyz = jnp.concatenate([turbine_positions, anchor_positions], axis=1)

    return xyz


def distance_point_to_mooring(point: np.ndarray, P_mooring: np.ndarray) -> float:
    """Find the distance from a point to a set of mooring lines for a single floating wind turbine.
        While arguments may be given in either 2d ([x,y]) or 3d ([x,y,z]), the point of interest
        and the mooring line points must all be given with the same number of dimensions.

    Args:
        point (np.ndarray): Point of interest in 2d ([x,y]) or 3d ([x,y,z]).
        P_mooring (np.ndarray): The set of points defining the mooring line layout. The first point should
                                be the center, the rest of the points define the anchor points. Points may
                                be given in 2d ([x,y]) or 3d ([x,y,z]).

    Returns:
        float: The shortest distance from the point of interest to the set of mooring lines.
    """

    p_center = P_mooring[0]
    anchors = P_mooring[1:]

    distances = jax.vmap(
        ard.utils.geometry.distance_point_to_lineseg_nd,
        in_axes=(None, None, 0),
    )(point, p_center, anchors)

    return ard.utils.mathematics.smooth_min(distances)


def distance_mooring_to_mooring(
    P_mooring_A: np.ndarray, P_mooring_B: np.ndarray
) -> float:
    """Calculate the distance from one set of mooring lines to another. Moorings
    are defined with the center point (platform location) first, followed by the
    anchor points in no specific order.

    Args:
        P_mooring_A (np.ndarray): ndarray of points of mooring A of shape (npoints, nd) (e.g. (4, (x, y, z))).
            Center point must come first.
        P_mooring_B (np.ndarray): ndarray of points of mooring B of shape (npoints, nd) (e.g. (4, (x, y, z))).
            Center point must come first.

    Returns:
        float: shortest distance between the two sets of moorings
    """

    p_center_A = P_mooring_A[0]
    p_center_B = P_mooring_B[0]
    anchors_A = P_mooring_A[1:]
    anchors_B = P_mooring_B[1:]

    # Vectorize the computation of distances between all pairs of line segments
    def compute_segment_distance(anchor_A, anchor_B):
        return ard.utils.geometry.distance_lineseg_to_lineseg_nd(
            p_center_A, anchor_A, p_center_B, anchor_B
        )

    # Use vmap to compute distances for all combinations of anchors
    distances = jax.vmap(
        lambda anchor_A: jax.vmap(
            lambda anchor_B: compute_segment_distance(anchor_A, anchor_B)
        )(anchors_B)
    )(anchors_A)

    # Find the smooth minimum distance
    return ard.utils.mathematics.smooth_min(distances.flatten())
