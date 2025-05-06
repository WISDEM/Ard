import numpy as np
import jax.numpy as jnp
import jax
import ard.utils.mathematics
import openmdao.api as om


class TurbineSpacing(om.ExplicitComponent):
    """
    A class to return distances between turbines

    Options
    -------
    modeling_options : dict
        a modeling options dictionary (inherited from `FarmAeroTemplate`)

    Inputs
    ------
    x_turbines : np.ndarray
        a 1D numpy array indicating the x-dimension locations of the turbines,
        with length `N_turbines` (mirrored w.r.t. `FarmAeroTemplate`)
    y_turbines : np.ndarray
        a 1D numpy array indicating the y-dimension locations of the turbines,
        with length `N_turbines` (mirrored w.r.t. `FarmAeroTemplate`)
    """

    def initialize(self):
        """Initialization of the OpenMDAO component."""
        self.options.declare("modeling_options")

    def setup(self):
        """Setup of the OpenMDAO component."""

        # load modeling options
        self.modeling_options = self.options["modeling_options"]
        self.N_turbines = int(self.modeling_options["farm"]["N_turbines"])
        self.N_distances = int((self.N_turbines - 1) * self.N_turbines / 2)
        # MANAGE ADDITIONAL LATENT VARIABLES HERE!!!!!

        # set up inputs and outputs for mooring system
        self.add_input(
            "x_turbines", jnp.zeros((self.N_turbines,)), units="km"
        )  # x location of the mooring platform in km w.r.t. reference coordinates
        self.add_input(
            "y_turbines", jnp.zeros((self.N_turbines,)), units="km"
        )  # y location of the mooring platform in km w.r.t. reference coordinates

        self.add_output(
            "turbine_spacing",
            jnp.zeros(self.N_distances),
            units="km",
        )

    def setup_partials(self):
        """Derivative setup for the OpenMDAO component."""
        # the default (but not preferred!) derivatives are FDM
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """Computation for the OpenMDAO component."""

        # unpack the working variables
        x_turbines = inputs["x_turbines"]
        y_turbines = inputs["y_turbines"]

        spacing_distances = calculate_turbine_spacing(x_turbines, y_turbines)

        outputs["turbine_spacing"] = spacing_distances

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        # unpack the working variables
        x_turbines = inputs["x_turbines"]
        y_turbines = inputs["y_turbines"]

        jacobian = calculate_turbine_spacing_jac(x_turbines, y_turbines)

        partials["turbine_spacing", "x_turbines"] = jacobian[0]
        partials["turbine_spacing", "y_turbines"] = jacobian[1]


def calculate_turbine_spacing(
    x_turbines: np.ndarray,
    y_turbines: np.ndarray,
):
    N_turbines = len(x_turbines)

    # Create index pairs for i < j (upper triangle without diagonal)
    i_indices, j_indices = jnp.triu_indices(N_turbines, k=1)

    # Compute deltas
    dx = x_turbines[j_indices] - x_turbines[i_indices]
    dy = y_turbines[j_indices] - y_turbines[i_indices]
    deltas = jnp.stack([dx, dy], axis=1)

    # Vectorized norm calculation
    spacing_distance = ard.utils.mathematics.smooth_norm_vec(deltas)

    return spacing_distance


calculate_turbine_spacing = jax.jit(calculate_turbine_spacing)
calculate_turbine_spacing_jac = jax.jacrev(calculate_turbine_spacing, argnums=[0, 1])
