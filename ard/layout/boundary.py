import numpy as np
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)
import ard.utils.mathematics
import ard.utils.geometry
import openmdao.api as om


class FarmBoundaryDistancePolygon(om.ExplicitComponent):
    """
    A class to return distances between turbines and a polygonal boundary, or
    sets of polygonal boundary regions.

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
        self.boundary_vertices = self.modeling_options["farm"]["boundary"]["vertices"]
        self.boundary_regions = self.modeling_options["farm"]["boundary"][
            "turbine_region_assignments"
        ]

        self.distance_multi_point_to_multi_polygon_ray_casting_jac = jax.jacfwd(
            ard.utils.geometry.distance_multi_point_to_multi_polygon_ray_casting, [0, 1]
        )
        # MANAGE ADDITIONAL LATENT VARIABLES HERE!!!!!

        # set up inputs and outputs for mooring system
        self.add_input(
            "x_turbines", jnp.zeros((self.N_turbines,)), units="km"
        )  # x location of the mooring platform in km w.r.t. reference coordinates
        self.add_input(
            "y_turbines", jnp.zeros((self.N_turbines,)), units="km"
        )  # y location of the mooring platform in km w.r.t. reference coordinates

        self.add_output(
            "boundary_distances",
            jnp.zeros(self.N_turbines),
            units="km",
        )

    def setup_partials(self):
        """Derivative setup for the OpenMDAO component."""
        # the default (but not preferred!) derivatives are FDM
        self.declare_partials(
            "*",
            "*",
            method="exact",
            rows=np.arange(self.N_turbines),
            cols=np.arange(self.N_turbines),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """Computation for the OpenMDAO component."""

        # unpack the working variables
        x_turbines = inputs["x_turbines"]
        y_turbines = inputs["y_turbines"]

        boundary_distances = (
            ard.utils.geometry.distance_multi_point_to_multi_polygon_ray_casting(
                x_turbines,
                y_turbines,
                boundary_vertices=self.boundary_vertices,
                regions=self.boundary_regions,
            )
        )

        outputs["boundary_distances"] = boundary_distances

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        # unpack the working variables
        x_turbines = inputs["x_turbines"]
        y_turbines = inputs["y_turbines"]

        jacobian = self.distance_multi_point_to_multi_polygon_ray_casting_jac(
            x_turbines, y_turbines, self.boundary_vertices, self.boundary_regions
        )

        partials["boundary_distances", "x_turbines"] = jacobian[0].diagonal()
        partials["boundary_distances", "y_turbines"] = jacobian[1].diagonal()
