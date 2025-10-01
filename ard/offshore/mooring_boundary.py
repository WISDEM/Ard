import numpy as np
import jax.numpy as jnp
import jax
from shapely import boundary

jax.config.update("jax_enable_x64", True)
import ard.utils.geometry
import openmdao.api as om


class FarmBoundaryAnchorDistancePolygon(om.ExplicitComponent):
    """
    A class to return distances between anchors and a polygonal boundary, or
    sets of polygonal boundary regions.

    Options
    -------
    modeling_options : dict
        a modeling options dictionary

    Inputs
    ------
    x_anchors : np.ndarray
        a 1D numpy array indicating the x-dimension locations of the turbines,
        with size `N_turbines` x `N_anchors`
    y_anchors : np.ndarray
        a 1D numpy array indicating the y-dimension locations of the turbines,
        with size `N_turbines` x `N_anchors`

    Outputs
    -------
    anchor_boundary_distances : np.ndarray
        a 1D numpy array indicating the minimum distance of each anchor to the
        polygonal boundary, with size `N_turbines` x `N_anchors`
    """

    def initialize(self):
        """Initialization of the OpenMDAO component."""
        self.options.declare("modeling_options")

    def setup(self):
        """Setup of the OpenMDAO component."""

        # load modeling options
        self.modeling_options = self.options["modeling_options"]
        self.windIO = self.modeling_options["windIO_plant"]
        self.N_turbines = int(self.modeling_options["layout"]["N_turbines"])
        self.N_anchors = int(self.modeling_options["platform"]["N_anchors"])

        # load boundary vertices from windIO file
        if "boundaries" not in self.windIO["site"]:
            raise KeyError(
                "You have requested a boundary but no boundaries were found in the windIO file."
            )
        if "circle" in self.windIO["site"]["boundaries"]:
            raise NotImplementedError(
                "The circular boundaries from windIO have not been implemented here, yet."
            )
        if "polygons" not in self.windIO["site"]["boundaries"]:
            raise KeyError(
                "Currently only polygon boundaries from windIO have been implemented and none were found."
            )

        self.boundary_vertices = [
            np.array(
                [
                    polygon["x"],
                    polygon["y"],
                ]
            ).T
            for polygon in self.windIO["site"]["boundaries"]["polygons"]
        ]
        self.boundary_regions = self.modeling_options.get("boundary", {}).get(
            "turbine_region_assignments",  # get the region assignments from modeling_options, if there
            np.zeros(self.N_turbines, dtype=int),  # default to zero for all turbines
        )

        # prep the jacobian
        self.distance_multi_point_to_multi_polygon_ray_casting_jac = jax.jacfwd(
            ard.utils.geometry.distance_multi_point_to_multi_polygon_ray_casting, [0, 1]
        )

        # set up inputs and outputs for mooring system
        self.add_input(
            "x_anchors", jnp.zeros((self.N_turbines, self.N_anchors)), units="m"
        )  # x location of the mooring anchors in m w.r.t. reference coordinates
        self.add_input(
            "y_anchors", jnp.zeros((self.N_turbines, self.N_anchors)), units="m"
        )  # y location of the mooring anchors in m w.r.t. reference coordinates

        self.add_output(
            "anchor_boundary_distances",
            jnp.zeros(self.N_turbines * self.N_anchors),
            units="m",
        )

    def setup_partials(self):
        """Derivative setup for the OpenMDAO component."""
        # the default (but not preferred!) derivatives are FDM
        self.declare_partials(
            "*",
            "*",
            method="exact",
            # rows=np.arange(self.N_turbines*self.N_anchors),
            # cols=np.arange(self.N_turbines*self.N_anchors),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """Computation for the OpenMDAO component."""

        # unpack the working variables
        x_anchors = inputs["x_anchors"].flatten()
        y_anchors = inputs["y_anchors"].flatten()
        # construct boundary region limits from turbine assignments
        self.boundary_regions_anchors = np.vstack(
            3 * [self.boundary_regions]
        ).T.flatten()

        boundary_distances = (
            ard.utils.geometry.distance_multi_point_to_multi_polygon_ray_casting(
                x_anchors,
                y_anchors,
                boundary_vertices=self.boundary_vertices,
                regions=self.boundary_regions_anchors,
            )
        )

        outputs["anchor_boundary_distances"] = boundary_distances

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        # unpack the working variables
        x_anchors = inputs["x_anchors"].flatten()
        y_anchors = inputs["y_anchors"].flatten()

        # compute and store the jacobian
        jacobian = self.distance_multi_point_to_multi_polygon_ray_casting_jac(
            x_anchors, y_anchors, self.boundary_vertices, self.boundary_regions_anchors
        )
        partials["anchor_boundary_distances", "x_anchors"] = jacobian[0]
        partials["anchor_boundary_distances", "y_anchors"] = jacobian[1]
