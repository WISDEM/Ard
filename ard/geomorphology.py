import numpy as np

import openmdao.api as om


class GeomorphologyData:
    """
    A class to represent geomorphology data for a given wind farm site domain.

    Represents either bathymetry data for offshore sites or topography data for
    onshore sites.
    """

    x_mesh = np.atleast_2d([0.0])  # x location in km
    y_mesh = np.atleast_2d([0.0])  # y location in km
    depth_mesh = np.atleast_2d([0.0])  # depth in m
    sea_level = 0.0  # sea level in m

    material_mesh = np.array(["soil"])  # bed material at each point

    def is_valid(self):
        """Check if the geomorphology data is valid."""
        assert np.all(
            x_mesh.shape == y_mesh.shape
        ), "x and y mesh must be the same shape"
        assert np.all(
            x_mesh.shape == depth_mesh.shape
        ), "x and depth mesh must be the same shape"
        assert (
            np.all(x_mesh.shape == material_mesh.shape) or len(material_mesh) == 1
        ), "x and material mesh must be the same shape or material mesh must be a singleton"

    def set_values(
        self,
        x_mesh_in,
        y_mesh_in,
        depth_mesh_in,
        material_mesh_in=None,
    ):
        """
        Set the values of the geomorphology data.

        Parameters
        ----------
        x_mesh_in : np.ndarray
            A 2D numpy array indicating the x-dimension locations of the points.
        y_mesh_in : np.ndarray
            A 2D numpy array indicating the y-dimension locations of the points.
        depth_mesh_in : np.ndarray
            A 2D numpy array indicating the depth at each point.
        material_mesh_in : np.ndarray, optional
            A 2D numpy array indicating the bed material at each point.
        """

        # set the values that are handed in
        self.x_mesh = x_mesh_in.copy()
        self.y_mesh = y_mesh_in.copy()
        self.depth_mesh = depth_mesh_in.copy()
        if material_mesh_in is not None:
            self.material_mesh = material_mesh_in.copy()

        assert self.is_valid()  # ensure that the input data is valid

    def get_depth_data(self):
        """Get the depth at a given location."""
        return self.depth_mesh

    def evaluate_depth(self, x_query, y_query, return_derivs=False):
        """
        Evaluate the depth at a given location.

        Parameters
        ----------
        x_query : np.array
            The x locations to sample in km
        y_query : np.array
            The y locations to sample in km

        Returns
        -------
        np.array
            The depth at the given locations
        """
        raise NotImplementedError(
            "interpolation scheme for evaluate_depth not implemented yet. -cfrontin"
        )


class BathymetryData(GeomorphologyData):
    """
    A class to represent bathymetry data for a given wind farm site domain.

    Represents the bathymetry data for offshore sites. Can be used for floating
    mooring system anchors or for fixed-bottom foundations. Should specialize
    geomorphology data for bathymetry-specific considerations.
    """

    pass


class TopographyData(GeomorphologyData):
    """
    A class to represent terrain data for a given wind farm site domain.

    Represents the terrain data for onshore sites. Should specialize
    geomorphology data for topography-specific considerations.
    """

    pass
