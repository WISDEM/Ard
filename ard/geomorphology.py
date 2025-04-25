from os import PathLike
from pathlib import Path

import numpy as np

import openmdao.api as om


class GeomorphologyGridData:
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

    def check_valid(self):
        """Check if the geomorphology data is valid."""
        assert np.all(
            self.x_mesh.shape == self.y_mesh.shape
        ), "x and y mesh must be the same shape"
        assert np.all(
            self.x_mesh.shape == self.depth_mesh.shape
        ), "x and depth mesh must be the same shape"
        assert (
            np.all(self.x_mesh.shape == self.material_mesh.shape)
            or (self.material_mesh.size == 1)
        ), "x and material mesh must be the same shape or material mesh must be a singleton"

        return True

    def get_shape(self):
        """
        Get the shape of the geomorphology data.

        Returns
        -------
        tuple
            The shape of the geomorphology data.
        """

        self.check_valid()  # ensure that the current data is valid

        return self.depth_mesh.shape  # x, y coordinates

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

        self.check_valid()  # ensure that the input data is valid

    def get_depth_data(self):
        """Get the depth at a given location."""
        return self.depth_mesh

    def evaluate_depth(
        self,
        x_query,
        y_query,
        return_derivs=False,
        interp_method="gaussian_process",
    ):
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

        if interp_method == "gaussian_process":
            raise NotImplementedError(
                f"{interp_method} interpolation scheme for evaluate_depth not implemented yet. -cfrontin"
            )
        else:
            raise NotImplementedError(
                f"{interp_method} interpolation scheme for evaluate_depth not implemented yet. -cfrontin"
            )


class BathymetryGridData(GeomorphologyGridData):
    """
    A class to represent bathymetry data for a given wind farm site domain.

    Represents the bathymetry data for offshore sites. Can be used for floating
    mooring system anchors or for fixed-bottom foundations. Should specialize
    geomorphology data for bathymetry-specific considerations.
    """

    def load_moorpy_bathymetry(self, file_bathymetry: PathLike):
        """
        Load bathymetry data from a MoorPy bathymetry grid file.

        Experimental: reader may not be able to read validly formatted comments,
        whitespace, etc.

        Parameters
        ----------
        file_bathymetry : str
            The path to the bathymetry data file
        """

        # create placeholder objects in function local scope
        grid_bathy = None
        x_coord = None
        y_coord = None

        with open(file_bathymetry, "r") as f_bathy:
            idx_y = 0  # indexer for y coordinate as file is read

            # iterate over lines in the bathymetry file
            for idx_line, line in enumerate(f_bathy.readlines()):

                if idx_line == 0:  # moorpy header line must be first
                    assert line.startswith("--- MoorPy Bathymetry Input File ---")
                    continue
                if idx_line == 1:  # next line defines the grid size in x
                    assert line.startswith("nGridX")  # guarantee this is the case
                    nGridX = int(line.split()[1])  # extract the number
                    x_coord = np.zeros((nGridX,))  # prepare a coord array
                    continue
                if idx_line == 2:  # next line defines the grid size in y
                    assert line.startswith("nGridY")  # guarantee this is the case
                    nGridY = int(line.split()[1])  # extract the number
                    y_coord = np.zeros((nGridY,))  # prepare a coord array
                    grid_bathy = np.zeros((nGridX, nGridY))  # prepare a grid
                    continue

                if idx_line == 3:  # next line should define the x coordinates
                    x_coord_tgt = [float(x) for x in line.split()]  # extract
                    assert len(x_coord_tgt) == nGridX  # verify length
                    x_coord = np.array(x_coord_tgt)  # convert to array
                    continue

                if (
                    idx_line > 3
                ):  # all other lines should be y coordinate then gridpoint data
                    if not line.strip():
                        continue  # if the line is empty or whitespace, skip it

                    y_coord_tgt = float(line.split()[0])  # extract the y coordinate
                    bathy_row_tgt = [
                        float(b) for b in line.split()[1:]
                    ]  # extract the bathymetry data
                    assert len(bathy_row_tgt) == nGridX  # verify length
                    y_coord[idx_y] = y_coord_tgt  # set the y coordinate
                    grid_bathy[:, idx_y] = bathy_row_tgt  # set the bathymetry data
                    idx_y += 1  # increment the y indexer
            assert idx_y == nGridY  # verify that all y coordinates were read

        # save into the geomorphology data object
        self.y_mesh, self.x_mesh = np.meshgrid(y_coord, x_coord)
        self.depth_mesh = grid_bathy

        self.check_valid()  # make sure the loaded file is legit before exiting


class TopographyGridData(GeomorphologyGridData):
    """
    A class to represent terrain data for a given wind farm site domain.

    Represents the terrain data for onshore sites. Should specialize
    geomorphology data for topography-specific considerations.
    """

    pass
