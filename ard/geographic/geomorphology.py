from os import PathLike
from pathlib import Path

import numpy as np
from scipy.interpolate import SmoothBivariateSpline

import openmdao.api as om


class GeomorphologyGridData:
    """
    A class to represent gridded geomorphology data for a given wind farm site
    domain.

    Represents either bathymetry data for offshore sites or topography data for
    onshore sites.
    """

    # alias for meshed data, and promote dimension to 2
    x_data = np.atleast_2d([0.0])  # x location in km
    y_data = np.atleast_2d([0.0])  # y location in km
    z_data = np.atleast_2d([0.0])  # depth in m

    # alias for meshed material data, promote dimension to 2
    x_material_data = np.atleast_2d([0.0])  # x location in km of material datapoint
    y_material_data = np.atleast_2d([0.0])  # y location in km of material datapoint
    material_data = np.atleast_2d(["soil"])  # bed material at each point

    sea_level = 0.0  # sea level in m

    _interpolator_device = None  # placeholder for interpolator (for depth evaluation)

    def check_valid_geomorphology(self):
        assert self.x_data.ndim == 2, "data must be 2D"  # make sure it's 2D first

        assert np.all(
            self.x_data.shape == self.y_data.shape
        ), "x and y data must be the same shape"
        assert np.all(
            self.x_data.shape == self.z_data.shape
        ), "x and depth data must be the same shape"

        return True

    def check_valid_material(self):
        assert (
            self.x_material_data.ndim == 2
        ), "data must be 2D"  # make sure it's 2D first

        assert np.all(
            self.x_material_data.shape == self.y_material_data.shape
        ), "x and y material data must be the same shape"
        assert np.all(self.x_material_data.shape == self.material_data.shape) or (
            self.material_data.size == 1
        ), "x and material data must be the same shape or material data must be a singleton"

        return True

    def get_shape(self):
        """
        Get the shape of the geomorphology data.

        Returns
        -------
        tuple
            The shape of the geomorphology data.
        """

        self.check_valid_geomorphology()  # ensure that the current data is valid
        return self.z_data.shape  # data shape

    def get_material_shape(self):
        """
        Get the shape of the material data.

        Returns
        -------
        tuple
            The shape of the material data.
        """

        self.check_valid_material()  # ensure that the current data is valid
        return self.material_data.shape  # data shape

    def set_data_values(
        self,
        x_data_in,
        y_data_in,
        z_data_in,
    ):
        """
        Set the values of the geomorphology data.

        Parameters
        ----------
        x_data_in : np.ndarray
            A 2D numpy array indicating the x-dimension locations of the points.
        y_data_in : np.ndarray
            A 2D numpy array indicating the y-dimension locations of the points.
        z_data_in : np.ndarray
            A 2D numpy array indicating the depth at each point.
        material_data_in : np.ndarray, optional
            A 2D numpy array indicating the bed material at each point.
        """

        # set the values that are handed in
        self.x_data = x_data_in.copy()
        self.y_data = y_data_in.copy()
        self.z_data = z_data_in.copy()

        self.check_valid_geomorphology()  # ensure that the input data is valid

    def set_material_values(
        self,
        x_material_data_in,
        y_material_data_in,
        material_data_in,
    ):
        """
        Set the values of the material data.

        Parameters
        ----------
        x_material_data_in : np.ndarray
            A 2D numpy array indicating the x-dimension locations of the points.
        y_material_data_in : np.ndarray
            A 2D numpy array indicating the y-dimension locations of the points.
        material_data_in : np.ndarray
            A 2D numpy array indicating the bed material at each point.
        """

        # set the values that are handed in
        self.x_material_data = x_material_data_in.copy()
        self.y_material_data = y_material_data_in.copy()
        self.material_data = material_data_in.copy()

        self.check_valid_material()  # ensure that the input data is valid

    def get_z_data(self):
        """Get the depth at a given location."""
        return self.z_data

    def get_material_data(self):
        """Get the material data at a given location."""
        return self.material_data

    def evaluate(
        self,
        x_query,
        y_query,
        return_derivs=False,
        interp_method="spline",
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
            the depth at the given locations if return_derivs is False
        tuple
            the derivatives if return_derivs is True
        """

        x_query = np.atleast_1d(x_query)  # ensure x_query is a 1D array
        y_query = np.atleast_1d(y_query)  # ensure y_query is a 1D array

        if interp_method == "spline":
            # or, smooth bivariate spline from scipy implementation

            if self._interpolator_device is None:
                # create and alias the interpolator object
                interpolator_sbs = self._interpolator_device = SmoothBivariateSpline(
                    self.x_data.flatten(),
                    self.y_data.flatten(),
                    self.z_data.flatten(),
                    bbox=[
                        np.min(self.x_data),
                        np.max(self.x_data),
                        np.min(self.y_data),
                        np.max(self.y_data),
                    ],
                )
            else:
                # assert the interpolator is of the smoothbivariate spline type or its parent class
                assert isinstance(
                    self._interpolator_device, SmoothBivariateSpline
                ), "interpolator must be a SmoothBivariateSpline"
                # alias
                interpolator_sbs = self._interpolator_device

            # make interpolation
            if return_derivs:
                # and if desired, take its derivatives
                dz_dx = interpolator_sbs(x_query, y_query, grid=False, dx=1, dy=0)
                dz_dy = interpolator_sbs(x_query, y_query, grid=False, dx=0, dy=1)
                return (dz_dx, dz_dy)  # and return
            else:
                z_query = interpolator_sbs(x_query, y_query, grid=False)
                return z_query  # just return

        else:
            raise NotImplementedError(
                f"{interp_method} interpolation scheme for evaluate not implemented yet. -cfrontin"
            )


class BathymetryGridData(GeomorphologyGridData):
    """
    A class to represent gridded bathymetry data for a given wind farm site
    domain.

    Represents the bathymetry data for offshore sites. Can be used for floating
    mooring system anchors or for fixed-bottom foundations. Should specialize
    geomorphology data for bathymetry-specific considerations.
    """

    def load_moorpy_soil(self, file_soil: PathLike):
        """
        Load soil data from a MoorPy soil file.

        Experimental: reader may not be able to read validly formatted file in
        in the presence of unanticipated comments, whitespace, etc.

        Parameters
        ----------
        file_soil : PathLike
            The path to the soil data file
        """

        # create placholder objects in function local scope
        grid_soil = None
        x_coord = None
        y_coord = None

        with open(file_soil, "r") as f_soil:
            idx_y = 0  # indexer for y coordinate as file is read

            # iterate over lines in the soil file
            for idx_line, line in enumerate(f_soil.readlines()):

                if idx_line == 0:  # moorpy header line must be first
                    assert line.startswith("--- MoorPy Soil Input File ---")
                    continue
                if idx_line == 1:  # next line defines the grid size in x
                    assert line.startswith("nGridX")  # guarantee this is the case
                    nGridX = int(line.split()[1])  # extract the number
                    x_coord = np.zeros((nGridX,))  # prepare a coord array
                    continue

    def load_moorpy_bathymetry(self, file_bathymetry: PathLike):
        """
        Load bathymetry data from a MoorPy bathymetry grid file.

        Experimental: reader may not be able to read validly formatted file in
        in the presence of unanticipated comments, whitespace, etc.

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
        self.y_data, self.x_data = np.meshgrid(y_coord, x_coord)
        self.z_data = grid_bathy

        self.check_valid_geomorphology()  # make sure the loaded file is legit before exiting


class TopographyGridData(GeomorphologyGridData):
    """
    A class to represent gridded terrain data for a given wind farm site domain.

    Represents the terrain data for onshore sites. Should specialize
    geomorphology data for topography-specific considerations.
    """

    pass
