from pathlib import Path

import numpy as np
import pytest

import ard


class TestGeomorphologyGridData:
    """
    Test the GeomorphologyGridData class.

    This class tests the basic functionality of the GeomorphologyGridData class.
    It checks the following:
    - the ability to set values for the x, y, depth, and material meshes.
    - the ability to check if the data is valid.
    - the ability to get the shape of the data.
    - the ability to set values for the data.
    """

    def setup_method(self):

        # create a geomorphology object before each test
        self.geomorphology = ard.geographic.GeomorphologyGridData()

    def test_check_valid(self):

        # create a mesh and try to upload it
        y_data, x_data = np.meshgrid([-1.0, 0.0, 1.0], [0.0, 2.0])
        z_data = np.ones_like(x_data)
        material_data = np.array(
            [["soil", "rock"], ["rock", "soil"], ["rock", "soil"]]
        ).T

        # set up a geomorphology grid data object
        self.geomorphology = ard.geographic.GeomorphologyGridData()

        for idx_case in range(4):

            # do a setup that should fail because of check_valid
            with pytest.raises(AssertionError):
                self.geomorphology.set_values(
                    x_data_in=x_data if idx_case != 0 else x_data[:1, :],
                    y_data_in=y_data if idx_case != 1 else y_data[:1, :],
                    z_data_in=z_data if idx_case != 2 else z_data[:1, :],
                    material_data_in=(
                        material_data if idx_case != 3 else material_data[:1, :]
                    ),
                )

            # reset to a legitimate setup
            self.geomorphology.set_values(
                x_data_in=x_data,
                y_data_in=y_data,
                z_data_in=z_data,
                material_data_in=material_data,
            )

            # override one of the values to be invalid
            if idx_case == 0:
                self.geomorphology.x_data = self.geomorphology.x_data[:1, :]
            elif idx_case == 1:
                self.geomorphology.y_data = self.geomorphology.y_data[:1, :]
            elif idx_case == 2:
                self.geomorphology.z_data = self.geomorphology.z_data[:1, :]
            else:
                self.geomorphology.material_data = self.geomorphology.material_data[
                    :1, :
                ]

            # make sure check valid raises an exception
            with pytest.raises(AssertionError):
                assert self.geomorphology.check_valid()

    def test_set_values(self):

        # create a mesh and try to upload it
        y_data, x_data = np.meshgrid([-1.0, 0.0, 1.0], [0.0, 2.0])
        z_data = np.ones_like(x_data)

        # set up a geomorphology grid data object
        self.geomorphology = ard.geographic.GeomorphologyGridData()
        # set the values
        self.geomorphology.set_values(
            x_data_in=x_data,
            y_data_in=y_data,
            z_data_in=z_data,
        )

        # make sure the values are set in correctly
        assert np.allclose(self.geomorphology.x_data, x_data)
        assert np.allclose(self.geomorphology.y_data, y_data)
        assert np.allclose(self.geomorphology.z_data, z_data)
        assert np.allclose(self.geomorphology.get_z_data(), z_data)
        assert np.all(self.geomorphology.get_shape() == x_data.shape)
        assert self.geomorphology.material_data.size == 1
        assert self.geomorphology.material_data == "soil"  # default value

        assert self.geomorphology.check_valid()  # check if the data is valid

    def test_set_values_material(self):

        # create a mesh and try to upload it
        y_data, x_data = np.meshgrid([-1.0, 0.0, 1.0], [0.0, 2.0])
        z_data = np.ones_like(x_data)
        material_data = np.array(
            [["soil", "rock"], ["rock", "soil"], ["rock", "soil"]]
        ).T

        # set up a geomorphology grid data object
        self.geomorphology = ard.geographic.GeomorphologyGridData()
        # set the values
        self.geomorphology.set_values(
            x_data_in=x_data,
            y_data_in=y_data,
            z_data_in=z_data,
            material_data_in=material_data,
        )

        # make sure the values are set in correctly
        assert np.allclose(self.geomorphology.x_data, x_data)
        assert np.allclose(self.geomorphology.y_data, y_data)
        assert np.allclose(self.geomorphology.z_data, z_data)
        assert np.allclose(self.geomorphology.get_z_data(), z_data)
        assert np.all(self.geomorphology.material_data == material_data)
        assert np.all(self.geomorphology.get_shape() == x_data.shape)

        assert self.geomorphology.check_valid()  # check if the data is valid

    def test_evaluate_depth_default(self):

        # create a mesh and try to upload it
        y_data, x_data = np.meshgrid(
            np.linspace(-1.0, 1.0, 5), np.linspace(0.0, 2.0, 5)
        )
        z_data = np.ones_like(x_data)

        # set up a geomorphology grid data object
        self.geomorphology = ard.geographic.GeomorphologyGridData()
        # set the values
        self.geomorphology.set_values(
            x_data_in=x_data,
            y_data_in=y_data,
            z_data_in=z_data,
        )

        # grab the depth at points in the mesh domain
        y_sample, x_sample = np.meshgrid(
            [-0.75, -0.5, -0.25, 0.25, 0.5, 0.75], [0.5, 1.5]
        )
        depth_sample = self.geomorphology.evaluate_depth(
            x_sample.flatten(), y_sample.flatten()
        )
        # check that the values match a pyrite file
        validation_data = {
            "depth_sample": depth_sample,
        }
        ard.utils.test_utils.pyrite_validator(
            validation_data,
            Path(__file__).parent / "test_geomorphology_depth_default_pyrite.npz",
            # rewrite=True,  # uncomment to write new pyrite file
        )

    def test_evaluate_depth_gaussian(self):

        # create a mesh and try to upload it
        y_data, x_data = np.meshgrid([-1.0, 0.0, 1.0], [0.0, 2.0])
        z_data = np.ones_like(x_data)

        # set up a geomorphology grid data object
        self.geomorphology = ard.geographic.GeomorphologyGridData()
        # set the values
        self.geomorphology.set_values(
            x_data_in=x_data,
            y_data_in=y_data,
            z_data_in=z_data,
        )

        with pytest.raises(NotImplementedError):
            # make sure the evaluate_depth method has notimplemented protection
            depth = self.geomorphology.evaluate_depth(
                0.5, 0.5, interp_method="gaussian_process"
            )

    def test_evaluate_depth_nonexistent(self):

        # create a mesh and try to upload it
        y_data, x_data = np.meshgrid([-1.0, 0.0, 1.0], [0.0, 2.0])
        z_data = np.ones_like(x_data)

        # set up a geomorphology grid data object
        self.geomorphology = ard.geographic.GeomorphologyGridData()
        # set the values
        self.geomorphology.set_values(
            x_data_in=x_data,
            y_data_in=y_data,
            z_data_in=z_data,
        )

        with pytest.raises(NotImplementedError):
            # make sure the evaluate_depth method has notimplemented protection
            depth = self.geomorphology.evaluate_depth(0.5, 0.5, interp_method="magic")


class TestTopographyGridData(TestGeomorphologyGridData):
    """
    Test the TopographyGridData class.

    This class tests the basic functionality of the TopographyGridData class.
    It inherits from the TestGeomorphologyGridData class and runs all of the
    general tests for the GeomorphologyGridData class.
    It also should test the specialized functionality of the TopographyGridData
    class, which is currently null.
    """

    def setup_method(self):

        # create a specialized geomorphology object before each test
        self.geomorphology = ard.geographic.TopographyGridData()


class TestBathymetryGridData(TestGeomorphologyGridData):
    """
    Test the BathymetryGridData class.

    This class tests the basic functionality of the BathymetryGridData class.
    It inherits from the TestGeomorphologyGridData class and runs all of the
    general tests for the GeomorphologyGridData class.
    It also should test the specialized functionality of the BathymetryGridData
    class, which includes:
    - MoorPy bathymetry data loading
    """

    def setup_method(self):

        # create a specialized geomorphology object before each test
        self.bathymetry = ard.geographic.BathymetryGridData()

    def test_load_moorpy_bathymetry(self):

        # path to the example MoorPy bathymetry grid file
        file_bathy = (
            Path(ard.__file__).parents[1]
            / "examples"
            / "data"
            / "GulfOfMaine_bathymetry_100x99.txt"
        )

        # load the bathymetry data
        self.bathymetry.load_moorpy_bathymetry(file_bathymetry=file_bathy)

        # check the shape of the data
        assert np.all(self.bathymetry.get_shape() == np.array([100, 99]))

        # make sure the data matches the statistical properties of the original data
        validation_data = {
            "min": np.min(self.bathymetry.z_data),
            "max": np.max(self.bathymetry.z_data),
            "mean": np.mean(self.bathymetry.z_data),
            "std": np.std(self.bathymetry.z_data),
        }
        ard.utils.test_utils.pyrite_validator(
            validation_data,
            Path(__file__).parent / "test_geomorphology_bathymetry_pyrite.npz",
            # rewrite=True,  # uncomment to write new pyrite file
        )
