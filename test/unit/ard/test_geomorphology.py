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
        self.geomorphology = ard.geomorphology.GeomorphologyGridData()

    def test_check_valid(self):

        # create a mesh and try to upload it
        y_mesh, x_mesh = np.meshgrid([-1.0, 0.0, 1.0], [0.0, 2.0])
        depth_mesh = np.ones_like(x_mesh)
        material_mesh = np.array(
            [["soil", "rock"], ["rock", "soil"], ["rock", "soil"]]
        ).T

        # set up a geomorphology grid data object
        self.geomorphology = ard.geomorphology.GeomorphologyGridData()

        for idx_case in range(4):

            # do a setup that should fail because of check_valid
            with pytest.raises(AssertionError):
                self.geomorphology.set_values(
                    x_mesh_in=x_mesh if idx_case != 0 else x_mesh[:1, :],
                    y_mesh_in=y_mesh if idx_case != 1 else y_mesh[:1, :],
                    depth_mesh_in=depth_mesh if idx_case != 2 else depth_mesh[:1, :],
                    material_mesh_in=(
                        material_mesh if idx_case != 3 else material_mesh[:1, :]
                    ),
                )

            # reset to a legitimate setup
            self.geomorphology.set_values(
                x_mesh_in=x_mesh,
                y_mesh_in=y_mesh,
                depth_mesh_in=depth_mesh,
                material_mesh_in=material_mesh,
            )

            # override one of the values to be invalid
            if idx_case == 0:
                self.geomorphology.x_mesh = self.geomorphology.x_mesh[:1, :]
            elif idx_case == 1:
                self.geomorphology.y_mesh = self.geomorphology.y_mesh[:1, :]
            elif idx_case == 2:
                self.geomorphology.depth_mesh = self.geomorphology.depth_mesh[:1, :]
            else:
                self.geomorphology.material_mesh = self.geomorphology.material_mesh[
                    :1, :
                ]

            # make sure check valid raises an exception
            with pytest.raises(AssertionError):
                assert self.geomorphology.check_valid()

    def test_set_values(self):

        # create a mesh and try to upload it
        y_mesh, x_mesh = np.meshgrid([-1.0, 0.0, 1.0], [0.0, 2.0])
        depth_mesh = np.ones_like(x_mesh)

        # set up a geomorphology grid data object
        self.geomorphology = ard.geomorphology.GeomorphologyGridData()
        # set the values
        self.geomorphology.set_values(
            x_mesh_in=x_mesh,
            y_mesh_in=y_mesh,
            depth_mesh_in=depth_mesh,
        )

        # make sure the values are set in correctly
        assert np.allclose(self.geomorphology.x_mesh, x_mesh)
        assert np.allclose(self.geomorphology.y_mesh, y_mesh)
        assert np.allclose(self.geomorphology.depth_mesh, depth_mesh)
        assert np.allclose(self.geomorphology.get_depth_data(), depth_mesh)
        assert np.all(self.geomorphology.get_shape() == x_mesh.shape)
        assert self.geomorphology.material_mesh.size == 1
        assert self.geomorphology.material_mesh == "soil"  # default value

        assert self.geomorphology.check_valid()  # check if the data is valid

    def test_set_values_material(self):

        # create a mesh and try to upload it
        y_mesh, x_mesh = np.meshgrid([-1.0, 0.0, 1.0], [0.0, 2.0])
        depth_mesh = np.ones_like(x_mesh)
        material_mesh = np.array(
            [["soil", "rock"], ["rock", "soil"], ["rock", "soil"]]
        ).T

        # set up a geomorphology grid data object
        self.geomorphology = ard.geomorphology.GeomorphologyGridData()
        # set the values
        self.geomorphology.set_values(
            x_mesh_in=x_mesh,
            y_mesh_in=y_mesh,
            depth_mesh_in=depth_mesh,
            material_mesh_in=material_mesh,
        )

        # make sure the values are set in correctly
        assert np.allclose(self.geomorphology.x_mesh, x_mesh)
        assert np.allclose(self.geomorphology.y_mesh, y_mesh)
        assert np.allclose(self.geomorphology.depth_mesh, depth_mesh)
        assert np.allclose(self.geomorphology.get_depth_data(), depth_mesh)
        assert np.all(self.geomorphology.material_mesh == material_mesh)
        assert np.all(self.geomorphology.get_shape() == x_mesh.shape)

        assert self.geomorphology.check_valid()  # check if the data is valid

    def test_evaluate_depth_default(self):

        # create a mesh and try to upload it
        y_mesh, x_mesh = np.meshgrid(
            np.linspace(-1.0, 1.0, 5), np.linspace(0.0, 2.0, 5)
        )
        depth_mesh = np.ones_like(x_mesh)

        # set up a geomorphology grid data object
        self.geomorphology = ard.geomorphology.GeomorphologyGridData()
        # set the values
        self.geomorphology.set_values(
            x_mesh_in=x_mesh,
            y_mesh_in=y_mesh,
            depth_mesh_in=depth_mesh,
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
        ard.test_utils.pyrite_validator(
            validation_data,
            Path(__file__).parent / "test_geomorphology_depth_default_pyrite.npz",
            # rewrite=True,  # uncomment to write new pyrite file
        )

    def test_evaluate_depth_gaussian(self):

        # create a mesh and try to upload it
        y_mesh, x_mesh = np.meshgrid([-1.0, 0.0, 1.0], [0.0, 2.0])
        depth_mesh = np.ones_like(x_mesh)

        # set up a geomorphology grid data object
        self.geomorphology = ard.geomorphology.GeomorphologyGridData()
        # set the values
        self.geomorphology.set_values(
            x_mesh_in=x_mesh,
            y_mesh_in=y_mesh,
            depth_mesh_in=depth_mesh,
        )

        with pytest.raises(NotImplementedError):
            # make sure the evaluate_depth method has notimplemented protection
            depth = self.geomorphology.evaluate_depth(
                0.5, 0.5, interp_method="gaussian_process"
            )

    def test_evaluate_depth_nonexistent(self):

        # create a mesh and try to upload it
        y_mesh, x_mesh = np.meshgrid([-1.0, 0.0, 1.0], [0.0, 2.0])
        depth_mesh = np.ones_like(x_mesh)

        # set up a geomorphology grid data object
        self.geomorphology = ard.geomorphology.GeomorphologyGridData()
        # set the values
        self.geomorphology.set_values(
            x_mesh_in=x_mesh,
            y_mesh_in=y_mesh,
            depth_mesh_in=depth_mesh,
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
        self.geomorphology = ard.geomorphology.TopographyGridData()


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
        self.bathymetry = ard.geomorphology.BathymetryGridData()

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
            "min": np.min(self.bathymetry.depth_mesh),
            "max": np.max(self.bathymetry.depth_mesh),
            "mean": np.mean(self.bathymetry.depth_mesh),
            "std": np.std(self.bathymetry.depth_mesh),
        }
        ard.test_utils.pyrite_validator(
            validation_data,
            Path(__file__).parent / "test_geomorphology_bathymetry_pyrite.npz",
            # rewrite=True,  # uncomment to write new pyrite file
        )
