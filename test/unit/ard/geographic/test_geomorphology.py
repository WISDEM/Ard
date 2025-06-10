from pathlib import Path

import numpy as np
import pytest

import ard


@pytest.mark.usefixtures("subtests")
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

    def test_check_valid(self, subtests):

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
            with subtests.test(f"check_valid bad build assertion test {idx_case}"):
                with pytest.raises(ValueError):
                    self.geomorphology.set_data_values(
                        x_data_in=(
                            x_data.copy() if idx_case != 0 else x_data.copy()[:1, :]
                        ),
                        y_data_in=(
                            y_data.copy() if idx_case != 1 else y_data.copy()[:1, :]
                        ),
                        z_data_in=(
                            z_data.copy() if idx_case != 2 else z_data.copy()[:1, :]
                        ),
                    )
                    self.geomorphology.set_material_values(
                        x_material_data_in=(
                            x_data.copy() if idx_case != 0 else x_data.copy()[:1, :]
                        ),
                        y_material_data_in=(
                            y_data.copy() if idx_case != 1 else y_data.copy()[:1, :]
                        ),
                        material_data_in=(
                            material_data.copy()
                            if idx_case != 3
                            else material_data.copy()[:1, :]
                        ),
                    )

            # reset to a legitimate setup
            self.geomorphology.set_data_values(
                x_data_in=x_data.copy(),
                y_data_in=y_data.copy(),
                z_data_in=z_data.copy(),
            )
            self.geomorphology.set_material_values(
                x_material_data_in=x_data.copy(),
                y_material_data_in=y_data.copy(),
                material_data_in=material_data.copy(),
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
            with subtests.test(f"check_valid bad override assertion test {idx_case}"):
                with pytest.raises(ValueError):
                    if idx_case == 3:
                        assert self.geomorphology.check_valid_material()
                    else:
                        assert self.geomorphology.check_valid_geomorphology()

    def test_set_data_values(self, subtests):

        # create a mesh and try to upload it
        y_data, x_data = np.meshgrid([-1.0, 0.0, 1.0], [0.0, 2.0])
        z_data = np.ones_like(x_data)

        # set up a geomorphology grid data object
        self.geomorphology = ard.geographic.GeomorphologyGridData()
        # set the values
        self.geomorphology.set_data_values(
            x_data_in=x_data,
            y_data_in=y_data,
            z_data_in=z_data,
        )

        # make sure the values are set in correctly
        with subtests.test(f"set_data_values data equivalence tests, x"):
            assert np.allclose(self.geomorphology.x_data, x_data)
        with subtests.test(f"set_data_values data equivalence tests, y"):
            assert np.allclose(self.geomorphology.y_data, y_data)
        with subtests.test(f"set_data_values data equivalence tests, z"):
            assert np.allclose(self.geomorphology.z_data, z_data)
        with subtests.test(f"set_data_values data equivalence tests, z getter"):
            assert np.allclose(self.geomorphology.get_z_data(), z_data)
        with subtests.test(f"set_data_values data equivalence tests, get_shape"):
            assert np.all(self.geomorphology.get_shape() == x_data.shape)
        with subtests.test(
            f"set_data_values data equivalence tests, material singleton"
        ):
            assert self.geomorphology.material_data.size == 1
        with subtests.test(f"set_data_values data equivalence tests, material default"):
            assert np.array_equal(
                self.geomorphology.material_data, np.array([["soil"]])
            )  # default value

        with subtests.test(f"check_valid final test"):
            assert (
                self.geomorphology.check_valid_geomorphology()
            )  # check if the data is valid
            assert (
                self.geomorphology.check_valid_material()
            )  # check if the data is valid

    def test_set_data_values_material(self, subtests):

        # create a mesh and try to upload it
        y_data, x_data = np.meshgrid([-1.0, 0.0, 1.0], [0.0, 2.0])
        z_data = np.ones_like(x_data)
        material_data = np.array(
            [["soil", "rock"], ["rock", "soil"], ["rock", "soil"]]
        ).T

        # set up a geomorphology grid data object
        self.geomorphology = ard.geographic.GeomorphologyGridData()
        # set the values
        self.geomorphology.set_data_values(
            x_data_in=x_data,
            y_data_in=y_data,
            z_data_in=z_data,
        )
        self.geomorphology.set_material_values(
            x_material_data_in=x_data,
            y_material_data_in=y_data,
            material_data_in=material_data,
        )

        # make sure the values are set in correctly
        with subtests.test(f"set_data_values data equivalence tests, x"):
            assert np.allclose(self.geomorphology.x_data, x_data)
        with subtests.test(f"set_data_values data equivalence tests, y"):
            assert np.allclose(self.geomorphology.y_data, y_data)
        with subtests.test(f"set_data_values data equivalence tests, z"):
            assert np.allclose(self.geomorphology.z_data, z_data)
        with subtests.test(f"set_data_values data equivalence tests, z getter"):
            assert np.allclose(self.geomorphology.get_z_data(), z_data)
        with subtests.test(f"set_data_values data equivalence tests, material"):
            assert np.all(self.geomorphology.material_data == material_data)
        with subtests.test(f"set_data_values data equivalence tests, get_shape"):
            assert np.all(self.geomorphology.get_shape() == x_data.shape)

        with subtests.test(f"check_valid final test"):
            assert (
                self.geomorphology.check_valid_geomorphology()
            )  # check if the data is valid
            assert (
                self.geomorphology.check_valid_material()
            )  # check if the data is valid

    def test_evaluate_default(self, subtests):

        # create a mesh and try to upload it
        y_data, x_data = np.meshgrid(
            np.linspace(-1.0, 1.0, 5), np.linspace(0.0, 2.0, 5)
        )
        fun_data = lambda x, y: np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        z_data = fun_data(x_data, y_data)

        # set up a geomorphology grid data object
        self.geomorphology = ard.geographic.GeomorphologyGridData()
        # set the values
        self.geomorphology.set_data_values(
            x_data_in=x_data,
            y_data_in=y_data,
            z_data_in=z_data,
        )

        # grab the depth at points in the mesh domain
        y_sample, x_sample = np.meshgrid(
            [-0.75, -0.5, -0.25, 0.25, 0.5, 0.75], [0.5, 1.5]
        )
        depth_sample = self.geomorphology.evaluate(
            x_sample.flatten(), y_sample.flatten()
        )
        # check that the values match a pyrite file
        with subtests.test(f"evaluate interpolator pyrite test"):
            validation_data = {
                "depth_sample": depth_sample,
            }
            ard.utils.test_utils.pyrite_validator(
                validation_data,
                Path(__file__).parent / "test_geomorphology_depth_default_pyrite.npz",
                # rewrite=True,  # uncomment to write new pyrite file
            )

        # check that the gradients are correct by matching a pyrite file
        with subtests.test(f"evaluate derivatives pyrite test"):
            dx_depth_sample, dy_depth_sample = self.geomorphology.evaluate(
                x_sample.flatten(),
                y_sample.flatten(),
                return_derivs=True,
            )
            validation_data = {
                "dx_depth_sample": dx_depth_sample,
                "dy_depth_sample": dy_depth_sample,
            }
            ard.utils.test_utils.pyrite_validator(
                validation_data,
                Path(__file__).parent
                / "test_geomorphology_depth_default_gradients_pyrite.npz",
                # rewrite=True,  # uncomment to write new pyrite file
            )

        # check that the interpolator device is not changed from one run to another
        with subtests.test(f"evaluate interpolator caching test"):
            id_initial = id(self.geomorphology._interpolator_device)
            depth_sample_2 = self.geomorphology.evaluate(
                x_sample.flatten(), y_sample.flatten()
            )
            # escalating tests
            assert np.allclose(depth_sample, depth_sample_2)  # same return
            assert id_initial == id(
                self.geomorphology._interpolator_device
            )  # same identity

    def test_evaluate_gaussian(self, subtests):

        # create a mesh and try to upload it
        y_data, x_data = np.meshgrid([-1.0, 0.0, 1.0], [0.0, 2.0])
        z_data = np.ones_like(x_data)

        # set up a geomorphology grid data object
        self.geomorphology = ard.geographic.GeomorphologyGridData()
        # set the values
        self.geomorphology.set_data_values(
            x_data_in=x_data,
            y_data_in=y_data,
            z_data_in=z_data,
        )

        with subtests.test(f"notimplemented test"):
            with pytest.raises(NotImplementedError):
                # make sure the evaluate method has notimplemented protection
                depth = self.geomorphology.evaluate(
                    0.5, 0.5, interp_method="gaussian_process"
                )

    def test_evaluate_nonexistent(self, subtests):

        # create a mesh and try to upload it
        y_data, x_data = np.meshgrid([-1.0, 0.0, 1.0], [0.0, 2.0])
        z_data = np.ones_like(x_data)

        # set up a geomorphology grid data object
        self.geomorphology = ard.geographic.GeomorphologyGridData()
        # set the values
        self.geomorphology.set_data_values(
            x_data_in=x_data,
            y_data_in=y_data,
            z_data_in=z_data,
        )

        with subtests.test(f"notimplemented test"):
            with pytest.raises(NotImplementedError):
                # make sure the evaluate method has notimplemented protection
                depth = self.geomorphology.evaluate(0.5, 0.5, interp_method="magic")


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

    def test_load_moorpy_bathymetry(self, subtests):

        # path to the example MoorPy bathymetry grid file
        file_bathy = (
            Path(ard.__file__).parents[1]
            / "examples"
            / "data"
            / "offshore"
            / "GulfOfMaine_bathymetry_100x99.txt"
        )

        # load the bathymetry data
        self.bathymetry.load_moorpy_bathymetry(file_bathymetry=file_bathy)

        # check the shape of the data
        with subtests.test(f"moorpy load shape test"):
            assert np.all(self.bathymetry.get_shape() == np.array([100, 99]))

        # make sure the data matches the statistical properties of the original data

        with subtests.test(f"moorpy load statistics test: min"):
            assert np.isclose(np.min(self.bathymetry.z_data), 160.069000000000000)
        with subtests.test(f"moorpy load statistics test: max"):
            assert np.isclose(np.max(self.bathymetry.z_data), 183.896000000000000)
        with subtests.test(f"moorpy load statistics test: mean"):
            assert np.isclose(np.mean(self.bathymetry.z_data), 172.50993464646467)
        with subtests.test(f"moorpy load statistics test: std"):
            assert np.isclose(np.std(self.bathymetry.z_data), 4.555364127422273)

    def test_load_moorpy_soil(self, subtests):

        # path to the example MoorPy bathymetry grid file
        file_bathy = (
            Path(ard.__file__).parents[1]
            / "examples"
            / "data"
            / "offshore"
            / "GulfOfMaine_bathymetry_100x99.txt"
        )
        # path to the example MoorPy bathymetry grid file
        file_soil = (
            Path(ard.__file__).parents[1]
            / "examples"
            / "data"
            / "offshore"
            / "GulfOfMaine_soil_100x99.txt"
        )

        # load the bathymetry data
        self.bathymetry.load_moorpy_bathymetry(file_bathymetry=file_bathy)

        # load the soil data
        self.bathymetry.load_moorpy_soil(file_soil=file_soil)

        # check the shape of the data
        with subtests.test(f"moorpy load shape test"):
            assert np.all(self.bathymetry.get_material_shape() == np.array([100, 99]))
            assert np.all(self.bathymetry.material_data.shape == np.array([100, 99]))
            assert np.all(self.bathymetry.x_material_data.shape == np.array([100, 99]))
            assert np.all(self.bathymetry.y_material_data.shape == np.array([100, 99]))

        # make sure the data matches the some properties of the original data
        with subtests.test(f"moorpy soil coordinate checks: x min"):
            assert np.isclose(np.min(self.bathymetry.x_material_data), -4757.64)
        with subtests.test(f"moorpy soil coordinate checks: x max"):
            assert np.isclose(np.max(self.bathymetry.x_material_data), 3340.58)
        with subtests.test(f"moorpy soil coordinate checks: y min"):
            assert np.isclose(np.min(self.bathymetry.y_material_data), -4336.04)
        with subtests.test(f"moorpy soil coordinate checks: y max"):
            assert np.isclose(np.max(self.bathymetry.y_material_data), 3453.22)
        with subtests.test(f"moorypy type count checks"):
            assert int(np.sum(self.bathymetry.material_data == "mud_0")) == 646
            assert int(np.sum(self.bathymetry.material_data == "mud_1")) == 2792
            assert int(np.sum(self.bathymetry.material_data == "mud_2")) == 3090
            assert int(np.sum(self.bathymetry.material_data == "mud_3")) == 2656
            assert int(np.sum(self.bathymetry.material_data == "mud_4")) == 716
        # purely making sure the data is coming in correctly

        # make sure all of the types appear
        with subtests.test(f"moorpy soil set"):
            # print(f"DEBUG!!!!! material_data: {self.bathymetry.material_data}")
            umd = np.unique(self.bathymetry.material_data).tolist()
            for mv in ["mud_4", "mud_3", "mud_2", "mud_1", "mud_0"]:
                assert mv in umd
                umd.remove(mv)
            assert not umd  # no extra entries

        # make sure the types and their units are pulled out correctly
        with subtests.test(f"moorpy soil type headers"):

            quantities_expected = [
                "Class",
                "Gamma",
                "Su0",
                "k",
                "alpha",
                "phi",
                "UCS",
                "Em",
            ]
            units_expected = dict.fromkeys(
                quantities_expected,
                ["name", "kN/m^3", "kPa", "kPa/m", "-", "deg", "MPa", "MPa"],
            )

            # make sure material types that appear match expected, unit storage
            for k, v in self.bathymetry.material_types.items():
                for kv in v.keys():
                    assert kv in quantities_expected  # expected values
                    assert (
                        kv in self.bathymetry.material_type_units.keys()
                    )  # unit holder
                for k_quantity in v.keys():  # each QoI for a material
                    assert (
                        k_quantity in quantities_expected
                    )  # make sure the quantity is eqpected

            # make sure the units for the thing match up
            for k, v in units_expected.items():
                assert k in self.bathymetry.material_type_units.keys()
                assert self.bathymetry.material_type_units[k] in v
