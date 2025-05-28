from pathlib import Path
import openmdao.api as om
import ard
import ard.utils.io
from ard.cost.surrogate_turbine_spacing import PrimarySpacingSurrogate
import pytest

import openmdao.api as om
from ard.cost.surrogate_turbine_spacing import LandBOSSEWithSurrogate
import pytest


class TestLandBOSSEWithSurrogate:
    def setup_method(self):

        # specify the configuration/specification files to use
        filename_turbine_spec = (
            Path(ard.__file__).parents[1]
            / "examples"
            / "data"
            / "turbine_spec_IEA-3p4-130-RWT.yaml"
        ).absolute()  # toolset generalized turbine specification

        # load the turbine specification
        data_turbine = ard.utils.io.load_turbine_spec(filename_turbine_spec)

        # set up the modeling options
        modeling_options = {
            "farm": {
                "N_turbines": 25,
            },
            "turbine": data_turbine,
        }

        # Create the problem
        prob = om.Problem()

        # Add the LandBOSSEWithSurrogate group
        prob.model.add_subsystem(
            "landbosse_group",
            LandBOSSEWithSurrogate(modeling_options=modeling_options),
            promotes=["*"],
        )

        # Set up the problem
        prob.setup()

        # Set the input value
        prob.set_val("total_length_cables", 1000.0)  # Total cable length in meters

        # Run the model
        prob.run_model()

        self.prob = prob
        self.modeling_options = modeling_options

    def test_primary_turbine_spacing(self):
        """Test the primary turbine spacing calculation."""
        # Check the output value
        primary_turbine_spacing = self.prob.get_val(
            "spacing_surrogate.primary_turbine_spacing_diameters"
        )
        expected_spacing = 1000.0 / (
            self.modeling_options["farm"]["N_turbines"]
            * self.modeling_options["turbine"]["geometry"]["diameter_rotor"]
        )
        assert primary_turbine_spacing == pytest.approx(expected_spacing, abs=1e-12)

    def test_internal_turbine_spacing(self):
        """Test that the internal turbine spacing is passed correctly to LandBOSSE."""
        # Check that LandBOSSE receives the correct input
        internal_turbine_spacing = self.prob.get_val(
            "internal_turbine_spacing_rotor_diameters"
        )
        primary_turbine_spacing = self.prob.get_val(
            "spacing_surrogate.primary_turbine_spacing_diameters"
        )
        assert internal_turbine_spacing == pytest.approx(
            primary_turbine_spacing, abs=1e-12
        )

    # def test_partial_derivatives(self):
    #     """Test the partial derivatives."""
    #     # Check the partial derivatives
    #     partials = self.prob.check_partials(out_stream=None, method="fd")
    #     total_length_cables_partials = partials["landbosse_group"][
    #         ("total_capex", "total_length_cables")
    #     ]["J_fwd"]
    #     expected_partial = 1.0 / (
    #         self.modeling_options["farm"]["N_turbines"]
    #         * self.modeling_options["turbine"]["geometry"]["diameter_rotor"]
    #     )
    #     assert total_length_cables_partials == pytest.approx(expected_partial, abs=1E-12)


class TestPrimarySpacingSurrogate:

    def setup_method(self):
        # Create the problem
        prob = om.Problem()

        # set modeling options
        modeling_options = {
            "farm": {
                "N_turbines": 10,
            },
            "turbine": {
                "geometry": {
                    "diameter_rotor": 10,
                }
            },
        }

        # Add the PrimarySpacingSurrogate component with 10 turbines
        prob.model.add_subsystem(
            "spacing_calc",
            PrimarySpacingSurrogate(modeling_options=modeling_options),
            promotes=["*"],
        )

        # Set up the problem
        prob.setup()

        # Set the input value
        prob.set_val("total_length_cables", 1000.0)  # Total cable length in meters

        # Run the model
        prob.run_model()

        self.prob = prob

    def test_turbine_spacing_calculation(self):
        """Test the turbine spacing calculation."""

        # Check the output value
        turbine_spacing = self.prob.get_val("primary_turbine_spacing_diameters")
        assert turbine_spacing == pytest.approx(10.0, abs=1e-12)

    def test_partial_derivatives(self):
        """Test the partial derivatives."""

        # Check the partial derivatives
        partials = self.prob.check_partials(out_stream=None)
        total_length_cables_partials = partials["spacing_calc"][
            ("primary_turbine_spacing_diameters", "total_length_cables")
        ]["J_fwd"]
        assert total_length_cables_partials == pytest.approx(0.01, abs=1e-12)
