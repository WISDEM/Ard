import openmdao.api as om
from ard.cost.surrogate_turbine_spacing import PrimarySpacingSurrogate
import pytest

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
        assert turbine_spacing == pytest.approx(10.0, abs=1E-12)

    def test_partial_derivatives(self):
        """Test the partial derivatives."""

        # Check the partial derivatives
        partials = self.prob.check_partials(out_stream=None)
        total_length_cables_partials = partials["spacing_calc"][("primary_turbine_spacing_diameters", "total_length_cables")]["J_fwd"]
        assert total_length_cables_partials == pytest.approx(0.01, abs=1E-12)