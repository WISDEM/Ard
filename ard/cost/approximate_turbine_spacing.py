import openmdao.api as om
from ard.cost.wisdem_wrap import LandBOSSE


class LandBOSSEWithSpacingApproximations(om.Group):
    """
    OpenMDAO group that connects the SpacingApproximations component to the LandBOSSE component.

    This group calculates the turbine spacing using the SpacingApproximations and passes it
    to the LandBOSSE component for further cost estimation.
    """

    def initialize(self):
        """Initialize the group and declare options."""
        self.options.declare(
            "modeling_options", types=dict, desc="Ard modeling options"
        )

    def setup(self):
        """Set up the group by adding and connecting components."""
        # Add the PrimarySpacingApproximations component
        self.add_subsystem(
            "spacing_approximations",
            SpacingApproximations(modeling_options=self.options["modeling_options"]),
            promotes_inputs=["total_length_cables"],
        )

        # Add the LandBOSSE component
        self.add_subsystem(
            "landbosse",
            LandBOSSE(),
            promotes_inputs=[
                "*",
                (
                    "turbine_spacing_rotor_diameters",
                    "internal_turbine_spacing_rotor_diameters",
                ),
                (
                    "row_spacing_rotor_diameters",
                    "internal_row_spacing_rotor_diameters",
                ),
            ],
            promotes_outputs=["*"],  # Expose all outputs from LandBOSSE
        )

        # Connect the turbine and row spacing outputs from the approximations to LandBOSSE
        self.connect(
            "spacing_approximations.primary_turbine_spacing_diameters",
            "internal_turbine_spacing_rotor_diameters",
        )

        self.connect(
            "spacing_approximations.secondary_turbine_spacing_diameters",
            "internal_row_spacing_rotor_diameters",
        )


class SpacingApproximations(om.ExplicitComponent):
    """
    OpenMDAO component to calculate approximations for turbine spacing based on the total length of cables
    and the number of wind turbines.

    Inputs
    ------
    total_length_cables : float
        Total length of cables in meters.

    Outputs
    -------
    primary_turbine_spacing_diameters : float
        Approximation of spacing between turbines in diameters for use in cost estimation using LandBOSSE.
    secondary_spacing_diameters : float
        Approximation of spacing between rows of turbines in diameters for use in cost estimation using LandBOSSE.

    Options
    -------
    modeling_options : dict
        Dictionary of modeling options including at least ["farm"]["N_turbines"] and ["turbine"]["geometry"]["diameter_rotor"]
    """

    def initialize(self):
        """Initialize the component and declare options."""
        self.options.declare(
            "modeling_options", types=dict, desc="Ard modeling options"
        )

    def setup(self):
        """Set up the inputs and outputs."""
        self.add_input(
            "total_length_cables", val=0.0, units="m", desc="Total cable length"
        )
        self.add_output(
            "primary_turbine_spacing_diameters",
            val=0.0,
            units=None,
            desc="Turbine spacing",
        )
        self.add_output(
            "secondary_turbine_spacing_diameters",
            val=0.0,
            units=None,
            desc="Row spacing",
        )

    def setup_partials(self):
        """Declare partial derivatives."""
        N_turbines = self.options["modeling_options"]["farm"]["N_turbines"]
        rotor_diameter_m = self.options["modeling_options"]["turbine"]["geometry"][
            "diameter_rotor"
        ]

        # Partial derivative of primary_turbine_spacing_diameters w.r.t. total_length_cables are constant
        const_partial = 1.0 / (rotor_diameter_m * N_turbines)
        self.declare_partials(
            "primary_turbine_spacing_diameters",
            "total_length_cables",
            val=const_partial,
        )
        self.declare_partials(
            "secondary_turbine_spacing_diameters",
            "total_length_cables",
            val=const_partial,
        )

    def compute(self, inputs, outputs):
        """Compute the turbine spacing."""
        total_length_cables = inputs["total_length_cables"]
        N_turbines = self.options["modeling_options"]["farm"]["N_turbines"]
        rotor_diameter_m = self.options["modeling_options"]["turbine"]["geometry"][
            "diameter_rotor"
        ]

        # Calculate turbine and row spacing
        outputs["primary_turbine_spacing_diameters"] = total_length_cables / (
            rotor_diameter_m * N_turbines
        )
        outputs["secondary_turbine_spacing_diameters"] = total_length_cables / (
            rotor_diameter_m * N_turbines
        )

    def compute_partials(self, inputs, partials):
        "partials are constant, so no calculations needed here"
        pass
