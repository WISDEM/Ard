import numpy as np

import openmdao.api as om


class CollectionTemplate(om.ExplicitComponent):
    """
    Template component for a energy collection system.

    A energy collection system component, based on this template, will compute the
    energy collection system necessary for a farm given its layout, turbine
    definitions, and substation location.

    Options
    -------
    modeling_options : dict
        a modeling options dictionary

    Inputs
    ------
    x_turbines : np.ndarray
        a 1D numpy array indicating the x-dimension locations of the turbines,
        with length `N_turbines`
    y_turbines : np.ndarray
        a 1D numpy array indicating the y-dimension locations of the turbines,
        with length `N_turbines`
    x_substations : np.ndarray
        a 1D numpy array indicating the x-dimension locations of the substations,
        with length `N_substations`
    y_substations : np.ndarray
        a 1D numpy array indicating the y-dimension locations of the substations,
        with length `N_substations`

    Outputs
    -------
    total_length_cables : float
        the total length of cables to collect energy generated

    Discrete Outputs
    -------
    length_cables : np.ndarray
        a (variable-length) 1D numpy array that holds the lengths of each of the cables necessary
        to collect energy generated, with length `N_turbines`
    load_cables : np.ndarray
        a (variable-length) 1D numpy array that holds the load integer (i.e. total number of
        turbines) collected up to each cable, with length `N_turbines`
    max_load_cables : np.ndarray
        the maximum cable capacity required by the system
    """

    def initialize(self):
        """Initialization of OM component."""
        self.options.declare("modeling_options")

    def setup(self):
        """Setup of OM component."""
        # load modeling options
        self.modeling_options = self.options["modeling_options"]
        self.N_turbines = self.modeling_options["farm"]["N_turbines"]
        self.N_substations = self.modeling_options["farm"]["N_substations"]

        # set up inputs for farm layout
        self.add_input("x_turbines", np.zeros((self.N_turbines,)), units="m")
        self.add_input("y_turbines", np.zeros((self.N_turbines,)), units="m")
        self.add_input("x_substations", np.zeros((self.N_substations,)), units="m")
        self.add_input("y_substations", np.zeros((self.N_substations,)), units="m")

        # set up outputs for the collection system
        self.add_discrete_output(
            "length_cables",
            np.zeros((self.N_turbines,)),  # units="m",
        )
        self.add_discrete_output("load_cables", np.zeros((self.N_turbines,)))
        self.add_output("total_length_cables", 0.0, units="m")
        self.add_discrete_output("max_load_cables", 0.0)

    def compute(
        self,
        inputs,
        outputs,
        discrete_inputs=None,
        discrete_outputs=None,
    ):
        """
        Computation for the OM component.

        For a template class this is not implemented and raises an error!
        """

        ###########################################
        #                                         #
        # IMPLEMENT THE COLLECTION COMPONENT HERE #
        #                                         #
        ###########################################

        raise NotImplementedError(
            "This is an abstract class for a derived class to implement!"
        )
