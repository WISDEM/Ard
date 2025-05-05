import numpy as np

from optiwindnet.mesh import make_planar_embedding as own_make_planar_embedding
from optiwindnet.interarraylib import G_from_S as own_G_from_S
from optiwindnet.interarraylib import L_from_site as own_L_from_site
from optiwindnet.heuristics import EW_presolver as own_EW_presolver
from optiwindnet.pathfinding import PathFinder as OWNPathFinder
from optiwindnet.MILP import pyomo as own_pyomo

from pyomo import environ as pyo

import ard.collection.templates as templates

import logging

logging.getLogger("optiwindnet").setLevel(logging.CRITICAL)


# custom length calculation
def distance_function(x0, y0, x1, y1):
    return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5


def distance_function_deriv(x0, y0, x1, y1):
    return np.array(
        [
            ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** (-0.5) * (x1 - x0),
            ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** (-0.5) * (y1 - y0),
            -(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** (-0.5)) * (x1 - x0),
            -(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** (-0.5)) * (y1 - y0),
        ]
    )


def optiwindnet_wrapper(
    XY_turbines: np.ndarray,
    XY_substations: np.ndarray,
    XY_boundaries: np.ndarray,
    name_case: str,
    max_turbines_per_string: int,
    solver_name: str = "appsi_highs",
    solver_options: dict = None,
    verbose: bool = False,
):
    """Simple wrapper to run OptiWindNet to get a cable layout

    Args:
        XY_turbines (np.ndarray): x and y positions of turbines (easting and northing)
        XY_substations (np.ndarray): x and y positions of substations (easting and northing)
        XY_boundaries (np.ndarray): x and y locations of boundary nodes (easting and northing)
        name_case (str):  what to name the case
        max_turbines_per_string (int): maximum number of turbines per cable string
        solver_name (str, optional): which solver to use in pyomo. Defaults to "appsi_highs".
        solver_options (dict, optional): pyomo solver options. Defaults to None.
        verbose (bool, optional): whether to print information. Defaults to False.

    Returns:
        result: pyomo result
        S: OptiWindNet pyomo solution
        G: output from OptiWindNet G_from_S function
        H: output from OptiWindNet PathFinder function
    """

    # initialize solver
    solver = pyo.SolverFactory(solver_name)
    solver.available(), type(solver)

    # start the network definition
    L = own_L_from_site(
        T=len(XY_turbines),
        B=len(XY_boundaries),
        R=len(XY_substations),
        VertexC=np.vstack([XY_turbines, XY_boundaries, XY_substations]),
        border=np.arange(len(XY_turbines), len(XY_turbines) + len(XY_boundaries)),
        name=name_case,
        handle=name_case,
    )

    # create a planar embedding for presolve
    P, A = own_make_planar_embedding(L)

    # presolve
    S = own_EW_presolver(A, capacity=max_turbines_per_string)
    G = own_G_from_S(S, A)

    # create minimum length model
    model = own_pyomo.make_min_length_model(
        A,
        max_turbines_per_string,
        gateXings_constraint=False,
        branching=True,
        gates_limit=False,
    )
    own_pyomo.warmup_model(model, S)

    # create the solver and solve
    time_lim_val = 60
    if solver_options is None:
        if solver_name == "appsi_highs":
            solver_options = dict(
                time_limit=time_lim_val,
                mip_rel_gap=0.05,  # TODO ???
            )
        elif solver_name == "scip":
            solver_options = {
                "limits/gap": 0.005,
                "limits/time": time_lim_val,
                "display/freq": 0.5,
                # this is currently useless, as pyomo is not calling the concurrent solver
                # 'parallel/maxnthreads': 16,
            }
        else:
            raise (
                ValueError(
                    f"No default solver options available for pyomo solver {solver_name}"
                )
            )

    solver.options.update(solver_options)
    result = solver.solve(model, tee=verbose)

    # do some postprocessing
    S = own_pyomo.S_from_solution(model, solver, result)
    G = own_G_from_S(S, A)
    H = OWNPathFinder(G, planar=P, A=A).create_detours()

    return result, S, G, H


class optiwindnetCollection(templates.CollectionTemplate):
    """
    Component class for modeling optiwindnet-optimized energy collection systems.

    A component class to make a heuristic-based optimized energy collection and
    management system using optiwindnet! Inherits the interface from
    `templates.CollectionTemplate`.

    Options
    -------
    modeling_options : dict

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
    length_cables : np.ndarray
        a 1D numpy array that holds the lengths of all of the cables necessary to
        collect energy generated
    load_cables : np.ndarray
        a 1D numpy array that holds the load integer (i.e. total number of
        turbines) collected up to this point of the cable
    """

    def initialize(self):
        """Initialization of OM component."""
        super().initialize()

    def setup(self):
        """Setup of OM component."""
        super().setup()

    def setup_partials(self):
        """Setup of OM component gradients."""

        self.declare_partials(
            ["total_length_cables"],
            ["x_turbines", "y_turbines", "x_substations", "y_substations"],
            method="exact",
        )

    def compute(
        self,
        inputs,
        outputs,
        discrete_inputs=None,
        discrete_outputs=None,
    ):
        """
        Computation for the OptiWindNet collection system design
        """

        name_case = "farm"
        max_turbines_per_string = self.modeling_options["collection"][
            "max_turbines_per_string"
        ]
        solver_name = self.modeling_options["collection"]["solver_name"]
        solver_options = self.modeling_options["collection"]["solver_options"]

        # roll up the coordinates into a form that optiwindnet #TODO consider adjusting the buffer (0.25)
        XY_turbines = np.vstack([inputs["x_turbines"], inputs["y_turbines"]]).T
        x_min = np.min(XY_turbines[:, 0]) - 0.25 * np.ptp(XY_turbines[:, 0])
        x_max = np.max(XY_turbines[:, 0]) + 0.25 * np.ptp(XY_turbines[:, 0])
        y_min = np.min(XY_turbines[:, 1]) - 0.25 * np.ptp(XY_turbines[:, 1])
        y_max = np.max(XY_turbines[:, 1]) + 0.25 * np.ptp(XY_turbines[:, 1])
        XY_boundaries = np.array(
            [
                [x_max, y_max],
                [x_min, y_max],
                [x_min, y_min],
                [x_max, y_min],
            ]
        )
        XY_substations = np.vstack([inputs["x_substations"], inputs["y_substations"]]).T

        result, S, G, H = optiwindnet_wrapper(
            XY_turbines,
            XY_substations,
            XY_boundaries,
            name_case,
            max_turbines_per_string,
            solver_name,
            solver_options,
        )

        # extract the outputs
        lengths = []
        loads = []
        edges = H.edges()
        self.graph = H

        for edge in edges:
            lengths.append(edges[edge]["length"])
            loads.append(edges[edge]["load"])

        # pack and ship
        discrete_outputs["length_cables"] = np.array(lengths, dtype=np.float64)
        discrete_outputs["load_cables"] = np.array(loads, dtype=np.float64)
        outputs["total_length_cables"] = np.sum(discrete_outputs["length_cables"])
        discrete_outputs["max_load_cables"] = np.max(discrete_outputs["load_cables"])

    def compute_partials(self, inputs, J, discrete_inputs=None):

        # re-load the key variables back as locals
        XY_turbines = np.vstack([inputs["x_turbines"], inputs["y_turbines"]]).T
        XY_substations = np.vstack([inputs["x_substations"], inputs["y_substations"]]).T
        # print(self.graph)
        H = self.graph
        edges = H.edges()

        # J["length_cables", "x_turbines"] = 0.0
        # J["length_cables", "y_turbines"] = 0.0
        # J["length_cables", "x_substations"] = 0.0
        # J["length_cables", "y_substations"] = 0.0
        J["total_length_cables", "x_turbines"] = 0.0
        J["total_length_cables", "y_turbines"] = 0.0
        J["total_length_cables", "x_substations"] = 0.0
        J["total_length_cables", "y_substations"] = 0.0

        for idx_edge, edge in enumerate(edges):
            e0, e1 = edge
            x0, y0 = (
                XY_substations[self.N_substations + e0, :]
                if e0 < 0
                else XY_turbines[e0, :]
            )
            x1, y1 = (
                XY_substations[self.N_substations + e1, :]
                if e1 < 0
                else XY_turbines[e1, :]
            )

            # get the derivative function
            dLdx0, dLdy0, dLdx1, dLdy1 = distance_function_deriv(x0, y0, x1, y1)

            if e0 >= 0:
                # J["length_cables", "x_turbines"][idx_edge, e0] -= dLdx0
                # J["length_cables", "y_turbines"][idx_edge, e0] -= dLdy0
                J["total_length_cables", "x_turbines"][0, e0] -= dLdx0
                J["total_length_cables", "y_turbines"][0, e0] -= dLdy0
            else:
                # J["length_cables", "x_substations"][
                #     idx_edge, self.N_substations + e0
                # ] -= dLdx0
                # J["length_cables", "y_substations"][
                #     idx_edge, self.N_substations + e0
                # ] -= dLdy0
                J["total_length_cables", "x_substations"][
                    0, self.N_substations + e0
                ] -= dLdx0
                J["total_length_cables", "y_substations"][
                    0, self.N_substations + e0
                ] -= dLdy0
            if e1 >= 0:
                # J["length_cables", "x_turbines"][idx_edge, e1] -= dLdx1
                # J["length_cables", "y_turbines"][idx_edge, e1] -= dLdy1
                J["total_length_cables", "x_turbines"][0, e1] -= dLdx1
                J["total_length_cables", "y_turbines"][0, e1] -= dLdy1
            else:
                # J["length_cables", "x_substations"][
                #     idx_edge, self.N_substations + e1
                # ] -= dLdx1
                # J["length_cables", "y_substations"][
                #     idx_edge, self.N_substations + e1
                # ] -= dLdy1
                J["total_length_cables", "x_substations"][
                    0, self.N_substations + e1
                ] -= dLdx1
                J["total_length_cables", "y_substations"][
                    0, self.N_substations + e1
                ] -= dLdy1
