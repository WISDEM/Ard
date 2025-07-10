import numpy as np

from optiwindnet.mesh import make_planar_embedding
from optiwindnet.interarraylib import L_from_site
from optiwindnet.heuristics import EW_presolver
from optiwindnet.MILP import solver_factory, ModelOptions

from . import templates


def _own_L_from_inputs(inputs: dict) -> dict:
    T = len(inputs["x_turbines"])
    R = len(inputs["x_substations"])
    name_case = "farm"
    if "x_borders" in inputs:
        B = len(inputs["x_borders"])
    else:
        B = 0
    VertexC = np.empty((R + T + B, 2), dtype=float)
    VertexC[:T, 0] = inputs["x_turbines"]
    VertexC[:T, 1] = inputs["y_turbines"]
    VertexC[-R:, 0] = inputs["x_substations"]
    VertexC[-R:, 1] = inputs["y_substations"]
    site = dict(
        T=T,
        R=R,
        name=name_case,
        handle=name_case,
        VertexC=VertexC,
    )
    if B > 0:
        VertexC[T:-R, 0] = inputs["x_borders"]
        VertexC[T:-R, 1] = inputs["y_borders"]
        site["B"] = B
        site["border"] = np.arange(T, T + B)
    return L_from_site(**site)


class OptiwindnetCollection(templates.CollectionTemplate):
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

        max_turbines_per_string = self.modeling_options["collection"][
            "max_turbines_per_string"
        ]
        solver_name = self.modeling_options["collection"]["solver_name"]

        # get a graph representing the updated location
        L = _own_L_from_inputs(inputs)
        T = L.graph['T']

        # create planar embedding and set of available links
        P, A = make_planar_embedding(L)

        # presolve
        S_warm = EW_presolver(A, capacity=max_turbines_per_string)

        # do the branch-and-bound MILP search
        solver = solver_factory(solver_name)
        solver.set_problem(
            P,
            A,
            max_turbines_per_string,
            ModelOptions(**self.modeling_options["collection"]["model_options"]),
            warmstart=S_warm,
        )
        info = solver.solve(**self.modeling_options["collection"]["solver_options"])
        S, G = solver.get_solution()

        # extract the outputs
        terse_links = np.zeros((T,), dtype=np.int_)
        length_cables = np.zeros((T,))
        load_cables = np.zeros((T,))

        d2roots = A.graph['d2roots']
        # convert the graph to array representing the tree (edges i->terse[i])
        for u, v, edgeD in S.edges(data=True):
            u, v = (u, v) if u < v else (v, u)
            i, target = (u, v) if edgeD['reverse'] else (v, u)
            terse_links[i] = target
            load = edgeD['load']
            load_cables[i] = load
            if u < 0:
                # u is a substation
                if v in G[u]:
                    # feeder <u, v> has a straight route
                    length_cables[i] = d2roots[v, u]
                else:
                    # feeder <u, v> is segmented (detoured route)
                    v_neighbors = G[v]
                    for fwd_hop in v_neighbors:
                        if fwd_hop >= T and v_neighbors[fwd_hop]['load'] == load:
                            break
                    length_cables[i] = v_neighbors[fwd_hop]['length']
                    rev_hop = v
                    while fwd_hop >= T:
                        s, t = G[fwd_hop]
                        temp = s if t == rev_hop else t
                        fwd_hop, cur_hop, rev_hop = temp, fwd_hop, cur_hop
                        length_cables[i] += G[cur_hop][fwd_hop]['length']
            else:
                # link (u, v) is not a feeder, so A has length data
                length_cables[i] = A[u][v]['length']
                
        # pack and ship
        self.graph = G
        discrete_outputs["terse_links"] = terse_links
        outputs["length_cables"] = length_cables
        outputs["load_cables"] = load_cables
        outputs["max_load_cables"] = S.graph['max_load']
        # TODO: remove this assert after enough testing
        assert abs(length_cables.sum() - G.size(weight='length')) < 1e-7, f'difference: {length_cables.sum() - G.size(weight='length')}'
        outputs["total_length_cables"] = length_cables.sum()

    def compute_partials(self, inputs, J, discrete_inputs=None):

        # re-load the key variables back as locals
        G = self.graph
        T = G.graph["T"]
        R = G.graph["R"]
        VertexC = G.graph["VertexC"]
        gradients = np.zeros_like(VertexC)

        fnT = G.graph.get("fnT")
        if fnT is not None:
            _u, _v = fnT[np.array(G.edges)].T
        else:
            _u, _v = np.array(G.edges).T
        vec = VertexC[_u] - VertexC[_v]
        norm = np.hypot(*vec.T)
        # suppress the contributions of zero-length edges
        norm[np.isclose(norm, 0.0)] = 1.0
        vec /= norm[:, None]

        np.add.at(gradients, _u, vec)
        np.subtract.at(gradients, _v, vec)

        # wind turbines
        J["total_length_cables", "x_turbines"] = gradients[:T, 0]
        J["total_length_cables", "y_turbines"] = gradients[:T, 1]

        # substations
        J["total_length_cables", "x_substations"] = gradients[-R:, 0]
        J["total_length_cables", "y_substations"] = gradients[-R:, 1]

        return J
