from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import wisdem.orbit.orbit_api as orbit_wisdem
from ORBIT import ProjectManager
from ORBIT.core.library import default_library
from ORBIT.core.library import initialize_library

from ard.cost.wisdem_wrap import ORBIT_setup_latents


def generate_orbit_location_from_graph(
    graph,  # TODO: replace with a terse_links representation
    X_turbines,
    Y_turbines,
    X_substations,
    Y_substations,
):
    """
    go from a optiwindnet graph to an ORBIT input CSV

    convert a optiwindnet graph representation of a collection system and get to
    a best-possible approximation of the same collection system for
    compatibility with ORBIT. ORBIT doesn't allow branching and optiwindnet
    does by default, so we get allow some cable duplication if necessary to get
    a conservative approximation of the BOS costs if the graph isn't compatible
    with ORBIT

    Parameters
    ----------
    graph : networkx.Graph
        the graph representation of the collection system design
    X_turbines : np.array
        the cartesian X locations, in kilometers, of the turbines
    Y_turbines : np.array
        the cartesian Y locations, in kilometers, of the turbines
    X_substations : np.array
        the cartesian X locations, in kilometers, of the substations
    Y_substations : np.array
        the cartesian Y locations, in kilometers, of the substations

    Returns
    -------
    pandas.DataFrame
        a dataframe formatted for ORBIT to specify a farm layout

    Raises
    ------
    RecursionError
        if the recursive setup seems to be stuck in a loop
    """

    # get all edges, sorted by the first node then the second node
    edges_to_process = [edge for edge in graph.edges]
    edges_to_process.sort(key=lambda x: (x[0], x[1]))
    # get the edges with a negative index node (a substation)
    edges_inclsub = [edge for edge in edges_to_process if edge[0] < 0 or edge[1] < 0]
    edges_inclsub.sort(key=lambda x: (x[0], x[1]))

    # data for ORBIT
    data_orbit = {
        "id": [],
        "substation_id": [],
        "name": [],
        "longitude": [],
        "latitude": [],
        "string": [],
        "order": [],
        "cable_length": [],
        "bury_speed": [],
    }

    idx_string = 0
    order = 0

    for edge in edges_inclsub:  # every edge w/ a substation starts a string

        def handle_edge(
            edge, turbine_origination, idx_string, order, recursion_level=0
        ):
            # recursively handle the edges

            if recursion_level > 10:  # for safe recursion
                raise RecursionError("Recursion limit reached")

            # get the target turbine index
            turbine_tgt_index = edge[0] if edge[0] != turbine_origination else edge[1]
            # get the turbine name
            turbine_name = turbine_id = f"t{turbine_tgt_index:03d}"

            # add the turbine to the dataset
            data_orbit["id"].append(turbine_id)
            data_orbit["substation_id"].append(substation_id)
            data_orbit["name"].append(turbine_name)
            data_orbit["longitude"].append(X_turbines[turbine_tgt_index])
            data_orbit["latitude"].append(Y_turbines[turbine_tgt_index])
            data_orbit["string"].append(int(idx_string))
            data_orbit["order"].append(int(order))
            data_orbit["cable_length"].append(0)  # ORBIT computes automatically
            data_orbit["bury_speed"].append(0)  # ORBIT computes automatically

            # pop this edge out of the edges list
            edges_to_process.remove(edge)

            # get the set of remaining edges that include the terminal turbine
            edges_turbine = [e for e in edges_to_process if (turbine_tgt_index in e)]

            order += 1

            for new_string, edge_next in enumerate(edges_turbine):
                if new_string:
                    idx_string += 1
                    order = 0
                idx_string, order = handle_edge(
                    edge_next,
                    turbine_tgt_index,
                    idx_string,
                    order,
                    recursion_level=recursion_level + 1,
                )

            return idx_string, order

        # get the substation id as a one-liner
        substation_index = len(X_substations) + (edge[0] if edge[0] < 0 else edge[1])
        # get the substation name
        substation_name = substation_id = f"oss{substation_index:01d}"

        # add the substation to the dataset
        if not substation_id in data_orbit["id"]:
            data_orbit["id"].append(substation_id)
            data_orbit["substation_id"].append(substation_id)
            data_orbit["name"].append(substation_name)
            data_orbit["longitude"].append(X_substations[substation_index] / 1.0e3)
            data_orbit["latitude"].append(Y_substations[substation_index] / 1.0e3)
            data_orbit["string"].append(None)
            data_orbit["order"].append(None)
            data_orbit["cable_length"].append(None)
            data_orbit["bury_speed"].append(None)

        # handle the edge that we get
        idx_string, order = handle_edge(
            edge, substation_index - len(X_substations), idx_string, order
        )

        order = 0
        idx_string += 1

    df_orbit = pd.DataFrame(data_orbit).fillna("")
    df_orbit.string = [int(v) if v != "" else "" for v in df_orbit.string]
    df_orbit.order = [int(v) if v != "" else "" for v in df_orbit.order]

    return df_orbit


class ORBITDetail(orbit_wisdem.Orbit):
    """
    Wrapper for WISDEM's ORBIT offshore BOS calculators.

    A thicker wrapper of `wisdem.orbit_api` that 1) replaces capabilities that
    assume a grid farm layout that is default in WISDEM's ORBIT with a custom
    array layout, and 2) traps warning messages that are recognized not to be
    issues.

    See: https://github.com/WISDEM/ORBIT
    """

    def initialize(self):
        """Initialize for API connections."""
        super().initialize()

        self.options.declare("case_title", default="working")
        self.options.declare("modeling_options")

    def setup(self):
        """Define all input variables from all models."""

        # load modeling options
        self.modeling_options = self.options["modeling_options"]
        self.N_turbines = self.modeling_options["farm"]["N_turbines"]
        self.N_substations = self.modeling_options["farm"]["N_substations"]

        self.set_input_defaults("wtiv", "example_wtiv")
        self.set_input_defaults("feeder", "example_feeder")
        # self.set_input_defaults("num_feeders", 1)
        # self.set_input_defaults("num_towing", 1)
        # self.set_input_defaults("num_station_keeping", 3)
        # self.set_input_defaults(
        #    "oss_install_vessel", "example_heavy_lift_vessel",
        # )
        self.set_input_defaults("site_distance", 40.0, units="km")
        self.set_input_defaults("site_distance_to_landfall", 40.0, units="km")
        self.set_input_defaults("interconnection_distance", 40.0, units="km")
        self.set_input_defaults("plant_turbine_spacing", 7)
        self.set_input_defaults("plant_row_spacing", 7)
        self.set_input_defaults("plant_substation_distance", 1, units="km")
        # self.set_input_defaults("num_port_cranes", 1)
        # self.set_input_defaults("num_assembly_lines", 1)
        self.set_input_defaults("takt_time", 170.0, units="h")
        self.set_input_defaults("port_cost_per_month", 2e6, units="USD/mo")
        self.set_input_defaults("construction_insurance", 44.0, units="USD/kW")
        self.set_input_defaults("construction_financing", 183.0, units="USD/kW")
        self.set_input_defaults("contingency", 316.0, units="USD/kW")
        self.set_input_defaults("commissioning_cost_kW", 44.0, units="USD/kW")
        self.set_input_defaults("decommissioning_cost_kW", 58.0, units="USD/kW")
        self.set_input_defaults("site_auction_price", 100e6, units="USD")
        self.set_input_defaults("site_assessment_cost", 50e6, units="USD")
        self.set_input_defaults("construction_plan_cost", 1e6, units="USD")
        self.set_input_defaults("installation_plan_cost", 2.5e5, units="USD")
        self.set_input_defaults("boem_review_cost", 0.0, units="USD")

        self.add_subsystem(
            "orbit",
            ORBITWisdemDetail(
                modeling_options=self.modeling_options,
                case_title=self.options["case_title"],
                floating=self.options["floating"],
                jacket=self.options["jacket"],
                jacket_legs=self.options["jacket_legs"],
            ),
            promotes=["*"],
        )


class ORBITWisdemDetail(orbit_wisdem.OrbitWisdem):
    """ORBIT-WISDEM Fixed Substructure API, modified for detailed layouts"""

    _path_library = None

    def initialize(self):
        super().initialize()

        self.options.declare("case_title", default="working")
        self.options.declare("modeling_options")

    def setup(self):
        """Define all the inputs."""

        # load modeling options
        self.modeling_options = self.options["modeling_options"]
        self.N_turbines = self.modeling_options["farm"]["N_turbines"]
        self.N_substations = self.modeling_options["farm"]["N_substations"]

        # Inputs
        # self.add_discrete_input(
        #     'weather_file',
        #     'block_island',
        #     desc='Weather file to use for installation times.'
        # )

        # Vessels
        self.add_discrete_input(
            "wtiv",
            "example_wtiv",
            desc=(
                "Vessel configuration to use for installation of foundations"
                " and turbines."
            ),
        )
        self.add_discrete_input(
            "feeder",
            "future_feeder",
            desc="Vessel configuration to use for (optional) feeder barges.",
        )
        self.add_discrete_input(
            "num_feeders",
            1,
            desc=(
                "Number of feeder barges to use for installation of"
                " foundations and turbines."
            ),
        )
        self.add_discrete_input(
            "num_towing",
            1,
            desc=(
                "Number of towing vessels to use for floating platforms that"
                " are assembled at port (with or without the turbine)."
            ),
        )
        self.add_discrete_input(
            "num_station_keeping",
            3,
            desc=(
                "Number of station keeping or AHTS vessels that attach to floating"
                " platforms under tow-out."
            ),
        )
        self.add_discrete_input(
            "oss_install_vessel",
            "example_heavy_lift_vessel",
            desc="Vessel configuration to use for installation of offshore substations.",  # noqa: E501
        )

        # Site
        self.add_input("site_depth", 40.0, units="m", desc="Site depth.")
        self.add_input(
            "site_distance",
            40.0,
            units="km",
            desc="Distance from site to installation port.",
        )
        self.add_input(
            "site_distance_to_landfall",
            50.0,
            units="km",
            desc="Distance from site to landfall for export cable.",
        )
        self.add_input(
            "interconnection_distance",
            3.0,
            units="km",
            desc="Distance from landfall to interconnection.",
        )
        self.add_input(
            "site_mean_windspeed",
            9.0,
            units="m/s",
            desc="Mean windspeed of the site.",
        )

        # Plant
        self.add_discrete_input(
            "number_of_turbines",
            60,
            desc="Number of turbines.",
        )
        self.add_input(
            "plant_turbine_spacing",
            7,
            desc="Turbine spacing in rotor diameters.",
        )
        self.add_input(
            "plant_row_spacing",
            7,
            desc="Row spacing in rotor diameters. Not used in ring layouts.",
        )
        self.add_input(
            "plant_substation_distance",
            1,
            units="km",
            desc="Distance from first turbine in string to substation.",
        )
        self.add_input("x_turbines", np.zeros((self.N_turbines,)), units="km")
        self.add_input("y_turbines", np.zeros((self.N_turbines,)), units="km")
        self.add_input("x_substations", np.zeros((self.N_substations,)), units="km")
        self.add_input("y_substations", np.zeros((self.N_substations,)), units="km")

        # Turbine
        self.add_input(
            "turbine_rating",
            8.0,
            units="MW",
            desc="Rated capacity of a turbine.",
        )
        self.add_input(
            "turbine_rated_windspeed",
            11.0,
            units="m/s",
            desc="Rated windspeed of the turbine.",
        )
        self.add_input(
            "turbine_capex",
            1100,
            units="USD/kW",
            desc="Turbine CAPEX",
        )
        self.add_input(
            "hub_height",
            100.0,
            units="m",
            desc="Turbine hub height.",
        )
        self.add_input(
            "turbine_rotor_diameter",
            130,
            units="m",
            desc="Turbine rotor diameter.",
        )
        self.add_input(
            "tower_mass",
            400.0,
            units="t",
            desc="mass of the total tower.",
        )
        self.add_input(
            "tower_length",
            100.0,
            units="m",
            desc="Total length of the tower.",
        )
        self.add_input(
            "tower_deck_space",
            25.0,
            units="m**2",
            desc=(
                "Deck space required to transport the tower. Defaults to 0 in"
                " order to not be a constraint on installation."
            ),
        )
        self.add_input(
            "nacelle_mass",
            500.0,
            units="t",
            desc="mass of the rotor nacelle assembly (RNA).",
        )
        self.add_input(
            "nacelle_deck_space",
            25.0,
            units="m**2",
            desc=(
                "Deck space required to transport the rotor nacelle assembly"
                " (RNA). Defaults to 0 in order to not be a constraint on"
                " installation."
            ),
        )
        self.add_discrete_input(
            "number_of_blades",
            3,
            desc="Number of blades per turbine.",
        )
        self.add_input(
            "blade_mass",
            50.0,
            units="t",
            desc="mass of an individual blade.",
        )
        self.add_input(
            "blade_deck_space",
            100.0,
            units="m**2",
            desc=(
                "Deck space required to transport a blade. Defaults to 0 in"
                " order to not be a constraint on installation."
            ),
        )

        # Mooring
        self.add_discrete_input(
            "num_mooring_lines",
            3,
            desc="Number of mooring lines per platform.",
        )
        self.add_input(
            "mooring_line_mass",
            1e4,
            units="kg",
            desc="Total mass of a mooring line",
        )
        self.add_input(
            "mooring_line_diameter",
            0.1,
            units="m",
            desc="Cross-sectional diameter of a mooring line",
        )
        self.add_input(
            "mooring_line_length",
            1e3,
            units="m",
            desc="Unstretched mooring line length",
        )
        self.add_input(
            "anchor_mass",
            1e4,
            units="kg",
            desc="Total mass of an anchor",
        )
        self.add_input(
            "mooring_line_cost",
            0.5e6,
            units="USD",
            desc="Mooring line unit cost.",
        )
        self.add_input(
            "mooring_anchor_cost",
            0.1e6,
            units="USD",
            desc="Mooring line unit cost.",
        )
        self.add_discrete_input(
            "anchor_type",
            "drag_embedment",
            desc="Number of mooring lines per platform.",
        )

        # Port
        self.add_input(
            "port_cost_per_month",
            2e6,
            units="USD/mo",
            desc="Monthly port costs.",
        )
        self.add_input(
            "takt_time",
            170.0,
            units="h",
            desc="Substructure assembly cycle time when doing assembly at the port.",  # noqa: E501
        )
        self.add_discrete_input(
            "num_assembly_lines",
            1,
            desc="Number of assembly lines used when assembly occurs at the port.",  # noqa: E501
        )
        self.add_discrete_input(
            "num_port_cranes",
            1,
            desc=(
                "Number of cranes used at the port to load feeders / WTIVS"
                " when assembly occurs on-site or assembly cranes when"
                " assembling at port."
            ),
        )

        # Floating Substructures
        self.add_input(
            "floating_substructure_cost",
            10e6,
            units="USD",
            desc="Floating substructure unit cost.",
        )

        # Monopile
        self.add_input(
            "monopile_length",
            100.0,
            units="m",
            desc="Length of monopile (including pile).",
        )
        self.add_input(
            "monopile_diameter",
            7.0,
            units="m",
            desc="Diameter of monopile.",
        )
        self.add_input(
            "monopile_mass",
            900.0,
            units="t",
            desc="mass of an individual monopile.",
        )
        self.add_input(
            "monopile_cost",
            4e6,
            units="USD",
            desc="Monopile unit cost.",
        )

        # Jacket
        self.add_input(
            "jacket_length",
            65.0,
            units="m",
            desc="Length/height of jacket (including pile/buckets).",
        )
        self.add_input(
            "jacket_mass",
            900.0,
            units="t",
            desc="mass of an individual jacket.",
        )
        self.add_input(
            "jacket_cost",
            4e6,
            units="USD",
            desc="Jacket unit cost.",
        )
        self.add_input(
            "jacket_r_foot",
            10.0,
            units="m",
            desc="Radius of jacket legs at base from centeroid.",
        )

        # Generic fixed-bottom
        self.add_input(
            "transition_piece_mass",
            250.0,
            units="t",
            desc="mass of an individual transition piece.",
        )
        self.add_input(
            "transition_piece_deck_space",
            25.0,
            units="m**2",
            desc=(
                "Deck space required to transport a transition piece."
                " Defaults to 0 in order to not be a constraint on"
                " installation."
            ),
        )
        self.add_input(
            "transition_piece_cost",
            1.5e6,
            units="USD",
            desc="Transition piece unit cost.",
        )

        # Project
        self.add_input(
            "construction_insurance",
            44.0,
            units="USD/kW",
            desc="Cost for construction insurance",
        )
        self.add_input(
            "construction_financing",
            183.0,
            units="USD/kW",
            desc="Cost for construction financing",
        )
        self.add_input(
            "contingency",
            316.0,
            units="USD/kW",
            desc="Cost in case of contingency",
        )
        self.add_input(
            "site_auction_price",
            100e6,
            units="USD",
            desc="Cost to secure site lease",
        )
        self.add_input(
            "site_assessment_cost",
            50e6,
            units="USD",
            desc="Cost to execute site assessment",
        )
        self.add_input(
            "construction_plan_cost",
            1e6,
            units="USD",
            desc="Cost to do construction planning",
        )
        self.add_input(
            "installation_plan_cost",
            2.5e5,
            units="USD",
            desc="Cost to do construction planning",
        )
        self.add_input(
            "boem_review_cost",
            0.0,
            units="USD",
            desc=(
                "Cost for additional review by U.S. Dept of Interior Bureau"
                " of Ocean Energy Management (BOEM)"
            ),
        )
        self.add_input(
            "commissioning_cost_kW", 44.0, units="USD/kW", desc="Commissioning cost."
        )
        self.add_input(
            "decommissioning_cost_kW",
            58.0,
            units="USD/kW",
            desc="Decommissioning cost.",
        )

        # Collection System
        self.add_discrete_input("graph", None)

        # Outputs
        # Totals
        self.add_output(
            "bos_capex",
            0.0,
            units="USD",
            desc="Sum of system and installation capex",
        )
        self.add_output(
            "soft_capex",
            0.0,
            units="USD",
            desc="Project costs associated with commissioning, decommissioning and financing",
        )
        self.add_output(
            "project_capex",
            0.0,
            units="USD",
            desc="costs associated with the lease area, "
            + "the development of the construction operations plan,"
            + "and any environmental review and other upfront project costs.",
        )
        self.add_output(
            "total_capex",
            0.0,
            units="USD",
            desc="Total capex of bos + soft + project",
        )
        self.add_output(
            "total_capex_kW",
            0.0,
            units="USD/kW",
            desc="Total capex of bos + soft + project per rated project capacity in kW",
        )
        self.add_output(
            "installation_time",
            0.0,
            units="h",
            desc="Total balance of system installation time.",
        )
        self.add_output(
            "installation_capex",
            0.0,
            units="USD",
            desc="Total balance of system installation cost.",
        )

    def compile_orbit_config_file(
        self,
        inputs,
        outputs,
        discrete_inputs,
        discrete_outputs,
    ):

        config = super().compile_orbit_config_file(
            inputs,
            outputs,
            discrete_inputs,
            discrete_outputs,
        )  # run the superclass

        # copy the default library to a local directory under case_files
        path_library_default = Path(default_library)
        self._path_library = (
            Path("case_files") / self.options["case_title"] / "ORBIT_library"
        )
        if path_library_default.exists():
            shutil.copytree(
                path_library_default, self._path_library, dirs_exist_ok=True
            )
        else:
            raise FileNotFoundError(
                f"Can not find default ORBIT library at {path_library_default}."
            )

        # remove the grid plant option, and replace with a custom plant
        config["plant"] = {
            "layout": "custom",
            "num_turbines": int(discrete_inputs["number_of_turbines"]),
        }

        # switch to the custom array system design
        if not ("ArraySystemDesign" in config["design_phases"]):
            raise KeyError(
                "I assumed that 'ArraySystemDesign' would be in the config. Something changed."
            )
        config["design_phases"][
            config["design_phases"].index("ArraySystemDesign")
        ] = "CustomArraySystemDesign"

        # add a turbine location csv on the config
        basename_farm_location = "wisdem_detailed_array"
        config["array_system_design"]["distance"] = True  # don't use WGS84 lat/long
        config["array_system_design"]["location_data"] = basename_farm_location
        config["array_system_design"]["cables"] = [
            "XLPE_185mm_66kV_dynamic",
            "XLPE_500mm_132kV_dynamic",
            "XLPE_630mm_66kV_dynamic",
            "XLPE_1000mm_220kV_dynamic",
            # "HVDC_2000mm_320kV",
            # "HVDC_2000mm_320kV_dynamic",
            # "HVDC_2000mm_400kV",
            # "HVDC_2500mm_525kV",
            # "XLPE_1000mm_220kV",
            # "XLPE_1000mm_220kV_dynamic",
            # "XLPE_1200mm_220kV",
            # "XLPE_1200mm_275kV",
            # "XLPE_1600mm_275kV",
            # "XLPE_185mm_66kV",
            # "XLPE_185mm_66kV_dynamic",
            # "XLPE_1900mm_275kV",
            # "XLPE_400mm_33kV",
            # "XLPE_500mm_132kV",
            # "XLPE_500mm_132kV_dynamic",
            # "XLPE_500mm_220kV",
            # "XLPE_630mm_220kV",
            # "XLPE_630mm_33kV",
            # "XLPE_630mm_66kV",
            # "XLPE_630mm_66kV_dynamic",
            # "XLPE_800mm_220kV",
        ]  # we require bigger cables, I think

        # create the csv file that holds the farm layout
        path_farm_location = (
            self._path_library / "cables" / (basename_farm_location + ".csv")
        )

        # generate the csv data needed to locate the farm elements
        generate_orbit_location_from_graph(
            discrete_inputs["graph"],
            inputs["x_turbines"],
            inputs["y_turbines"],
            inputs["x_substations"],
            inputs["y_substations"],
        ).to_csv(path_farm_location, index=False)

        self._orbit_config = config  # reinstall- probably not needed due to reference
        return config  # and return

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        config = self.compile_orbit_config_file(
            inputs, outputs, discrete_inputs, discrete_outputs
        )

        # setup the custom-location library
        if self._path_library:
            initialize_library(self._path_library)

        project = ProjectManager(config)
        project.run()

        # The ORBIT version of total_capex includes turbine capex, so we do our own sum of
        # the parts here that wisdem doesn't account for
        capacity_kW = (
            1e3 * inputs["turbine_rating"] * discrete_inputs["number_of_turbines"]
        )
        outputs["bos_capex"] = project.bos_capex
        outputs["soft_capex"] = project.soft_capex
        outputs["project_capex"] = project.project_capex
        outputs["total_capex"] = (
            project.bos_capex + project.soft_capex + project.project_capex
        )
        outputs["total_capex_kW"] = outputs["total_capex"] / capacity_kW
        outputs["installation_time"] = project.installation_time
        outputs["installation_capex"] = project.installation_capex
