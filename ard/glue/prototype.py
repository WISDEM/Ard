import openmdao.api as om

import floris.wind_data

import ard.layout.sunflower as sunflower
import ard.layout.gridfarm as gridfarm
import ard.farm_aero.floris as farmaero_floris
import ard.cost.wisdem_wrap as cost_wisdem


def create_setup_OM_problem(
    modeling_options,
    wind_rose: floris.wind_data.WindRose = None,
    aero_backend: str = "FLORIS",
    layout_type: str = "gridfarm",
    setup_glue=True,
):
    """
    A prototype to create and setup an Ard OpenMDAO problem.

    This is a prototype of the "glue code" functionality of Ard, which creates
    and manages the setup of an OM problem to define a wind farm.

    Parameters
    ----------
    modeling_options : dict
        a modeling options dictionary
    wind_rose : floris.wind_data.WindRose, optional
        a `floris.wind_data.WindRose` object for AEP calculation
    aero_backend : str, optional
        aerodynamic backend for calculation by default (and at present
        exclusively) "FLORIS"
    layout_type : str, optional
        layout parametrization model by default (and at present exclusively)
        "gridfarm"
    setup_glue : bool, optional (default True)
        switch to setup at the end, or return without setup (if other components
        should be added)

    Returns
    -------
    openmdao.api.Problem
        an OpenMDAO problem representing the wind farm analysis stack

    Raises
    ------
    NotImplementedError
        raised if a configuration we haven't prototyped out yet has been
        requested
    """

    if layout_type not in ["gridfarm", "sunflower"]:
        raise NotImplementedError(f"layout type {layout_type} is not implemented yet.")
    if aero_backend not in ["FLORIS"]:
        raise NotImplementedError(
            f"aerodynamic backend {aero_backend} is not implemented yet."
        )
    if wind_rose is None:
        raise NotImplementedError("this wind rose configuration is not implemented.")

    # create the OpenMDAO model
    model = om.Group()
    group_layout2aep = om.Group()

    # first the layout
    if layout_type == "gridfarm":
        group_layout2aep.add_subsystem(  # layout component
            "layout",
            gridfarm.GridFarmLayout(modeling_options=modeling_options),
            promotes=["*"],
        )
        layout_global_input_promotes = [
            "angle_orientation",
            "angle_skew",
            "spacing_primary",
            "spacing_secondary",
        ]
    elif layout_type == "sunflower":
        group_layout2aep.add_subsystem(  # layout component
            "layout",
            sunflower.SunflowerFarmLayout(modeling_options=modeling_options),
            promotes=["*"],
        )
        layout_global_input_promotes = ["spacing_target"]
    else:
        raise KeyError("you shouldn't be able to get here.")
    layout_global_output_promotes = [
        "spacing_effective_primary",
        "spacing_effective_secondary",
    ]  # all layouts have this

    group_layout2aep.add_subsystem(  # FLORIS AEP component
        "aepFLORIS",
        farmaero_floris.FLORISAEP(
            modeling_options=modeling_options,
            wind_rose=wind_rose,
            case_title="letsgo",
        ),
        # promotes=["AEP_farm"],
        promotes=["x_turbines", "y_turbines", "AEP_farm"],
    )
    farmaero_global_output_promotes = ["AEP_farm"]

    group_layout2aep.approx_totals(
        method="fd", step=1e-3, form="central", step_calc="rel_avg"
    )
    model.add_subsystem(
        "layout2aep",
        group_layout2aep,
        promotes_inputs=[
            *layout_global_input_promotes,
        ],
        promotes_outputs=[
            *layout_global_output_promotes,
            *farmaero_global_output_promotes,
        ],
    )

    if layout_type == "gridfarm":
        model.add_subsystem(  # landuse component
            "landuse",
            gridfarm.GridFarmLanduse(modeling_options=modeling_options),
            promotes_inputs=layout_global_input_promotes,
        )
    elif layout_type == "sunflower":
        model.add_subsystem(  # landuse component
            "landuse",
            sunflower.SunflowerFarmLanduse(modeling_options=modeling_options),
        )
        model.connect("layout2aep.x_turbines", "landuse.x_turbines")
        model.connect("layout2aep.y_turbines", "landuse.y_turbines")
    else:
        raise KeyError("you shouldn't be able to get here.")

    model.add_subsystem(  # turbine capital costs component
        "tcc",
        cost_wisdem.TurbineCapitalCosts(),
        promotes_inputs=[
            "turbine_number",
            "machine_rating",
            "tcc_per_kW",
            "offset_tcc_per_kW",
        ],
    )
    if modeling_options["offshore"]:
        model.add_subsystem(  # ORBIT component
            "orbit",
            cost_wisdem.ORBIT(),
        )
        model.connect(  # effective primary spacing for BOS
            "spacing_effective_primary", "orbit.plant_turbine_spacing"
        )
        model.connect(  # effective secondary spacing for BOS
            "spacing_effective_secondary", "orbit.plant_row_spacing"
        )
    else:
        model.add_subsystem(  # LandBOSSE component
            "landbosse",
            cost_wisdem.LandBOSSE(),
        )
        model.connect(  # effective primary spacing for BOS
            "spacing_effective_primary", "landbosse.turbine_spacing_rotor_diameters"
        )
        model.connect(  # effective secondary spacing for BOS
            "spacing_effective_secondary", "landbosse.row_spacing_rotor_diameters"
        )

    model.add_subsystem(  # operational expenditures component
        "opex",
        cost_wisdem.OperatingExpenses(),
        promotes_inputs=[
            "turbine_number",
            "machine_rating",
            "opex_per_kW",
        ],
    )

    model.add_subsystem(  # cost metrics component
        "financese",
        cost_wisdem.PlantFinance(),
        promotes_inputs=[
            "turbine_number",
            "machine_rating",
            "tcc_per_kW",
            "offset_tcc_per_kW",
            "opex_per_kW",
        ],
    )
    model.connect("AEP_farm", "financese.plant_aep_in")
    if modeling_options["offshore"]:
        model.connect("orbit.total_capex_kW", "financese.bos_per_kW")
    else:
        model.connect("landbosse.total_capex_kW", "financese.bos_per_kW")

    # build out the problem based on this model
    prob = om.Problem(model)
    if setup_glue:
        prob.setup()

    # return the problem
    return prob
