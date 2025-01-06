import openmdao.api as om

import floris.wind_data

import ard.layout.gridfarm as gridfarm
import ard.farm_aero.floris as farmaero_floris
import ard.cost.wisdem_wrap as cost_wisdem


def create_setup_OM_problem(
    modeling_options,
    wind_rose: floris.wind_data.WindRose = None,
    aero_backend: str = "FLORIS",
    layout_type: str = "gridfarm",
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

    if layout_type not in ["gridfarm"]:
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
    group_layout2aep.add_subsystem(  # layout component
        "layout",
        gridfarm.GridFarmLayout(modeling_options=modeling_options),
        promotes_inputs=[
            "spacing_primary",
            "spacing_secondary",
            "angle_orientation",
            "angle_skew",
        ],
        promotes_outputs=[
            "spacing_effective_primary",
            "spacing_effective_secondary",
        ],
    )
    group_layout2aep.add_subsystem(  # landuse component
        "landuse",
        gridfarm.GridFarmLanduse(modeling_options=modeling_options),
        promotes_inputs=[
            "spacing_primary",
            "spacing_secondary",
            "angle_orientation",
            "angle_skew",
        ],
        promotes_outputs=["area_tight"],
    )
    group_layout2aep.add_subsystem(  # FLORIS AEP component
        "aepFLORIS",
        farmaero_floris.FLORISAEP(
            modeling_options=modeling_options,
            wind_rose=wind_rose,
            case_title="letsgo",
        ),
        promotes_outputs=["AEP_farm"],
    )
    group_layout2aep.connect("layout.x_turbines", "aepFLORIS.x_turbines")
    group_layout2aep.connect("layout.y_turbines", "aepFLORIS.y_turbines")
    g_l2a = model.add_subsystem(
        "layout2aep",
        group_layout2aep,
        promotes=[
            "angle_orientation",
            "angle_skew",
            "spacing_primary",
            "spacing_secondary",
            "spacing_effective_primary",
            "spacing_effective_secondary",
            "AEP_farm",
        ],
    )
    g_l2a.approx_totals(
        method="fd", step=1.0e-1, form="forward", step_calc="abs",
    )
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
    model.add_subsystem(  # LandBOSSE component
        "landbosse",
        cost_wisdem.LandBOSSE(),
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
    model.connect(  # effective primary spacing for BOS
        "spacing_effective_primary", "landbosse.turbine_spacing_rotor_diameters"
    )
    model.connect(  # effective secondary spacing for BOS
        "spacing_effective_secondary", "landbosse.row_spacing_rotor_diameters"
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
    model.connect("landbosse.bos_capex_kW", "financese.bos_per_kW")

    # set default number of turbines to one
    model.set_input_defaults('turbine_number', val=1)

    model.approx_totals(
        method="fd", step=1e-1, form="forward", step_calc="abs",
    )  # DEBUG!!!!!

    # build out the problem based on this model
    prob = om.Problem(model)
    prob.setup()

    # return the problem
    return prob
