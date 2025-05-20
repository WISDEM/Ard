from pathlib import Path

import pprint as pp

import numpy as np
import matplotlib.pyplot as plt

import floris
import openmdao.api as om

from wisdem.optimization_drivers.nlopt_driver import NLoptDriver

import optiwindnet.plotting
import ard
import ard.glue.prototype as glue
import ard.layout.spacing

# layout type
layout_type = "gridfarm"

# create the wind query
wind_rose_wrg = floris.wind_data.WindRoseWRG(
    Path(ard.__file__).parents[1] / "examples" / "data" / "wrg_example.wrg"
)
wind_rose_wrg.set_wd_step(90.0)
wind_rose_wrg.set_wind_speeds(np.array([5.0, 10.0, 15.0, 20.0]))
wind_rose = wind_rose_wrg.get_wind_rose_at_point(0.0, 0.0)
wind_query = ard.wind_query.WindQuery.from_FLORIS_WindData(wind_rose)

# specify the configuration/specification files to use
filename_turbine_spec = (
    Path(ard.__file__).parents[1]
    / "examples"
    / "data"
    / "turbine_spec_IEA-22-284-RWT.yaml"
)  # toolset generalized turbine specification
data_turbine_spec = ard.utils.io.load_turbine_spec(filename_turbine_spec)

# set up the modeling options
modeling_options = {
    "farm": {
        "N_turbines": 25,
        "N_substations": 1,
    },
    "turbine": data_turbine_spec,
    "offshore": True,
    "floating": False,
    "site_depth": 50.0,
    "collection": {
        "max_turbines_per_string": 8,
        "solver_name": "appsi_highs",
        "solver_options": dict(
            time_limit=60,
            mip_rel_gap=0.005,
        ),
    },
}

# create the OpenMDAO model
model = om.Group()
group_layout2aep = om.Group()

# first the layout
if layout_type == "gridfarm":
    group_layout2aep.add_subsystem(  # layout component
        "layout",
        ard.layout.gridfarm.GridFarmLayout(modeling_options=modeling_options),
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
        ard.layout.sunflower.SunflowerFarmLayout(modeling_options=modeling_options),
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
    "aepPlaceholder",
    ard.farm_aero.placeholder.PlaceholderAEP(
        modeling_options=modeling_options,
        wind_rose=wind_rose,
    ),
    # promotes=["AEP_farm"],
    promotes=["x_turbines", "y_turbines", "AEP_farm"],
)
# group_layout2aep.add_subsystem(  # FLORIS AEP component
#     "aepFLORIS",
#     ard.farm_aero.floris.FLORISAEP(
#         modeling_options=modeling_options,
#         wind_rose=wind_rose,
#         case_title="letsgo",
#     ),
#     # promotes=["AEP_farm"],
#     promotes=["x_turbines", "y_turbines", "AEP_farm"],
# )
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
        ard.layout.gridfarm.GridFarmLanduse(modeling_options=modeling_options),
        promotes_inputs=layout_global_input_promotes,
    )
elif layout_type == "sunflower":
    model.add_subsystem(  # landuse component
        "landuse",
        ard.layout.sunflower.SunflowerFarmLanduse(modeling_options=modeling_options),
    )
    model.connect("layout2aep.x_turbines", "landuse.x_turbines")
    model.connect("layout2aep.y_turbines", "landuse.y_turbines")
else:
    raise KeyError("you shouldn't be able to get here.")

model.add_subsystem(  # collection component
    "optiwindnet_coll",
    ard.collection.optiwindnetCollection(
        modeling_options=modeling_options,
    ),
)
model.connect("layout2aep.x_turbines", "optiwindnet_coll.x_turbines")
model.connect("layout2aep.y_turbines", "optiwindnet_coll.y_turbines")

model.add_subsystem(  # constraints for turbine proximity
    "spacing_constraint",
    ard.layout.spacing.TurbineSpacing(
        modeling_options=modeling_options,
    ),
)
model.connect("layout2aep.x_turbines", "spacing_constraint.x_turbines")
model.connect("layout2aep.y_turbines", "spacing_constraint.y_turbines")

model.add_subsystem(  # turbine capital costs component
    "tcc",
    ard.cost.wisdem_wrap.TurbineCapitalCosts(),
    promotes_inputs=[
        "turbine_number",
        "machine_rating",
        "tcc_per_kW",
        "offset_tcc_per_kW",
    ],
)
if modeling_options["offshore"]:
    model.add_subsystem(  # Orbit component
        "orbit",
        ard.cost.wisdem_wrap.Orbit(floating=True),
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
        ard.cost.wisdem_wrap.LandBOSSE(),
    )
    model.connect(  # effective primary spacing for BOS
        "spacing_effective_primary",
        "landbosse.turbine_spacing_rotor_diameters",
    )
    model.connect(  # effective secondary spacing for BOS
        "spacing_effective_secondary",
        "landbosse.row_spacing_rotor_diameters",
    )

model.add_subsystem(  # operational expenditures component
    "opex",
    ard.cost.wisdem_wrap.OperatingExpenses(),
    promotes_inputs=[
        "turbine_number",
        "machine_rating",
        "opex_per_kW",
    ],
)

model.add_subsystem(  # cost metrics component
    "financese",
    ard.cost.wisdem_wrap.PlantFinance(),
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
prob.setup()

# setup the latent variables for LandBOSSE/ORBIT and FinanceSE
ard.cost.wisdem_wrap.ORBIT_setup_latents(prob, modeling_options)
# ard.cost.wisdem_wrap.LandBOSSE_setup_latents(prob, modeling_options)
ard.cost.wisdem_wrap.FinanceSE_setup_latents(prob, modeling_options)

# set up the working/design variables
prob.set_val("spacing_primary", 7.0)
prob.set_val("spacing_secondary", 7.0)
prob.set_val("angle_orientation", 0.0)
prob.set_val("angle_skew", 0.0)

prob.set_val("optiwindnet_coll.x_substations", [100.0])
prob.set_val("optiwindnet_coll.y_substations", [100.0])

# run the model
prob.run_model()

# collapse the test result data
test_data = {
    "AEP_val": float(prob.get_val("AEP_farm", units="GW*h")[0]),
    "CapEx_val": float(prob.get_val("tcc.tcc", units="MUSD")[0]),
    "BOS_val": float(prob.get_val("orbit.total_capex", units="MUSD")[0]),
    # "BOS_val": float(prob.get_val("landbosse.total_capex", units="MUSD")[0]),
    "OpEx_val": float(prob.get_val("opex.opex", units="MUSD/yr")[0]),
    "LCOE_val": float(prob.get_val("financese.lcoe", units="USD/MW/h")[0]),
    "area_tight": float(prob.get_val("landuse.area_tight", units="km**2")[0]),
    "coll_length": float(
        prob.get_val("optiwindnet_coll.total_length_cables", units="km")[0]
    ),
}

print("\n\nRESULTS:\n")
pp.pprint(test_data)
print("\n\n")

optimize = True
if optimize:
    # now set up an optimization driver

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"

    prob.model.add_design_var("spacing_primary", lower=3.0, upper=10.0)
    prob.model.add_design_var("spacing_secondary", lower=3.0, upper=10.0)
    prob.model.add_design_var("angle_orientation", lower=-180.0, upper=180.0)
    prob.model.add_design_var("angle_skew", lower=-75.0, upper=75.0)
    prob.model.add_constraint(
        "spacing_constraint.turbine_spacing", units="m", lower=284.0 * 3.0
    )
    # prob.model.add_constraint("landuse.area_tight", units="km**2", lower=50.0)
    prob.model.add_objective("optiwindnet_coll.total_length_cables")

    # create a recorder
    recorder = om.SqliteRecorder("opt_results.sql")

    # add the recorder to the problem
    prob.add_recorder(recorder)
    # add the recorder to the driver
    prob.driver.add_recorder(recorder)

    # set up the problem
    prob.setup()

    # setup the latent variables for LandBOSSE/ORBIT and FinanceSE
    ard.cost.wisdem_wrap.ORBIT_setup_latents(prob, modeling_options)
    # ard.cost.wisdem_wrap.LandBOSSE_setup_latents(prob, modeling_options)
    ard.cost.wisdem_wrap.FinanceSE_setup_latents(prob, modeling_options)

    # set up the working/design variables initial conditions
    prob.set_val("spacing_primary", 7.0)
    prob.set_val("spacing_secondary", 7.0)
    prob.set_val("angle_orientation", 0.0)
    prob.set_val("angle_skew", 0.0)

    prob.set_val("optiwindnet_coll.x_substations", [100.0])
    prob.set_val("optiwindnet_coll.y_substations", [100.0])

    # run the optimization
    prob.run_driver()

    # collapse the test result data
    test_data = {
        "AEP_val": float(prob.get_val("AEP_farm", units="GW*h")[0]),
        "CapEx_val": float(prob.get_val("tcc.tcc", units="MUSD")[0]),
        "BOS_val": float(prob.get_val("orbit.total_capex", units="MUSD")[0]),
        # "BOS_val": float(prob.get_val("landbosse.total_capex", units="MUSD")[0]),
        "OpEx_val": float(prob.get_val("opex.opex", units="MUSD/yr")[0]),
        "LCOE_val": float(prob.get_val("financese.lcoe", units="USD/MW/h")[0]),
        "area_tight": float(prob.get_val("landuse.area_tight", units="km**2")[0]),
        "coll_length": float(
            prob.get_val("optiwindnet_coll.total_length_cables", units="km")[0]
        ),
        "turbine_spacing": float(
            np.min(prob.get_val("spacing_constraint.turbine_spacing", units="km"))
        ),
    }

    # clean up the recorder
    prob.cleanup()

    # print the results
    print("\n\nRESULTS (opt):\n")
    pp.pprint(test_data)
    print("\n\n")

optiwindnet.plotting.gplot(prob.model.optiwindnet_coll.graph)
plt.show()
