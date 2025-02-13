from pathlib import Path

import ard.viz.get_stylesheets
import ard.viz.plot_collection
import ard.viz.plot_layout
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import floris
import openmdao.api as om

from wisdem.optimization_drivers.nlopt_driver import NLoptDriver

import ard.utils
import ard.wind_query as wq
import ard.glue.prototype as glue
import ard.cost.wisdem_wrap as cost_wisdem
import ard.collection.interarray_wrap as inter
import ard.viz

plt.style.use(ard.viz.get_stylesheets(dark=True))

# create the wind query
wind_rose_wrg = floris.wind_data.WindRoseWRG(
    Path(
        Path(ard.__file__).parents[1],
        "examples",
        "data",
        "wrg_example.wrg",
    )
)
wind_rose_wrg.set_wd_step(1.0)
wind_rose_wrg.set_wind_speeds(np.arange(0, 30, 0.5)[1:])
wind_rose = wind_rose_wrg.get_wind_rose_at_point(0.0, 0.0)
wind_query = wq.WindQuery.from_FLORIS_WindData(wind_rose)

# specify the configuration/specification files to use
filename_turbine_spec = Path(
    Path(ard.__file__).parents[1],
    "examples",
    "data",
    "turbine_spec_IEA-3p4-130-RWT.yaml",
)
data_turbine_spec = ard.utils.load_turbine_spec(filename_turbine_spec)

# set up the modeling options
N_turbines = 50
N_substations = 3
modeling_options = {
    "farm": {
        "N_turbines": N_turbines,
        "N_substations": N_substations,
    },
    "turbine": data_turbine_spec,
    "offshore": False,
}

# create the OM problem
prob = glue.create_setup_OM_problem(
    layout_type="sunflower",
    modeling_options=modeling_options,
    wind_rose=wind_rose,
    setup_glue=False,
)

# add collection system
prob.model.add_subsystem(
    "collection",
    inter.InterarrayCollection(modeling_options=modeling_options),
    promotes=["x_substations", "y_substations"],
)
prob.model.connect("layout2aep.x_turbines", "collection.x_turbines")
prob.model.connect("layout2aep.y_turbines", "collection.y_turbines")

# add the basic viz for the layout
prob.model.add_subsystem(
    "viz_layout",
    ard.viz.plot_layout.OutputLayout(modeling_options=modeling_options),
)
prob.model.connect("layout2aep.x_turbines", "viz_layout.x_turbines")
prob.model.connect("layout2aep.y_turbines", "viz_layout.y_turbines")

# add the interarray viz for the layout
prob.model.add_subsystem(
    "viz_collection",
    ard.viz.plot_collection.OutputCollection(modeling_options=modeling_options),
    promotes=["x_substations", "y_substations"],
)
prob.model.connect("layout2aep.x_turbines", "viz_collection.x_turbines")
prob.model.connect("layout2aep.y_turbines", "viz_collection.y_turbines")
prob.model.connect("collection.load_cables", "viz_collection.load_cables")
prob.model.connect("collection.edges", "viz_collection.edges")

prob.setup()

# set the substations
theta_vec = np.linspace(0.0, 2*np.pi, N_substations+1)[:-1]
prob.set_val("x_substations", 1000.0 * np.sin(theta_vec))
prob.set_val("y_substations", 1000.0 * np.cos(theta_vec))

# set up the working/design variables
prob.model.add_design_var("spacing_target", lower=2.0, upper=13.0)
prob.model.add_objective("collection.total_length_cables", units="km")
prob.model.add_constraint("landuse.area_tight", lower=15.0, units="km**2")

# setup the latent variables for LandBOSSE and FinanceSE
cost_wisdem.LandBOSSE_setup_latents(prob, modeling_options)
cost_wisdem.FinanceSE_setup_latents(prob, modeling_options)

# set up the working/design variables
prob.set_val("spacing_target", 7.0)

# run the model
prob.run_model()

print(
    f"total cable length: {prob.get_val('collection.total_length_cables', units="km")}"
)
print(f"landuse area: {prob.get_val('landuse.area_tight', units="km**2")}")
print(f"AEP: {prob.get_val('AEP_farm', units="GW*h")}")

plt.show()
