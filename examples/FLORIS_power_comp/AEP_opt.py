import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import openmdao.api as om

import floris

from wisdem.optimization_drivers.nlopt_driver import NLoptDriver

import ard.utils
import ard.wind_query as wq
import ard.layout.gridfarm as gridfarm
import ard.farm_aero.floris as farmaero_floris


# create the wind query
wind_rose_wrg = floris.wind_data.WindRoseWRG(
    Path(
        "..",
        "data",
        "wrg_example.wrg",
    )
)
wind_rose_wrg.set_wd_step(5.0)
wind_rose_wrg.set_wind_speeds(np.arange(0, 30, 2.5)[1:])
wind_rose = wind_rose_wrg.get_wind_rose_at_point(0.0, 0.0)
wind_query = wq.WindQuery.from_FLORIS_WindData(wind_rose)

# specify the configuration/specification files to use
filename_turbine_spec = Path(
    "..",
    "data",
    "turbine_spec_IEA-3p4-130-RWT.yaml",
)  # toolset generalized turbine specification
data_turbine_spec = ard.utils.load_turbine_spec(filename_turbine_spec)

# set up the modeling options
modeling_options = {
    "farm": {
        "N_turbines": 4,
    },
    "turbine": data_turbine_spec,
}

# create the OpenMDAO model
prob = om.Problem()
model = prob.model

model.add_subsystem(
    "layout",
    gridfarm.GridFarmLayout(modeling_options=modeling_options),
    promotes=["*"],
)

model.add_subsystem(
    "aepFLORIS",
    farmaero_floris.FLORISAEP(
        modeling_options=modeling_options,
        wind_rose=wind_rose,
        case_title="letsgo",
    ),
    promotes=["x_turbines", "y_turbines"],
)

prob.model.approx_totals(
    method="fd",
    step=5.0e0,
    form="central",
    step_calc="abs",
)

prob.setup()

prob.set_val("spacing_primary", 7.0)
prob.set_val("spacing_secondary", 7.0)
prob.set_val("angle_skew", 0.0)

orientation_vec = np.arange(-90.0, 90.0, 5.0)
AEP_vec = np.zeros_like(orientation_vec)
dAEP_vec = np.zeros_like(orientation_vec)

for idx, angle_orientation in enumerate(orientation_vec):
    prob.set_val("angle_orientation", angle_orientation)
    prob.run_model()
    dv_val = list(
        prob.compute_totals(of="aepFLORIS.AEP_farm", wrt="angle_orientation").values()
    )[0]
    AEP_val = float(prob.get_val("aepFLORIS.AEP_farm", units="GW*h")[0])
    print(f"AEP: {AEP_val}")
    AEP_vec[idx] = AEP_val
    dAEP_vec[idx] = dv_val

fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(orientation_vec, AEP_vec)
axes[1].plot(orientation_vec, dAEP_vec)
axes[1].plot(
    orientation_vec[1:-1],
    (np.diff(AEP_vec[1:]) + np.diff(AEP_vec[0:-1]))
    / (np.diff(orientation_vec[1:]) + np.diff(orientation_vec[0:-1])),
)
plt.show()

# prob.driver = NLoptDriver(optimizer="LD_SLSQP")
# prob.driver.options["debug_print"] = ["desvars", "nl_cons", "ln_cons", "objs"]
prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP")
prob.driver.options["debug_print"] = ["desvars", "nl_cons", "ln_cons", "objs"]

# set up the working/design variables
prob.model.add_design_var(
    "angle_orientation",
    lower=-90.0,
    upper=90.0,
    units="deg",
    ref=90.0,
)
prob.model.add_design_var(
    "angle_skew",
    lower=-90.0,
    upper=90.0,
    units="deg",
    ref=90.0,
)
prob.model.add_objective(
    "aepFLORIS.AEP_farm",
    units="GW*h",
    ref=500.0,
)

prob.set_val("spacing_primary", 7.0)
prob.set_val("spacing_secondary", 7.0)
prob.set_val("angle_skew", 0.0)
prob.set_val("angle_orientation", 0.0)

# setup the problem
prob.setup()

# run the optimization driver
prob.run_driver()

print(prob.get_val("angle_orientation", "deg"))


### FIN!
