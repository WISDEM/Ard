from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import openmdao.api as om

import ard.utils.io
import ard.wind_query as wq
import ard.farm_aero.floris as farmaero_floris

# create the wind query
directions = np.linspace(0.0, 360.0, 101)
speeds = np.linspace(0.0, 30.0, 101)[1:]
WS, WD = np.meshgrid(speeds, directions)
wind_query = wq.WindQuery(WD.flatten(), WS.flatten())
wind_query.set_TI_using_IEC_method()

# create the farm layout specification
farm_spec = {}
farm_spec["xD_farm"], farm_spec["yD_farm"] = [
    7 * v.flatten() for v in np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
]
if False:
    plt.scatter(farm_spec["xD_farm"], farm_spec["yD_farm"])
    plt.show()

# specify the configuration/specification files to use
filename_turbine_spec = Path(
    "..",
    "data",
    "turbine_spec_IEA-3p4-130-RWT.yaml",
)  # toolset generalized turbine specification
data_turbine_spec = ard.utils.io.load_turbine_spec(filename_turbine_spec)

# set up the modeling options
modeling_options = {
    "farm": {
        "N_turbines": len(farm_spec["xD_farm"]),
    },
    "turbine": data_turbine_spec,
}

# create the OpenMDAO model
model = om.Group()
model.add_subsystem(
    "batchFLORIS",
    farmaero_floris.FLORISBatchPower(
        modeling_options=modeling_options,
        wind_query=wind_query,
        case_title="letsgo",
    ),
)

prob = om.Problem(model)
prob.setup()

prob.set_val("batchFLORIS.x_turbines", 130.0 * farm_spec["xD_farm"])
prob.set_val("batchFLORIS.y_turbines", 130.0 * farm_spec["yD_farm"])

prob.run_model()

print(prob.get_val("batchFLORIS.power_farm", units="GW"))

plt.contourf(
    WD,
    WS,
    np.max(prob.get_val("batchFLORIS.power_turbines", units="MW"), axis=0).T.reshape(
        WD.shape
    ),
)
plt.colorbar()
plt.show()

plt.contourf(
    WD,
    WS,
    np.max(prob.get_val("batchFLORIS.thrust_turbines", units="kN"), axis=0).T.reshape(
        WD.shape
    ),
)
plt.colorbar()
plt.show()

### FIN!
