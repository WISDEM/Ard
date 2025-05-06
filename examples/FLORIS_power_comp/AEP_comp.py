from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import openmdao.api as om

import floris

import ard.utils.io
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
wind_rose_wrg.set_wd_step(1.0)
wind_rose_wrg.set_wind_speeds(np.arange(0, 30, 0.5)[1:])
wind_rose = wind_rose_wrg.get_wind_rose_at_point(0.0, 0.0)
wind_query = wq.WindQuery.from_FLORIS_WindData(wind_rose)

# # create the farm layout specification
# farm_spec = {}
# farm_spec["xD_farm"], farm_spec["yD_farm"] = [
#     7 * v.flatten() for v in np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
# ]
# if False:
#     plt.scatter(farm_spec["xD_farm"], farm_spec["yD_farm"])
#     plt.show()

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
        "N_turbines": 25,
    },
    "turbine": data_turbine_spec,
}

# create the OpenMDAO model
model = om.Group()
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

prob = om.Problem(model)
prob.setup()

prob.set_val("spacing_primary", 7.0)
prob.set_val("spacing_secondary", 7.0)
prob.set_val("angle_skew", 0.0)

orientation_vec = np.arange(0.0, 90.0, 2.5)
AEP_vec = np.zeros_like(orientation_vec)

for idx, angle_orientation in enumerate(orientation_vec):
    prob.set_val("angle_orientation", angle_orientation)
    prob.run_model()
    AEP_val = float(prob.get_val("aepFLORIS.AEP_farm", units="GW*h")[0])
    print(f"AEP: {AEP_val}")
    AEP_vec[idx] = AEP_val

plt.plot(orientation_vec, AEP_vec)
plt.show()

prob.set_val("angle_orientation", orientation_vec[np.argmax(AEP_vec)])
prob.run_model()

WS, WD = np.meshgrid(wind_rose.wind_speeds, wind_rose.wind_directions)
FREQ = 1000 * wind_rose.freq_table

plt.contourf(
    WD,
    WS,
    prob.get_val("aepFLORIS.power_farm", units="MW").reshape(WD.shape),
)
plt.colorbar()
plt.show()

plt.contourf(
    WD,
    WS,
    FREQ * 8760,
)
plt.colorbar()
plt.show()

plt.contourf(
    WD,
    WS,
    prob.get_val("aepFLORIS.power_farm", units="MW").reshape(WD.shape)
    * FREQ
    * 8760
    / 1000,
)
plt.colorbar()
plt.show()

### FIN!
