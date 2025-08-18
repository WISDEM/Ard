from pathlib import Path

from pprint import pprint

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import windIO

from ard.utils.io import load_yaml
from ard.api import set_up_ard_model

# get, validate, and load a windIO file
path_windIO = Path(__file__).parent / "inputs" / "windio.yaml"
windIO.validate(input=path_windIO, schema_type="plant/wind_energy_system")
windIOdict = windIO.load_yaml(path_windIO)

# load the Ard system input
path_ard_system = Path(__file__).parent / "inputs" / "ard_system.yaml"
input_dict = load_yaml(path_ard_system)

# build an Ard model using the setup
prob = set_up_ard_model(input_dict=input_dict, root_data_path="inputs")

# run the model
prob.run_model()

# print the AEP result
print(f"\n\n\nINITIAL RESULTS:\n")
print(f"turbines:")
for i_t, (x_t, y_t) in enumerate(
    zip(prob.get_val("x_turbines", units="km"), prob.get_val("x_turbines", units="km"))
):
    print(f"\t{i_t:03d}: ({x_t:.03f} km, {y_t:.03f} km)")
print(f"AEP result: {prob.get_val('AEP_farm', units='GW*h')[0]} GWh")
print(f"landuse result: {prob.get_val('landuse.area_tight', units='km**2')[0]} sq. km")
print(
    f"total cable length: {prob.get_val('optiwindnet_coll.total_length_cables', units='km')[0]} km"
)
print(f"boundary distances: {prob.get_val('boundary_distances', units='km')}")

run_optimize = True
if run_optimize:
    # run the driver
    prob.run_driver()

    # print the AEP result
    print(f"\n\n\nOPTIMIZED RESULTS:\n")
    print(f"turbines:")
    for i_t, (x_t, y_t) in enumerate(
        zip(
            prob.get_val("x_turbines", units="km"),
            prob.get_val("x_turbines", units="km"),
        )
    ):
        print(f"\t{i_t:03d}: ({x_t:.03f} km, {y_t:.03f} km)")
    print(f"AEP result: {prob.get_val('AEP_farm', units='GW*h')[0]} GWh")
    print(
        f"landuse result: {prob.get_val('landuse.area_tight', units='km**2')[0]} sq. km"
    )
    print(
        f"total cable length: {prob.get_val('optiwindnet_coll.total_length_cables', units='km')[0]} km"
    )
    print(f"boundary distances: {prob.get_val('boundary_distances', units='km')}")


# get plot limits based on the farm boundaries
def get_limits(windIOdict, lim_buffer=0.05):
    x_lim = [
        np.min(windIOdict["site"]["boundaries"]["polygons"][0]["x"])
        - lim_buffer * np.ptp(windIOdict["site"]["boundaries"]["polygons"][0]["x"]),
        np.max(windIOdict["site"]["boundaries"]["polygons"][0]["x"])
        + lim_buffer * np.ptp(windIOdict["site"]["boundaries"]["polygons"][0]["x"]),
    ]
    y_lim = [
        np.min(windIOdict["site"]["boundaries"]["polygons"][0]["y"])
        - lim_buffer * np.ptp(windIOdict["site"]["boundaries"]["polygons"][0]["y"]),
        np.max(windIOdict["site"]["boundaries"]["polygons"][0]["y"])
        + lim_buffer * np.ptp(windIOdict["site"]["boundaries"]["polygons"][0]["y"]),
    ]
    return x_lim, y_lim


# get the turbine locations to plot
x_turbines = prob.get_val("x_turbines", units="km")
y_turbines = prob.get_val("y_turbines", units="km")

# make a plot
fig, ax = plt.subplots()
ax.fill(
    windIOdict["site"]["boundaries"]["polygons"][0]["x"],
    windIOdict["site"]["boundaries"]["polygons"][0]["y"],
    linestyle="--",
    alpha=0.5,
    fill=False,
)
ax.plot(x_turbines, y_turbines, "ok")
x_lim, y_lim = get_limits(windIOdict)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
plt.show()
