from pathlib import Path

from pprint import pprint

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
print(f"turbines:")
for i_t, (x_t, y_t) in enumerate(zip(
    prob.get_val('x_turbines', units='km'), prob.get_val('x_turbines', units='km')
)):
    print(f"\t{i_t:03d}: ({x_t:.03f} km, {y_t:.03f} km)")
print(f"AEP result: {prob.get_val('AEP_farm', units='GW*h')[0]} GWh")
print(f"landuse result: {prob.get_val('landuse.area_tight', units='km**2')[0]} sq. km")
