import pprint as pp

import numpy as np
import matplotlib.pyplot as plt

import optiwindnet.plotting
from ard.utils.io import load_yaml
from ard.api import set_up_ard_model
import openmdao.api as om


def run_example():

    # load input
    input_dict = load_yaml("./inputs/ard_system.yaml")

    # set up system
    prob = set_up_ard_model(input_dict=input_dict, root_data_path="inputs")

    if False:
        # visualize model
        om.n2(prob)

    # run the model
    prob.run_model()

    # collapse the test result data
    test_data = {
        "AEP_val": float(prob.get_val("AEP_farm", units="GW*h")[0]),
        "CapEx_val": float(prob.get_val("tcc.tcc", units="MUSD")[0]),
        "BOS_val": float(prob.get_val("total_capex", units="MUSD")[0]),
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

    print("\n\nRESULTS:\n")
    pp.pprint(test_data)
    print("\n\n")

    optimize = True  # set to False to skip optimization

    if optimize:

        # run the optimization
        prob.run_driver()
        prob.cleanup()

        # collapse the test result data
        test_data = {
            "AEP_val": float(prob.get_val("AEP_farm", units="GW*h")[0]),
            "CapEx_val": float(prob.get_val("tcc.tcc", units="MUSD")[0]),
            "BOS_val": float(prob.get_val("total_capex", units="MUSD")[0]),
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


#         # plot convergence
#         ## read cases
#         cr = om.CaseReader(
#             prob.get_outputs_dir()
#             / input_dict["analysis_options"]["recorder"]["filepath"]
#         )
#
#         # Extract the driver cases
#         cases = cr.get_cases("driver")
#
#         # Initialize lists to store iteration data
#         iterations = []
#         objective_values = []
#
#         # Loop through the cases and extract iteration number and objective value
#         for i, case in enumerate(cases):
#             iterations.append(i)
#             objective_values.append(
#                 case.get_objectives()[
#                     input_dict["analysis_options"]["objective"]["name"]
#                 ]
#             )
#
#         # Plot the convergence
#         plt.figure(figsize=(8, 6))
#         plt.plot(iterations, objective_values, marker="o", label="Objective (LCOE)")
#         plt.xlabel("Iteration")
#         plt.ylabel("Objective Value (Total Cable Length (m))")
#         plt.title("Convergence Plot")
#         plt.legend()
#         plt.grid()
#         plt.show()
#
#     optiwindnet.plotting.gplot(prob.model.optiwindnet_coll.graph)
#
#     plt.show()

if __name__ == "__main__":

    run_example()
