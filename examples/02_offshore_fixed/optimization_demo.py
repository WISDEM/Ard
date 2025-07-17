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

    # run the model
    prob.run_model()

    if False:
        # visualize model
        om.n2(prob)

    # collapse the test result data
    test_data = {
        "AEP_val": float(prob.get_val("AEP_farm", units="GW*h")[0]),
        "CapEx_val": float(prob.get_val("tcc.tcc", units="MUSD")[0]),
        "BOS_val": float(prob.get_val("orbit.total_capex", units="MUSD")[0]),
        "OpEx_val": float(prob.get_val("opex.opex", units="MUSD/yr")[0]),
        "LCOE_val": float(prob.get_val("financese.lcoe", units="USD/MW/h")[0]),
        "area_tight": float(prob.get_val("landuse.area_tight", units="km**2")[0]),
        "coll_length": float(
            prob.get_val("collection.total_length_cables", units="km")[0]
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
            "BOS_val": float(prob.get_val("orbit.total_capex", units="MUSD")[0]),
            "OpEx_val": float(prob.get_val("opex.opex", units="MUSD/yr")[0]),
            "LCOE_val": float(prob.get_val("financese.lcoe", units="USD/MW/h")[0]),
            "area_tight": float(prob.get_val("landuse.area_tight", units="km**2")[0]),
            "coll_length": float(
                prob.get_val("collection.total_length_cables", units="km")[0]
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

        # plot convergence
        ## read cases
        cr = om.CaseReader(
            prob.get_outputs_dir()
            / input_dict["analysis_options"]["recorder"]["filepath"]
        )

        # Extract the driver cases
        cases = cr.get_cases("driver")

        # Initialize lists to store iteration data
        iterations = []
        objective_values = []

        # Loop through the cases and extract iteration number and objective value
        for i, case in enumerate(cases):
            iterations.append(i)
            objective_values.append(
                case.get_objectives()[
                    input_dict["analysis_options"]["objective"]["name"]
                ]
            )

        # Plot the convergence
        plt.figure(figsize=(8, 6))
        plt.plot(iterations, objective_values, marker="o", label="Objective (LCOE)")
        plt.xlabel("Iteration")
        plt.ylabel("Objective Value (Total Cable Length (m))")
        plt.title("Convergence Plot")
        plt.legend()
        plt.grid()
        plt.show()

    optiwindnet.plotting.gplot(prob.model.collection.graph)

    plt.show()


if __name__ == "__main__":

    run_example()

# RESULTS:


# {'AEP_val': 4818.0,
#  'BOS_val': 2127.5924853696597,
#  'CapEx_val': 768.4437570425,
#  'LCOE_val': 57.63858824842508,
#  'OpEx_val': 60.50000000000001,
#  'area_tight': 63.234304,
#  'coll_length': 47.761107521256534,
#  'mooring_spacing': 1.1208759839268934}


# /opt/anaconda3/envs/ard/lib/python3.12/site-packages/openmdao/recorders/sqlite_recorder.py:231: UserWarning:The existing case recorder file, /Users/jthomas2/Documents/programs/Ard/examples/offshore/optimization_demo_out/opt_results.sql, is being overwritten.
# Optimization terminated successfully    (Exit mode 0)
#             Current function value: 20495.94468696311
#             Iterations: 13
#             Function evaluations: 9
#             Gradient evaluations: 9
# Optimization Complete
# -----------------------------------


# RESULTS (opt):

# {'AEP_val': 4818.0,
#  'BOS_val': 2113.9686662769814,
#  'CapEx_val': 768.4437570425,
#  'LCOE_val': 57.42651136342073,
#  'OpEx_val': 60.50000000000001,
#  'area_tight': 11.614464,
#  'coll_length': 20.49594468696311,
#  'mooring_spacing': 0.0582385263254008,
#  'turbine_spacing': 0.8519999999999998}


## 20250714
# RESULTS:

# {'AEP_val': 4818.0,
#  'BOS_val': 1431.448205129097,
#  'CapEx_val': 768.4437570425,
#  'LCOE_val': 46.80197118365915,
#  'OpEx_val': 60.50000000000001,
#  'area_tight': 63.234304,
#  'coll_length': 47.712041428901635}

# RESULTS (opt):

# {'AEP_val': 4818.0,
#  'BOS_val': 1412.8143111685533,
#  'CapEx_val': 768.4437570425,
#  'LCOE_val': 46.51190434118493,
#  'OpEx_val': 60.50000000000001,
#  'area_tight': 11.614464,
#  'coll_length': 20.507898644867613,
#  'turbine_spacing': 0.8519999999999999}
