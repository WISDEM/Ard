from pathlib import Path

import numpy as np

import floris
import openmdao.api as om

from wisdem.optimization_drivers.nlopt_driver import NLoptDriver

import ard.utils
import ard.wind_query as wq
import ard.glue.prototype as glue
import ard.cost.wisdem_wrap as cost_wisdem

# create the wind query
wind_rose_wrg = floris.wind_data.WindRoseWRG(
    Path(
        Path(ard.__file__).parent,
        "..",
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
    Path(ard.__file__).parent,
    "..",
    "examples",
    "data",
    "turbine_spec_IEA-22-284-RWT.yaml",
)
data_turbine_spec = ard.utils.load_turbine_spec(filename_turbine_spec)

# set up the modeling options
modeling_options = {
    "farm": {"N_turbines": 25},
    "site_depth": 50.0,
    "turbine": data_turbine_spec,
    "offshore": True,
    "floating": True,
}

# create the OM problem
prob = glue.create_setup_OM_problem(
    modeling_options=modeling_options,
    wind_rose=wind_rose,
)

if False:  # set true to run one-shot analysis

    # setup the latent variables for Orbit and FinanceSE
    cost_wisdem.Orbit_setup_latents(prob, modeling_options)
    cost_wisdem.FinanceSE_setup_latents(prob, modeling_options)

    # set up the working/design variables
    prob.set_val("spacing_primary", 7.0)
    prob.set_val("spacing_secondary", 7.0)
    prob.set_val("angle_orientation", 0.0)
    prob.set_val("angle_skew", 0.0)

    # run the model
    prob.run_model()

else:

    # set up the working/design variables
    prob.model.add_design_var("spacing_primary", lower=1.0, upper=13.0)
    prob.model.add_design_var("spacing_secondary", lower=1.0, upper=13.0)
    prob.model.add_design_var("angle_orientation", lower=-90.0, upper=90.0)
    prob.model.add_design_var("angle_skew", lower=-90.0, upper=90.0)
    prob.model.add_objective("financese.lcoe")
    # prob.model.add_objective("landuse.area_tight")

    # setup an optimization
    if False:  # for SLSQP from pyoptsparse
        prob.driver = om.pyOptSparseDriver(optimizer="SLSQP")
    elif True:  # use COBYLA from NLopt via WISDEM
        prob.driver = NLoptDriver(optimizer="LN_COBYLA")
        prob.driver.options["debug_print"] = ["desvars", "nl_cons", "ln_cons", "objs"]
        prob.driver.options["maxiter"] = 25
    elif True:  # use SLSQP from NLopt via WISDEM
        prob.driver = NLoptDriver(optimizer="LD_SLSQP")
        prob.driver.options["debug_print"] = ["desvars", "nl_cons", "ln_cons", "objs"]
    elif False:  # use COBYLA from scipy
        prob.driver = om.ScipyOptimizeDriver(optimizer="COBYLA")
        prob.driver.options["debug_print"] = ["desvars", "nl_cons", "ln_cons", "objs"]
    elif True:  # use SLSQP from scipy
        prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP")
        prob.driver.options["debug_print"] = ["desvars", "nl_cons", "ln_cons", "objs"]
    else:  # use Differential Evolution from OpenMDAO
        # this didn't really work
        prob.driver = om.DifferentialEvolutionDriver()
        prob.driver.options["max_gen"] = 10  # DEBUG!!!!! short
        prob.driver.options["pop_size"] = 5  # DEBUG!!!!! short
        # prob.driver.options["Pc"] = 0.5
        # prob.driver.options["F"] = 0.5
        prob.driver.options["run_parallel"] = True
        prob.driver.options["debug_print"] = ["desvars", "nl_cons", "ln_cons", "objs"]

    # set up the recorder
    prob.driver.recording_options["record_objectives"] = True
    prob.driver.recording_options["record_constraints"] = True
    prob.driver.recording_options["record_desvars"] = True
    prob.driver.recording_options["record_residuals"] = True
    prob.driver.add_recorder(om.SqliteRecorder("case.sql"))

    # setup the problem
    prob.setup()

    # setup the latent variables for Orbit and FinanceSE
    cost_wisdem.Orbit_setup_latents(prob, modeling_options)
    cost_wisdem.FinanceSE_setup_latents(prob, modeling_options)

    # set up the working/design variables
    prob.set_val("spacing_primary", 7.0)
    prob.set_val("spacing_secondary", 7.0)
    prob.set_val("angle_orientation", 0.0)
    prob.set_val("angle_skew", 0.0)

    # run the optimization driver
    prob.run_driver()

# get and print the AEP
AEP_val = float(prob.get_val("AEP_farm", units="GW*h")[0])
CapEx_val = float(prob.get_val("tcc.tcc", units="MUSD")[0])
BOS_val = float(prob.get_val("orbit.installation_capex", units="MUSD")[0])
OpEx_val = float(prob.get_val("opex.opex", units="MUSD/yr")[0])
LCOE_val = float(prob.get_val("financese.lcoe", units="USD/MW/h")[0])

print(f"AEP: {AEP_val:.2f} GWh")
print(f"ICC: ${CapEx_val+BOS_val:.2f}M")
print(f"\tCapEx: ${CapEx_val:.2f}M")
print(f"\tBOS: ${BOS_val:.2f}M")
print(f"OpEx/yr.: ${OpEx_val:.2f}M")
print(f"LCOE: ${LCOE_val:.2f}/MWh")
