
# Component types

The design intention of `Ard` is to offer a principled, modular, extensible wind farm layout optimization tool.

In order to balance focus with the modularity and extensibility intended for the code, we will classify different types of components, and build "base `Ard`" with a set of default components.
Each type of component will be defined below, and each type of component will have a template parent class which can be used to derive custom user-defined components to serve as drop in replacements in `Ard`.

## Layout DV Components (`layout`)

Wind farm layout optimization is a significantly challenging problem for global optimization, due to the existence of many local minima.
One strategy for reducing the dimensionality of the design space is the use of layout models.
`layout` components are for connecting some reduced layout variables to (`x_turbines`, `y_turbines`) variables that explicitly describe the layout of a farm for computing the farm aerodynamics.

The default `layout` component is the grid farm layout defined in `ard/layout/gridfarm.py`, which parameterizes a structured parallelogram-shaped farm in terms of a primary axis, aligned with respect to North by an orientation angle coincident with compass rose angles, with rows spaced along this axis by a constant spacing measured in rotor diameters; and a secondary axis, skewed from orthagonal by a skew angle in the direction of compass angles, with rows spaced along this axis by a constant spacing.
This results in four parameters, two nondimensional spacing values and two angles.

**tl;dr:** `layout` components map from a simplified parameter set to Cartesian farm coordinates

## Farm Aero Components (`farm_aero`)

Fundamentally, `farm_aero` components will take in a set of farm layout design variables, in terms of `x_turbines` and `y_turbines` components of turbine locations, and potentially with some control command input, namely `yaw_turbines`.

In addition to these design variables, the turbine definitions to be used and some (possibly comprehensive) set of wind conditions to be queried will also be provided to a given `farm_aero` component.

The result of a `farm_aero` component will be a power or energy production quantity of interest.
Typically, these will be a power output estimate for the set of provided conditions or annual energy production estimate for the farm given the wind resource.

The default `farm_aero` component is [NREL's FLORIS tool](https://nrel.github.io/floris), which can be found at `ard/farm_aero/floris.py`.
We wrap FLORIS to map from `x_turbines`, `y_turbines`, and `yaw_turbines` to turbine powers, turbine thrusts, farm power, and, optionally, farm AEP.

**tl;dr:** `farm_aero` components map from a farm layouts and possibly control settings to some measure of farm power production

## Economics and Finance Components (`cost`)

The `cost` components are less formally structured than the components above, but generally take inputs that are consumable stocks or marketable products of a wind system.
Meanwhile, they give as outputs some measure of the monetary trade value that the wind system can create.
Complete details are provided in the components in the `ard/cost` folder.

The default `cost` component set are the WISDEM tools given in `ard/cost/windse_wrap.py`, which include WISDEM's `LandBOSSE` module for BOS calculation and WISDEM's `PlantFinance` module for computation of LCOE.

**tl;dr:** `cost` components map from machines and their production of energy to money inputs and outputs


<!-- FIN! -->
