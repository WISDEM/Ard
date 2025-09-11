
# Ard

[![CI/CD test suite](https://github.com/WISDEM/Ard/actions/workflows/python-tests-consolidated.yaml/badge.svg?branch=develop)](https://github.com/WISDEM/Ard/actions/workflows/python-tests-consolidated.yaml)

![Ard logo](assets/logomaker/logo.png)

**Dig into wind farm design.**

<!-- The (aspirationally) foolproof tool for preparing wind farm layouts. -->

[An ard is a type of simple and lightweight plow](https://en.wikipedia.org/wiki/Ard_\(plough\)), used through the single-digit centuries to prepare a farm for planting.
The intent of `Ard` is to be a modular, full-stack multi-disciplinary optimization tool for wind farms.

Wind farms are complicated, multi-disciplinary systems.
They are aerodynamic machines (composed of complicated control systems, power electronic devices, etc.), social and political objects, generators of electrical power and consumers of electricl demand, and the core value generator (and cost) of complicated financial instruments.
Moreover, the design of any *one* of these aspects affects all the rest!

`Ard` is a platform for wind farm layout optimization that seeks to enable plant-level design choices that can incorporate these different aspects _and their interactions_ to make wind energy projects more successful.
In brief, we are designing Ard to be: principled, modular, extensible, and effective, to allow resource-specific wind farm layout optimization with realistic, well-posed constraints, wholistic and complex objectives, and natural incorporation of multiple fidelities and disciplines.

## Documentation
Ard documentation is available at [https://wisdem.github.io/Ard/]()

## Installation instructions

<!-- `Ard` can be installed locally from the source code with `pip` or through a package manager from PyPI with `pip` or conda-forge with `conda`. -->
<!-- For Windows systems, `conda` is required due to constraints in the WISDEM installation system. -->
<!-- For macOS and Linux, any option is available. -->
`Ard` is currently in pre-release and is only available as a source-code installation.
The source can be cloned from github using the following command in your preferred location:
```shell
git clone git@github.com:WISDEM/Ard.git
```
Once downloaded, you can enter the `Ard` root directory using
```shell
cd Ard
```

At this point, although not strictly required, we recommend creating a dedicated conda environment with `pip`, `python=3.12`, and `mamba` in it:
```shell
conda create --name ard-env
conda activate ard-env
conda install python=3.12 pip mamba -y
```

From here, installation can be handled by `pip`.

For a basic and static installation, type:
```shell
pip install .
```
For development (and really for everyone during pre-release), we recommend a full development installation:
```shell
pip install -e .[dev,docs]
```
which will install in "editable mode" (`-e`), such that changes made to the source will not require re-installation, and with additional optional packages for development and documentation (`[dev,docs]`).

There can be some hardware-software mis-specification issues with WISDEM installation from `pip` for MacOS 12 and 13 on machines with Apple Silicon.
In the event of issues, WISDEM can be installed manually or using `conda` without issues, then `pip` installation can proceed.

```shell
mamba install wisdem -y
pip install -e .[dev,docs]
```

## Testing instructions

The installation can be tested comprehensively using `pytest` from the top-level directory.
The developers also provide some convenience scripts for testing new installations; from the `Ard` folder run unit and regression tests:
```shell
source test/run_local_test_unit.sh
source test/run_local_test_system.sh
```
These enable the generation of HTML-based coverage reports by default and can be used to track "coverage", or the percentage of software lines of code that are run by the testing systems.
`Ard`'s git repository includes requirements for both the `main` and `develop` branches to have 80% coverage on unit testing and 50% testing in system testing, which are, respectively, tests of individual parts of `Ard` and "systems" composed of multiple parts.
Failures are not tolerated in code that is merged onto these branches and code found therein *should* never cause a testing failure if it has been incorporated.
If installation and testing fails, please open a new issue [here](https://github.com/WISDEM/Ard/issues).

## Design philosophy

The design of `Ard` is centered around two user cases:
1) systems energy researchers who are focusing on one specific subdiscipline (e.g. layout strategies, social impacts, or aerodynamic modeling) but want to be able to easily keep track of how it impacts the entire value chain down to production, cost, and/or value of energy or even optimize with respect to these, and
2) private industry researchers who are interested in how public-sector research results change when proprietary analysis tools are dropped in and/or coupled the other tools in a systems-level simulation.

`Ard` is being developed as a modular tool to enable these types of research queries.
This starts from our software development goals, which are that `Ard` should be:
1) principled:
   - fully documented
   - adhering to best-practices for code development
2) modular and extensible:
   - choose the parts you want
   - skip the ones you don't
   - build yourself the ones we don't have
3) effective
    - fully tested and testable at the unit and system level
    - built with a forward-looking approach
        - ready for derivatives and control of their approximation
        - built to incorporate multi-fidelity approximation
        - structured for extensibility to other advanced methods

This, then, allows us to attempt to accomplish the research goals of `Ard`, to:
1) allow optimization of wind farm layouts for specific wind resource profiles
2) enable the incorporation of realistic but well-posed constraints
3) target wholistic and complex system-level optimization objectives like LCOE and beyond-LCOE metrics
4) naturally incorporate analyses across fidelities to efficiently integrate advanced simulation

## Current capabilities

For the beta pre-release of `Ard`, we concentrate on optimization problems for wind plants, starting from structured layouts and optimizing it to minimize the levelized cost of energy, or LCOE.
This capability is demonstrated for a land-based (LB) wind farm in `examples/01_onshore` and tested in an abridged form in `test/system/ard/api/test_LCOE_LB_stack.py`.
In this example, the wind farm layout is parametrized with two angles, named orientation and skewed, and turbine distancing for rows and columns.
Additionally, we have offshore examples co-located with the onshore one.
In the beta pre-release stage, the constituent subcomponents of these problems are known to work and fully tested;
any capabilities that are not touched by these problems should be considered to be experimental.

These cases start from a four parameter farm layout, compute landuse area, make FLORIS AEP estimates, compute turbine capital costs, balance-of-station (BOS), and operational costs using WISDEM components, and finally give summary estimates of plant finance figures.
The components that achieve this can be assembled to either run a single top-down analysis run, or run an optimization.

# Contributing to `Ard`

We have striven towards gold-standard documentation and testing for `Ard`.
Contribution is welcome, and we are happy [to field pull requests from github](https://github.com/WISDEM/Ard/pulls).
For acceptance, PRs must:
- be formatted using [`black`](https://github.com/psf/black)
- not fail any unit tests
- achieve coverage criteria for unit & system testing
- be documented enough for continued maintenance by core `Ard` developers

## Building Documentation

To build the documentation locally, run the following from the top-level `Ard/` directory:
```shell
jupyter-book build docs/
```
You can then open `Ard/docs/_build/html/index.html` to view the docs.

---

Released as open-source software by the National Renewable Energy Laboratory under NREL software record number SWR-25-18.

Copyright &copy; 2024, Alliance for Sustainable Energy, LLC.
