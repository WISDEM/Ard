# Installation instructions
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
mamba install wisdem=3.18.1 -y
pip install -e .[dev,docs]
```

To test the installation, from the `Ard` folder run unit and regression tests:
```shell
source test/run_local_test_unit.sh
source test/run_local_test_system.sh
```

For user information, in pre-release, we are using some co-developed changes to the `FLORIS` library.

If the installation fails, please open a new issue [here](https://github.com/WISDEM/Ard/issues).
