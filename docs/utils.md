# Explanation of `utils`

The `utils` sub-module of Ard contains the many generic capabilities needed for effective wind farm design and optimization in order to facilitate maximum code reuse. However, utils includes only generalized functions and methods and excludes most code that is technology-specific (see IO). As the capabilities in `utils` expand, some of them could eventually be broken off into a separate package. The utils are divided into the following categories:

## Core
The capabilities housed in `core.py` are generic and short enough that having a discipline-specific file is not warranted.

## Geometry
The capabilities housed in `geometry.py` are generic geometry-related code. They calculate distances, define shapes, and generally simplify working with geometric tasks in a continuously differentiable way in Ard.

## IO
The capabilities in `io.py` provide for loading and saving information and may be technology specific.

## Mathematics
The capabiliteis housed in `mathematics.py` are pure mathematical functions generally formulated to be continuously differentiable. These functions are useful in a range of applications within Ard and design optimization generally.

## Test Utils
The capabilities housed in `test_utils.py` are specifically for use in testing and not in other parts of Ard.