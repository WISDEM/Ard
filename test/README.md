
# Testing

In `ard/test` we aim to have a broad and comprehensive testing suite.
The goal is to divide tests into `unit`- and `system`- level tests, with corresponding subdirectories here to test, respectively, individual components and collections of components, in order to verify that the pieces are working as expected.

Two paradigms of testing are used herein:
1) gold-standard testing: using analytical or outside data to test to exactness
2) pyrite-standard testing: using recorded outputs to verify that outputs of a unit or system given consistent inputs have not changed significantly

The testing suite is run using the `pytest` package, with coverage testing.
Acceptance criteria for new code using our CI/CD process is 80% coverage for both unit and system-level testing.

For local testing purposes, from the `Ard` root directory, one can run
```shell
source test/run_local_test_unit.sh
```
for the unit tests and the following for the system tests:
```shell
source test/run_local_test_system.sh
```
