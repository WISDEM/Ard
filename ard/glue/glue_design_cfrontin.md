
# Proposed production glue code design

## Fundamental types

- terminal: no children
  - "component"?
    - a single, explicit $y = f(x; \theta)$ design component
      - i.e. active design variables $x$ and latent variables/parameters $\theta$
    - have flag to trigger finite differencing of derivatives if applicable
    - nominate active inputs and active outputs for promotion
    - design example:
      ```yaml
      - name: aerodynamics
        type: component
        finite_difference: false
        inputs_active:
        - "x_turbine"
        - "y_turbine"
        outputs_active:
        - "thrust_cases"
        - "power_cases"
      ```
  - "connection"?
    - the connection that takes an output $y$ to an input $x$
    - take a component and output name and pipe it to a target component input variable
    - for explicit connections: should all connections be explicit?
      - i.e.: no promotion; alternatively, use promotion
    - design example:
      ```yaml
      - component_origin: aerodynamics
        quantity_origin: aep
        component_destination: lcoe
        quantity_destination: AEP
      ```
- non-terminal: can have children
  - groups
    - design intent:
      - group few-input, few-output, many-internal-connection components to finite difference few-to-few and avoid excessive finite differencing
      - potentially a unit for surrogate modeling as well
    - have flag to trigger finite differencing of derivatives, if applicable
    - should be recursed by the code?
      - maybe? we could also restrict to one top-level group
    - design example:
      ```yaml
      - name: layout2aep
        type: group
        finite_difference: true
        inputs_active:
        - "theta0_layout"
        - "theta1_layout"
        outputs_active:
        - "aep"
        children:
        - name: child1
          type: component
      ```

## Demo example

```yaml
name: layout2lcoe
structure:
- name: layout2aep
  type: group
  finite_difference: true
  inputs_active:
  - "theta0_layout"
  - "theta1_layout"
  outputs_active:
  - "aep"
  children:
  - name: layout
    type: component
    finite_difference: false
    inputs_active:
    - "theta0_layout"
    - "theta1_layout"
    outputs_active:
    - "x_turbine"
    - "y_turbine"
  - name: aerodynamics
    type: component
    finite_difference: false
    inputs_active:
    - "x_turbine"
    - "y_turbine"
    outputs_active:
    - "thrust_cases"
    - "power_cases"
  - name: integrator
    type: component
    finite_difference: false
    inputs_active:
    - "power_cases"
    # outputs_active:
    # - "aep"
- name: aep2lcoe
  type: component
  finite_difference: false
  # inputs_active:
  # - "aep"
  outputs_active:
  - "lcoe"
connections:
  - 1:
    - component_origin: layout2aep
    - quantity_origin: aep
    - component_target: aep2lcoe
    - quantity_target: aep
```

## Discussion

- what is "active" vs. "design variables" vs. "lighting up the jacobian"
- jacobian coloring dependencies on "active" variable definition
- we should probably mimic the OpenMDAO structure
- default case, validator, schema
- what's missing?
  - the modeling options for the models
  - optimization control: driver, design variables and constraints, etc.
    - should these all be together?
  - windIO: bring in, figure out what we need
