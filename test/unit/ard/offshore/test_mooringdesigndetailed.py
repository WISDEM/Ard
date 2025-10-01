# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 10:09:25 2025

@author: elozon
"""

import pytest
import numpy as np
import openmdao.api as om
from pathlib import Path
import ard
from famodel.helpers import adjustMooring


class TestMooringDesignDetailed:
    def setup_method(self):

        self.D_rotor = 240.0

        # set turbine layout (3x3 grid 5D spacing)
        X, Y = [
            7.0 * self.D_rotor * v
            for v in np.meshgrid(np.arange(0, 3), np.arange(0, 3))
        ]

        self.x_turbines = X.flatten()
        self.y_turbines = Y.flatten()

        self.N_turbines = len(self.x_turbines)

        self.modeling_options = {
            "layout": {
                "N_turbines": self.N_turbines,
            },
            "offshore": True,
            "floating": True,
            "platform": {
                "N_anchors": 3,
                "min_mooring_line_length_m": 500.0,
                "N_anchor_dimensions": 2,
            },
            "site_depth": 200.0,
            "collection": {
                "max_turbines_per_string": 8,
                "solver_name": "appsi_highs",
                "solver_options": dict(
                    time_limit=60,
                    mip_rel_gap=0.005,  # TODO ???
                ),
            },
            "mooring_setup": {
                "site_conds": {
                    "general": {},
                    "bathymetry": {
                        "file": Path(__file__).parent.absolute()
                        / "inputs"
                        / "GulfOfMaine_bathymetry_100x99.txt"
                    },
                },
                "mooring_info": (
                    Path(__file__).parent.absolute()
                    / "inputs"
                    / "OntologySample200m.yaml"
                ),
                "adjuster_settings": {
                    "adjuster": adjustMooring,
                    "method": "horizontal",
                    "i_line": 1,
                },
            },
        }

    def test_FAModel_turbine_positions(self):

        # set up openmdao problem
        model = om.Group()
        model.add_subsystem(  # mooring system design
            "mooring_design",
            ard.offshore.mooring_design_detailed.DetailedMooringDesign(
                modeling_options=self.modeling_options,
                wind_query=None,
                data_path=Path(__file__).parent.absolute(),
            ),
            promotes_inputs=["x_turbines", "y_turbines"],
        )

        prob = om.Problem(model)
        prob.setup()

        # set up x and y turbine positions
        prob.set_val("x_turbines", self.x_turbines)
        prob.set_val("y_turbines", self.y_turbines)

        prob.run_model()

        # check that mooring_design turbine positions match the inputs
        assert np.all(
            np.isclose(prob.get_val("mooring_design.x_turbines"), self.x_turbines)
        )
        assert np.all(
            np.isclose(prob.get_val("mooring_design.y_turbines"), self.y_turbines)
        )

    def test_FAModel_anchor_positions(self):

        # change number of turbines to one
        self.modeling_options["layout"]["N_turbines"] = 1

        # set up openmdao problem
        model = om.Group()
        model.add_subsystem(  # mooring system design
            "mooring_design",
            ard.offshore.mooring_design_detailed.DetailedMooringDesign(
                modeling_options=self.modeling_options,
                wind_query=None,
                data_path=Path("inputs").absolute(),
            ),
            promotes_inputs=["x_turbines", "y_turbines"],
        )

        prob = om.Problem(model)
        prob.setup()

        # set up x and y turbine positions to (1 km, 1 km)
        prob.set_val("x_turbines", [1])
        prob.set_val("y_turbines", [1])

        prob.run_model()

        # calculate anchor positions in km
        x_anchors = [
            1 + 0.7 * np.cos(60 / 180 * np.pi),
            1 + 0.7 * np.cos(60 / 180 * np.pi),
            1 - 0.7,
        ]
        y_anchors = [
            1 - 0.7 * np.sin(60 / 180 * np.pi),
            1 + 0.7 * np.sin(60 / 180 * np.pi),
            1,
        ]

        # check that mooring_design anchor positions match expected
        assert np.all(np.isclose(prob.get_val("mooring_design.x_anchors"), x_anchors))
        assert np.all(np.isclose(prob.get_val("mooring_design.y_anchors"), y_anchors))
