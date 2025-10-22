from pathlib import Path

import openmdao.api as om

from ard.utils.io import load_yaml
from ard.api import set_up_ard_model


class TestMultiobjectiveSetUp:
    def setup_method(self):

        # create the simplest system that will compile
        input_dict = load_yaml(Path(__file__).parent / "inputs_onshore" / "ard_system_moo.yaml")
        input_dict["system"] = {
            "type": "group",
            "systems": {
                "layout": {
                    "type": "component",
                    "module": "ard.layout.gridfarm",
                    "object": "GridFarmLayout",
                    "promotes": ["*"],
                    "kwargs": {
                        "modeling_options": input_dict["modeling_options"],
                    },
                },
                "boundary": {
                    "type": "component",
                    "module": "ard.layout.boundary",
                    "object": "FarmBoundaryDistancePolygon",
                    "promotes": ["*"],
                    "kwargs": {
                        "modeling_options": input_dict["modeling_options"],
                    },
                },
                "aepFLORIS": {
                    "type": "component",
                    "module": "ard.farm_aero.floris",
                    "object": "FLORISAEP",
                    "promotes": ["x_turbines", "y_turbines", "AEP_farm"],
                    "kwargs": {
                        "modeling_options": input_dict["modeling_options"],
                        # "data_path":
                        "case_title": "default"
                    },
                },
            },
        }
        self.da_plough = set_up_ard_model(
            input_dict=input_dict,
        )

    def test_dummy(self):

        self.da_plough.run_driver()

        assert True
