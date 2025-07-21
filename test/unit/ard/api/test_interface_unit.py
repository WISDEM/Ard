import pytest
import numpy as np
import ard.layout.spacing
import openmdao.api as om

from ard.api import set_up_ard_model


class TestSetUpArdModel:
    def setup_method(self):
        pass

    def test_invalid_default_system(self):
        input_dict = {
            "system": "test",
            "modeling_options": {},
            "analysis_options": {},
        }

        with pytest.raises(
            ValueError,
            match=f"invalid default system 'test' specified. Must be one of \\['onshore', 'onshore_no_cable_design', 'offshore_monopile', 'offshore_monopile_no_cable_design', 'offshore_floating', 'offshore_floating_no_cable_design'\\]",
        ):
            set_up_ard_model(input_dict)
