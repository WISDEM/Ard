import pytest
from pathlib import Path
from ard.api import set_up_ard_model, replace_key_value, set_up_system_recursive
import numpy as np


@pytest.mark.usefixtures("subtests")
class TestSetUpArdModelOnshore:
    def setup_method(self):

        input_dict_path = Path(__file__).parent.absolute() / "inputs_onshore" / "ard_system.yaml"

        self.prob = set_up_ard_model(input_dict=str(input_dict_path), root_data_path=input_dict_path.parent)

        self.prob.run_model()

    def test_onshore_default_system_aep(self, subtests):
        with subtests.test("AEP_farm"):
            assert self.prob.get_val("AEP_farm", units="GW*h")[0] == pytest.approx(
                340.823649
            )
        with subtests.test("tcc.tcc"):
            assert self.prob.get_val("tcc.tcc", units="MUSD")[0] == pytest.approx(
                109.525
            )
        with subtests.test("BOS capex (landbosse.bos_capex)"):
            assert self.prob.get_val("bos_capex_kW", units="MUSD/GW")[
                0
            ] == pytest.approx(391.511636)
        with subtests.test("BOS capex (landbosse.total_capex)"):
            assert self.prob.get_val("total_capex", units="MUSD")[0] == pytest.approx(
                41.57835529481917
            )
        with subtests.test("opex.opex"):
            assert self.prob.get_val("opex.opex", units="MUSD/yr")[0] == pytest.approx(
                3.707
            )
        with subtests.test("financese.lcoe"):
            assert self.prob.get_val("financese.lcoe", units="USD/MW/h")[
                0
            ] == pytest.approx(44.127664498311255)


class TestSetUpArdModelOffshoreMonopile:
    def setup_method(self):

        input_dict_path = str(
            Path(__file__).parent.absolute()
            / "inputs_offshore_monopile"
            / "ard_system.yaml"
        )

        self.prob = set_up_ard_model(
            input_dict=input_dict_path, root_data_path="inputs_offshore_monopile"
        )

        self.prob.run_model()

    def test_offshore_monopile_default_system(self, subtests):

        with subtests.test("AEP_farm"):
            assert self.prob.get_val("AEP_farm", units="GW*h")[0] == pytest.approx(
                1829.994304
            )
        with subtests.test("tcc.tcc"):
            assert self.prob.get_val("tcc.tcc", units="MUSD")[0] == pytest.approx(
                768.4437570425
            )
        with subtests.test("BOS capex (orbit.total_capex_kW)"):
            assert self.prob.get_val("orbit.total_capex_kW", units="MUSD/GW")[
                0
            ] == pytest.approx(2602.633100)
        with subtests.test("opex.opex"):
            assert self.prob.get_val("opex.opex", units="MUSD/yr")[0] == pytest.approx(
                60.5
            )
        with subtests.test("financese.lcoe"):
            assert self.prob.get_val("financese.lcoe", units="USD/MW/h")[
                0
            ] == pytest.approx(123.219999)


class TestSetUpArdModelOffshoreFloating:
    def setup_method(self):

        input_dict_path = str(
            Path(__file__).parent.absolute()
            / "inputs_offshore_floating"
            / "ard_system.yaml"
        )

        self.prob = set_up_ard_model(
            input_dict=input_dict_path, root_data_path="inputs_offshore_floating"
        )

        self.prob.run_model()

    def test_offshore_floating_default_system(self, subtests):

        with subtests.test("AEP_farm"):
            assert self.prob.get_val("AEP_farm", units="GW*h")[0] == pytest.approx(
                1829.994304
            )
        with subtests.test("tcc.tcc"):
            assert self.prob.get_val("tcc.tcc", units="MUSD")[0] == pytest.approx(
                768.4437570425
            )
        with subtests.test("BOS capex (orbit.total_capex_kW)"):
            assert self.prob.get_val("orbit.total_capex_kW", units="MUSD/GW")[
                0
            ] == pytest.approx(2692.512849)
        with subtests.test("opex.opex"):
            assert self.prob.get_val("opex.opex", units="MUSD/yr")[0] == pytest.approx(
                60.5
            )
        with subtests.test("financese.lcoe"):
            assert self.prob.get_val("financese.lcoe", units="USD/MW/h")[
                0
            ] == pytest.approx(125.245983)
