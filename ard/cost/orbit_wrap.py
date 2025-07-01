from wisdem.orbit.api import wisdem as orbit_wisdem

from wisdem.orbit import ProjectManager

class ORBITDetail(orbit_wisdem.Orbit):
    def setup(self):
        """Define all input variables from all models."""
        self.set_input_defaults("wtiv", "example_wtiv")
        self.set_input_defaults("feeder", "example_feeder")
        #self.set_input_defaults("num_feeders", 1)
        #self.set_input_defaults("num_towing", 1)
        #self.set_input_defaults("num_station_keeping", 3)
        #self.set_input_defaults(
        #    "oss_install_vessel", "example_heavy_lift_vessel",
        #)
        self.set_input_defaults("site_distance", 40.0, units="km")
        self.set_input_defaults("site_distance_to_landfall", 40.0, units="km")
        self.set_input_defaults("interconnection_distance", 40.0, units="km")
        self.set_input_defaults("plant_turbine_spacing", 7)
        self.set_input_defaults("plant_row_spacing", 7)
        self.set_input_defaults("plant_substation_distance", 1, units="km")
        #self.set_input_defaults("num_port_cranes", 1)
        #self.set_input_defaults("num_assembly_lines", 1)
        self.set_input_defaults("takt_time", 170.0, units="h")
        self.set_input_defaults("port_cost_per_month", 2e6, units="USD/mo")
        self.set_input_defaults("construction_insurance", 44.0, units="USD/kW")
        self.set_input_defaults("construction_financing", 183.0, units="USD/kW")
        self.set_input_defaults("contingency", 316.0, units="USD/kW")
        self.set_input_defaults("commissioning_cost_kW", 44.0, units="USD/kW")
        self.set_input_defaults("decommissioning_cost_kW", 58.0, units="USD/kW")
        self.set_input_defaults("site_auction_price", 100e6, units="USD")
        self.set_input_defaults("site_assessment_cost", 50e6, units="USD")
        self.set_input_defaults("construction_plan_cost", 1e6, units="USD")
        self.set_input_defaults("installation_plan_cost", 2.5e5, units="USD")
        self.set_input_defaults("boem_review_cost", 0.0, units="USD")

        self.add_subsystem(
            "orbit",
            ORBITWisdemDetail(
                floating=self.options["floating"],
                jacket=self.options["jacket"],
                jacket_legs=self.options["jacket_legs"],
            ),
            promotes=["*"],
        )


class ORBITWisdemDetail(orbit_wisdem.OrbitWisdem):
    """ORBIT-WISDEM Fixed Substructure API, modified for detailed layouts"""

    def initialize(self):
        super().initialize()

    def setup(self):
        super().setup()

    def compile_orbit_config_file(
        self,
        inputs,
        outputs,
        discrete_inputs,
        discrete_outputs,
    ):

        config = super().compile_orbit_config_file(
            inputs, outputs, discrete_inputs, discrete_outputs,
        )  # run the superclass

        # remove the grid plant option, and replace with a custom plant
        del config["plant"]
        config["plant"] = {
            "layout": "custom",
            "num_turbines": int(discrete_inputs["number_of_turbines"]),
        }

        # switch to the custom array system design
        if not ("ArraySystemDesign" in config["design_phases"]):
            raise KeyError("I assumed that 'ArraySystemDesign' would be in the config. Something changed.")
        config["design_phases"][config["design_phases"].index("ArraySystemDesign")] = "CustomArraySystemDesign"

        raise NotImplementedError("need to copy and move the default library to a local version!")

        raise NotImplementedError("need to create a turbine location csv in the library!")

        # add a turbine location csv on the config
        filename = "wisdem_detailed_array"
        config["array_system_design"]["location_data"] = filename

        self._orbit_config = config  # reinstall- probably not needed due to reference
        return config  # and return

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        config = self.compile_orbit_config_file(inputs, outputs, discrete_inputs, discrete_outputs)

        debug = False
        if debug:
            import json
            fname = "orbit_dump.json"
            with open(fname, "w") as f:
                f.write(json.dumps(config))
            with open(fname) as f:
                config = json.loads(f.read())

        project = ProjectManager(config)
        project.run()
        print(f"DEBUG!!!!! project location: {project}")

        # The ORBIT version of total_capex includes turbine capex, so we do our own sum of
        # the parts here that wisdem doesn't account for
        capacity_kW = 1e3 * inputs["turbine_rating"] * discrete_inputs["number_of_turbines"]
        outputs["bos_capex"] = project.bos_capex
        outputs["soft_capex"] = project.soft_capex
        outputs["project_capex"] = project.project_capex
        outputs["total_capex"] = project.bos_capex + project.soft_capex + project.project_capex
        outputs["total_capex_kW"] = outputs["total_capex"] / capacity_kW
        outputs["installation_time"] = project.installation_time
        outputs["installation_capex"] = project.installation_capex
