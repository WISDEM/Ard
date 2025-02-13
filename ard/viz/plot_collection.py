import numpy as np
import matplotlib.pyplot as plt

import openmdao.api as om


class OutputCollection(om.ExplicitComponent):

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        # load modeling options
        modeling_options = self.modeling_options = self.options["modeling_options"]
        self.N_turbines = modeling_options["farm"]["N_turbines"]
        self.N_substations = modeling_options["farm"]["N_substations"]

        # add inputs
        self.add_input(
            "x_turbines",
            np.zeros((self.N_turbines,)),
            units="m",
            desc="turbine location in x-direction",
        )
        self.add_input(
            "y_turbines",
            np.zeros((self.N_turbines,)),
            units="m",
            desc="turbine location in y-direction",
        )
        self.add_input(
            "x_substations",
            np.zeros((self.N_substations,)),
            units="m",
            desc="substation location in x-direction",
        )
        self.add_input(
            "y_substations",
            np.zeros((self.N_substations,)),
            units="m",
            desc="substation location in y-direction",
        )
        self.add_input("load_cables", np.zeros((self.N_turbines,)))

        self.add_discrete_input("edges", np.zeros((self.N_turbines, 2), int))


    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs=None):

        fig, ax = plt.subplots()

        cable_color = "w"  # 1.0 - np.array(fig.get_facecolor())

        # plot the cables
        for idx_edge, (e0, e1) in enumerate(discrete_inputs["edges"]):
            x0 = (
                inputs["x_substations"][self.N_substations + e0]
                if e0 < 0
                else inputs["x_turbines"][e0]
            )/1.0e3
            y0 = (
                inputs["y_substations"][self.N_substations + e0]
                if e0 < 0
                else inputs["y_turbines"][e0]
            )/1.0e3
            x1 = inputs["x_turbines"][e1]/1.0e3
            y1 = inputs["y_turbines"][e1]/1.0e3
            load = inputs["load_cables"][idx_edge]
            plt.plot(
                [x0, x1], [y0, y1], "-",
                c=cable_color, alpha=0.5, zorder=-100, label="_cable",
            )
            plt.text(
                0.5 * (x0 + x1),
                0.5 * (y0 + y1),
                f"{int(load)}",
                c=cable_color,
                horizontalalignment="center",
                verticalalignment="center",
            )

        # plot the turbine locations
        ax.scatter(
            inputs["x_turbines"]/1.0e3,
            inputs["y_turbines"]/1.0e3,
            label="turbine",
        )
        ax.axis("equal")

        # plot the substation locations
        ax.scatter(
            inputs["x_substations"]/1.0e3,
            inputs["y_substations"]/1.0e3,
            label="substation",
        )

        # dummy legend entry for the cables
        ax.plot([], [], "-", alpha=0.5, c=cable_color, label="cable")

        ax.set_xlabel("relative easting, $x$ (km)")
        ax.set_ylabel("relative northing, $y$ (km)")
        ax.legend()

        # plt.show()
