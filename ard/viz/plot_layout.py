import numpy as np
import matplotlib.pyplot as plt

import openmdao.api as om


class OutputLayout(om.ExplicitComponent):

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        # load modeling options
        modeling_options = self.modeling_options = self.options["modeling_options"]
        self.N_turbines = modeling_options["farm"]["N_turbines"]

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

    def compute(self, inputs, outputs):

        fig, ax = plt.subplots()

        # plot the turbine locations
        ax.scatter(inputs["x_turbines"]/1.0e3, inputs["y_turbines"]/1.0e3, label="turbine")
        # add the turbine identity onto the plot
        for idx, (x, y) in enumerate(zip(inputs["x_turbines"]/1.0e3, inputs["y_turbines"]/1.0e3)):
            ax.text(x, y, str(idx), alpha=0.5, ha="right", va="bottom")
        ax.axis("equal")

        ax.set_xlabel("relative easting, $x$ (km)")
        ax.set_ylabel("relative northing, $y$ (km)")

        # plt.show()
