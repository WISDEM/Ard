import numpy as np
import matplotlib.pyplot as plt
import optiwindnet.plotting


# get plot limits based on the farm boundaries
def get_limits(windIOdict, lim_buffer=0.05):
    x_lim = [
        np.min(windIOdict["site"]["boundaries"]["polygons"][0]["x"])
        - lim_buffer * np.ptp(windIOdict["site"]["boundaries"]["polygons"][0]["x"]),
        np.max(windIOdict["site"]["boundaries"]["polygons"][0]["x"])
        + lim_buffer * np.ptp(windIOdict["site"]["boundaries"]["polygons"][0]["x"]),
    ]
    y_lim = [
        np.min(windIOdict["site"]["boundaries"]["polygons"][0]["y"])
        - lim_buffer * np.ptp(windIOdict["site"]["boundaries"]["polygons"][0]["y"]),
        np.max(windIOdict["site"]["boundaries"]["polygons"][0]["y"])
        + lim_buffer * np.ptp(windIOdict["site"]["boundaries"]["polygons"][0]["y"]),
    ]
    return x_lim, y_lim


def plot_layout(
    ard_prob,
    input_dict,
    ax=None,
    show_image=False,
    save_path=None,
    save_kwargs={},
    include_cable_routing=False,
):

    # get the turbine locations to plot
    x_turbines = ard_prob.get_val("x_turbines", units="m")
    y_turbines = ard_prob.get_val("y_turbines", units="m")

    # make axis object
    if ax is None:
        fig, ax = plt.subplots()

    # plot wind plant boundaries
    windIO_dict = input_dict["modeling_options"]["windIO_plant"]

    ax.fill(
        [x * 1e3 for x in windIO_dict["site"]["boundaries"]["polygons"][0]["x"]],
        [y * 1e3 for y in windIO_dict["site"]["boundaries"]["polygons"][0]["y"]],
        linestyle="--",
        alpha=0.5,
        fill=False,
        c="k",
        # linecolor="k",
    )

    # plot turbines
    ax.plot(x_turbines, y_turbines, "ok")

    # adjust plot limits
    x_lim, y_lim = get_limits(windIO_dict)
    ax.set_xlim([x * 1e3 for x in x_lim])
    ax.set_ylim([y * 1e3 for y in y_lim])

    if include_cable_routing:
        optiwindnet.plotting.gplot(
            ard_prob.model.collection.graph,
            ax=ax,
            dark=False,
            legend=False,
            hide_ST=True,
            infobox=False,
            landscape=False,
        )

    # show, save, or return
    if save_path is not None:
        plt.savefig(save_path, save_kwargs)

    if show_image:
        plt.show()

    return ax
