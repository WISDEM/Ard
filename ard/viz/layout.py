import numpy as np
import matplotlib.axes
import matplotlib.pyplot as plt

import openmdao

import optiwindnet.plotting


# get plot limits based on the farm boundaries
def get_limits(
    windIOdict : dict,
    lim_buffer : float = 0.05,
):
    """
    generate plot limits based on the expected values found in the windIO file

    Parameters
    ----------
    windIOplant : dict
        a full, presumed validated, windIO plant specification file
    lim_buffer : float, optional
        a percent buffer for plot edges, by default 0.05 (5%)

    Returns
    -------
    x_lim : np.ndarray
        the two-valued limits for the x-axis based on the windIO
    y_lim : np.ndarray
        the two-valued limits for the y-axis based on the windIO
    """

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
    ard_prob : openmdao.api.Problem,
    input_dict : dict,
    ax : matplotlib.axes.Axes = None,
    show_image : bool = False,
    save_path : bool = None,
    save_kwargs : bool = {},
    include_cable_routing : bool = False,
):
    """
    plot the layout of a farm

    Parameters
    ----------
    ard_prob : openmdao.api.Problem
        the active Ard/OpenMDAO problem
    input_dict : dict
        the active Ard input dictionary
    ax : matplotlib.axes.Axes, optional
        an already-active pyplot Axes, by default None
    show_image : bool, optional
        to show the image, rather than just saving, by default False
    save_path : bool, optional
        should the image be saved, by default None
    save_kwargs : bool, optional
        optional keyword arguments for plt.savefig, by default {}
    include_cable_routing : bool, optional
        should the collection system routing be plotted also, by default False

    Returns
    -------
    matplotlib.axes.Axes
        the matplotlib Axes that have been generated (or modified)
    """

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
