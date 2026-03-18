from collections import namedtuple
from typing import Callable, List

import numpy as np
import numpy.typing as npt
import pandas as pd

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

Coordinate = namedtuple("Coordinate", ["x", "y"])


def observations(
    coordinates: List[Coordinate],
    times: npt.NDArray[np.int_],
    field: npt.NDArray[np.float64],
    error: Callable,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate synthetic observations by adding noise to true field-values.

    Parameters
    ----------
    error: Callable
        Function that takes a single argument (the true field value) and returns
        a value to be used as the standard deviation of the noise.
    rng: numpy.random.Generator
        Random number generator used to draw the observation noise.
    """
    d = pd.DataFrame(
        {
            "k": pd.Series(dtype=int),
            "x": pd.Series(dtype=int),
            "y": pd.Series(dtype=int),
            "value": pd.Series(dtype=float),
            "sd": pd.Series(dtype=float),
        }
    )

    # Create dataframe with observations and necessary meta data.
    for coordinate in coordinates:
        for k in times:
            value = field[k, coordinate.x, coordinate.y]
            sd = error(value)
            _df = pd.DataFrame(
                {
                    "k": [k],
                    "x": [coordinate.x],
                    "y": [coordinate.y],
                    "value": [value + rng.normal(loc=0.0, scale=sd)],
                    "sd": [sd],
                }
            )
            d = pd.concat([d, _df])
    d = d.set_index(["k", "x", "y"], verify_integrity=True)

    return d


def plot_responses(
    time_steps: npt.NDArray[np.int_],
    observations: pd.DataFrame,
    responses_ensemble: pd.DataFrame,
    response_truth: npt.NDArray[np.float64],
    responses_ensemble_post: pd.DataFrame = None,
):
    x_levels = observations.index.get_level_values("x").to_list()
    y_levels = observations.index.get_level_values("y").to_list()

    obs_coordinates = set(zip(x_levels, y_levels))

    for x, y in obs_coordinates:
        fig, ax = plt.subplots()
        ax.set_title(f"Sensor readings at coordinate {x, y}")
        ax.set_ylabel("Temperature")
        ax.set_xlabel("Time step $k$")
        ax.grid()

        df_single_sensor = observations.query(f"x=={x} & y=={y}")

        y_sensor = df_single_sensor["value"]
        yerr_sensor = df_single_sensor["sd"]

        ax.errorbar(
            time_steps,
            y_sensor,
            yerr_sensor,
            fmt="o",
            color="red",
            ecolor="red",
            markersize=3,
            capsize=10,
            elinewidth=1,
            markeredgewidth=1,
        )

        ax.plot(
            time_steps,
            responses_ensemble.query(f"x=={x} & y=={y}"),
            color="gray",
            alpha=0.2,
            label="_nolegend_",
        )
        ax.plot([], [], "gray", label="Prior responses")

        ax.plot(
            time_steps, response_truth[time_steps, x, y], color="black", label="Truth"
        )

        if responses_ensemble_post is not None:
            ax.plot(
                time_steps,
                responses_ensemble_post.query(f"x=={x} & y=={y}"),
                color="orange",
                alpha=0.2,
            )
            ax.plot([], [], "orange", label="Posterior responses")

        ax.legend(loc="best")

        fig.tight_layout()


def colorbar(mappable):
    # https://joseph-long.com/writing/colorbars/
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar
