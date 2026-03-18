"""Partial Differential Equations to use as forward models."""

from typing import Optional

import numpy as np
import numpy.typing as npt
import jax
import jax.numpy as jnp


def heat_equation(
    u0: npt.NDArray[np.float64],
    alpha: npt.NDArray[np.float64],
    dx: float,
    dt: float,
    num_steps: int,
    rng: np.random.Generator,
    scale: Optional[float] = None,
) -> npt.NDArray[np.float64]:
    """2D heat equation that supports field of heat coefficients.

    Based on:
    https://levelup.gitconnected.com/solving-2d-heat-equation-numerically-using-python-3334004aa01a
    """
    ny, nx = u0.shape
    u = np.zeros((num_steps + 1, ny, nx))
    u[0] = u0
    gamma = (alpha * dt) / (dx**2)

    for k in range(num_steps):
        # Vectorized finite difference
        u[k + 1, 1:-1, 1:-1] = (
            gamma[1:-1, 1:-1]
            * (
                u[k, 2:, 1:-1]  # i+1
                + u[k, :-2, 1:-1]  # i-1
                + u[k, 1:-1, 2:]  # j+1
                + u[k, 1:-1, :-2]  # j-1
                - 4 * u[k, 1:-1, 1:-1]
            )
            + u[k, 1:-1, 1:-1]
        )

        # Add noise if needed
        if scale is not None:
            noise = rng.normal(0, scale, size=(ny - 2, nx - 2))
            u[k + 1, 1:-1, 1:-1] += noise

    return u


def heat_equation_jax(
    u0: "jax.Array",
    alpha: "jax.Array",
    dx: float,
    dt: float,
    num_steps: int,
    key: Optional["jax.Array"] = None,
    scale: Optional[float] = None,
) -> "jax.Array":
    """2D heat equation using JAX for automatic differentiation.

    Differentiable with respect to `u0` and `alpha`, making it suitable for
    gradient-based inference methods such as variational inference.

    Parameters
    ----------
    u0 : jax.Array, shape (ny, nx)
        Initial temperature field.
    alpha : jax.Array, shape (ny, nx)
        Spatially varying heat transfer coefficient field.
    dx : float
        Grid spacing.
    dt : float
        Time step size.
    num_steps : int
        Number of time steps to simulate.
    key : jax.Array, optional
        JAX random key for noise generation. Required if `scale` is not None.
    scale : float, optional
        Standard deviation of Gaussian noise added at each time step.

    Returns
    -------
    jax.Array, shape (num_steps + 1, ny, nx)
        Temperature field at each time step.
    """
    gamma = (alpha * dt) / (dx**2)

    def step_fn(carry, _):
        u_prev, rng_key = carry

        interior = (
            gamma[1:-1, 1:-1]
            * (
                u_prev[2:, 1:-1]
                + u_prev[:-2, 1:-1]
                + u_prev[1:-1, 2:]
                + u_prev[1:-1, :-2]
                - 4 * u_prev[1:-1, 1:-1]
            )
            + u_prev[1:-1, 1:-1]
        )

        if scale is not None and key is not None:
            rng_key, subkey = jax.random.split(rng_key)
            noise = jax.random.normal(subkey, shape=interior.shape) * scale
            interior = interior + noise

        u_next = u_prev.at[1:-1, 1:-1].set(interior)
        return (u_next, rng_key), u_next

    init_key = key if key is not None else jax.random.PRNGKey(0)
    # scan calls step_fn num_steps times, threading (u, key) as carry from each call to the next
    (_, _), u_all = jax.lax.scan(step_fn, (u0, init_key), jnp.arange(num_steps))

    # Prepend initial condition
    return jnp.concatenate([u0[None, ...], u_all], axis=0)
