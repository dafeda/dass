"""Partial Differential Equations to use as forward models."""

from typing import Optional
import numpy as np
import numpy.typing as npt


def heat_equation(
    u0: npt.NDArray[np.float_],
    alpha: npt.NDArray[np.float_],
    dx: float,
    dt: float,
    num_steps: int,
    rng: np.random.Generator,
    scale: Optional[float] = None,
) -> npt.NDArray[np.float_]:
    """Heat equation that supports 2D and 3D with field of heat coefficients.

    For 2D: u0 shape is (ny, nx), alpha shape is (ny, nx)
    For 3D: u0 shape is (nz, ny, nx), alpha shape is (nz, ny, nx)

    Based on:
    https://levelup.gitconnected.com/solving-2d-heat-equation-numerically-using-python-3334004aa01a
    """
    if u0.ndim == 2:
        return _heat_equation_2d(u0, alpha, dx, dt, num_steps, rng, scale)
    elif u0.ndim == 3:
        return _heat_equation_3d(u0, alpha, dx, dt, num_steps, rng, scale)
    else:
        raise ValueError(f"Only 2D and 3D arrays supported, got {u0.ndim}D")


def _heat_equation_2d(
    u0: npt.NDArray[np.float_],
    alpha: npt.NDArray[np.float_],
    dx: float,
    dt: float,
    num_steps: int,
    rng: np.random.Generator,
    scale: Optional[float] = None,
) -> npt.NDArray[np.float_]:
    """2D heat equation that supports field of heat coefficients."""
    ny, nx = u0.shape
    u = np.zeros((num_steps + 1, ny, nx))
    u[0] = u0
    gamma = (alpha * dt) / (dx**2)

    for k in range(num_steps):
        # Finite difference stencil for 2D Laplacian
        # Uses 4 neighboring points (up, down, left, right)
        u[k + 1, 1:-1, 1:-1] = (
            gamma[1:-1, 1:-1]
            * (
                u[k, 2:, 1:-1]  # i+1 (down)
                + u[k, :-2, 1:-1]  # i-1 (up)
                + u[k, 1:-1, 2:]  # j+1 (right)
                + u[k, 1:-1, :-2]  # j-1 (left)
                - 4 * u[k, 1:-1, 1:-1]  # central point * 4 neighbors
            )
            + u[k, 1:-1, 1:-1]
        )
        # Add noise if needed
        if scale is not None:
            noise = rng.normal(0, scale, size=(ny - 2, nx - 2))
            u[k + 1, 1:-1, 1:-1] += noise

    return u


def _heat_equation_3d(
    u0: npt.NDArray[np.float_],
    alpha: npt.NDArray[np.float_],
    dx: float,
    dt: float,
    num_steps: int,
    rng: np.random.Generator,
    scale: Optional[float] = None,
) -> npt.NDArray[np.float_]:
    """3D heat equation that supports field of heat coefficients."""
    nz, ny, nx = u0.shape
    u = np.zeros((num_steps + 1, nz, ny, nx))
    u[0] = u0
    gamma = (alpha * dt) / (dx**2)

    for k in range(num_steps):
        # Finite difference stencil for 3D Laplacian
        # Uses 6 neighboring points (forward/backward in each x, y, z direction)
        u[k + 1, 1:-1, 1:-1, 1:-1] = (
            gamma[1:-1, 1:-1, 1:-1]
            * (
                u[k, 2:, 1:-1, 1:-1]  # k+1 (forward in z)
                + u[k, :-2, 1:-1, 1:-1]  # k-1 (backward in z)
                + u[k, 1:-1, 2:, 1:-1]  # i+1 (forward in y)
                + u[k, 1:-1, :-2, 1:-1]  # i-1 (backward in y)
                + u[k, 1:-1, 1:-1, 2:]  # j+1 (forward in x)
                + u[k, 1:-1, 1:-1, :-2]  # j-1 (backward in x)
                - 6 * u[k, 1:-1, 1:-1, 1:-1]  # central point * 6 neighbors
            )
            + u[k, 1:-1, 1:-1, 1:-1]
        )
        # Add noise if needed
        if scale is not None:
            noise = rng.normal(0, scale, size=(nz - 2, ny - 2, nx - 2))
            u[k + 1, 1:-1, 1:-1, 1:-1] += noise

    return u
