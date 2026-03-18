import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from dass.pde import heat_equation, heat_equation_jax


@pytest.fixture
def setup():
    """Shared grid, initial condition, and parameters."""
    ny, nx = 10, 10
    dx = 1.0
    dt = 0.1
    num_steps = 20

    rng = np.random.default_rng(42)
    u0 = rng.standard_normal((ny, nx))
    # Both implementations enforce fixed boundary conditions;
    # the NumPy version implicitly sets them to zero, so we match that.
    u0[0, :] = u0[-1, :] = u0[:, 0] = u0[:, -1] = 0.0
    alpha = np.full((ny, nx), 0.5)

    return u0, alpha, dx, dt, num_steps


def test_numpy_jax_agree(setup):
    """JAX implementation should match the NumPy one (no noise)."""
    u0, alpha, dx, dt, num_steps = setup

    rng = np.random.default_rng(0)  # unused when scale=None
    u_np = heat_equation(u0, alpha, dx, dt, num_steps, rng, scale=None)

    u_jax = heat_equation_jax(
        jnp.array(u0), jnp.array(alpha), dx, dt, num_steps, key=None, scale=None
    )

    np.testing.assert_allclose(np.array(u_jax), u_np, atol=1e-10)


def test_jax_differentiable_wrt_alpha(setup):
    """Gradient of the final temperature w.r.t. alpha should be computable."""
    u0, alpha, dx, dt, num_steps = setup

    u0_j = jnp.array(u0)
    alpha_j = jnp.array(alpha)

    def scalar_output(a):
        u = heat_equation_jax(u0_j, a, dx, dt, num_steps)
        return u[-1].sum()

    grad_fn = jax.grad(scalar_output)
    grad_alpha = grad_fn(alpha_j)

    assert grad_alpha.shape == alpha_j.shape
    # Gradient should be non-trivial (not all zeros)
    assert jnp.abs(grad_alpha).max() > 0


def test_jax_differentiable_wrt_u0(setup):
    """Gradient of the final temperature w.r.t. initial condition."""
    u0, alpha, dx, dt, num_steps = setup

    u0_j = jnp.array(u0)
    alpha_j = jnp.array(alpha)

    def scalar_output(u_init):
        u = heat_equation_jax(u_init, alpha_j, dx, dt, num_steps)
        return u[-1].sum()

    grad_fn = jax.grad(scalar_output)
    grad_u0 = grad_fn(u0_j)

    assert grad_u0.shape == u0_j.shape
    assert jnp.abs(grad_u0).max() > 0
