# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)
# Relatively easy to find good solution
rng = np.random.default_rng(1234)
# More difficult
# rng = np.random.default_rng(100)

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (6, 6)
plt.rcParams.update({"font.size": 10})
# Ignore error when drawing many figures
plt.rcParams.update({"figure.max_open_warning": 0})
from ipywidgets import interact
import ipywidgets as widgets

from p_tqdm import p_map

# %%
# %load_ext autoreload
# %autoreload 2
from dass import pde, utils, analysis, taper, geostat

# %% [markdown]
# ## Define ensemble size and parameters related to the simulator

# %%
N = 50

# Number of grid-cells in x, y, and z directions
nx = 8
ny = 8
nz = 8

# time steps
k_start = 0
k_end = 10000

# %%
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
z = np.linspace(0, 1, nz)
mesh = np.meshgrid(x, y, z)

def sample_prior_conductivity(N, mesh):
    return np.exp(geostat.gaussian_fields(mesh, rng, N, r=0.8))

field_3D = sample_prior_conductivity(1, mesh).reshape(nx, ny, nz)

# %% [markdown]
# # Define and visualize the prior parameter ensemble

# %%
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

pio.renderers.default = "notebook"

X, Y, Z = mesh

# Create the figure
fig = go.Figure()

# 1. Volume rendering with corrected colorbar settings
fig.add_trace(go.Volume(
    x=X.flatten(),
    y=Y.flatten(), 
    z=Z.flatten(),
    value=field_3D.flatten(),
    isomin=field_3D.min(),
    isomax=field_3D.max(),
    opacity=0.05,  # More transparent for reservoir viz
    surface_count=15,
    colorscale='RdYlBu_r',  # Reservoir colors: red=high, blue=low
    showscale=True,
    colorbar=dict(
        title=dict(text="Property Value", side="right"),  # Corrected syntax
        thickness=15,
        len=0.7
    ),
    name="Volume"
))

# 2. Add slice planes
slice_x = nx // 2  # Middle slice
slice_y = ny // 2
slice_z = nz // 2

# XY slice (horizontal - most important for reservoirs)
fig.add_trace(go.Surface(
    x=X[slice_x, :, :],
    y=Y[slice_x, :, :],
    z=Z[slice_x, :, :],
    surfacecolor=field_3D[slice_x, :, :],
    colorscale='RdYlBu_r',
    showscale=False,
    opacity=0.9,
    name=f"XY Slice (z={z[slice_x]:.2f})"
))

# XZ slice (vertical)
fig.add_trace(go.Surface(
    x=X[:, slice_y, :],
    y=Y[:, slice_y, :],
    z=Z[:, slice_y, :],
    surfacecolor=field_3D[:, slice_y, :],
    colorscale='RdYlBu_r',
    showscale=False,
    opacity=0.8,
    name=f"XZ Slice (y={y[slice_y]:.2f})"
))

# YZ slice (vertical)
fig.add_trace(go.Surface(
    x=X[:, :, slice_z],
    y=Y[:, :, slice_z],
    z=Z[:, :, slice_z],
    surfacecolor=field_3D[:, :, slice_z],
    colorscale='RdYlBu_r',
    showscale=False,
    opacity=0.8,
    name=f"YZ Slice (x={x[slice_z]:.2f})"
))

# Reservoir-appropriate layout
fig.update_layout(
    title="3D Reservoir Property Field",
    scene=dict(
        xaxis_title='X Distance',
        yaxis_title='Y Distance',
        zaxis_title='Z Depth',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=0.8)  # Better view angle for reservoirs
        ),
        aspectmode='cube'
    ),
    width=800,
    height=700
)

fig.show()

# %%
u0 = np.zeros((nx, ny, nz))
# Hot column at center point through all Z layers
center_x, center_y = nx//2, ny//2
u0[center_x, center_y, :] = 100  # Hot in all Z layers

dx = 1
dt = 0.0001
scale = None
num_steps = k_end - k_start
u_t = pde.heat_equation(u0, field_3D, dx, dt, num_steps, rng=rng, scale=scale)


# %%
def interactive_truth_slice(k, slice_type='xy', slice_index=4):
    fig, ax = plt.subplots()
    fig.suptitle("True temperature field - 3D Slice")
    
    if slice_type == 'xy':
        # XY plane at fixed Z
        data = u_t[k, :, :, slice_index].T
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f"k = {k}, XY plane at Z = {slice_index}")
    elif slice_type == 'xz':
        # XZ plane at fixed Y
        data = u_t[k, :, slice_index, :].T
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_title(f"k = {k}, XZ plane at Y = {slice_index}")
    elif slice_type == 'yz':
        # YZ plane at fixed X
        data = u_t[k, slice_index, :, :].T
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        ax.set_title(f"k = {k}, YZ plane at X = {slice_index}")
    
    p = ax.pcolormesh(data, cmap=plt.cm.jet, vmin=0, vmax=100)
    utils.colorbar(p)
    fig.tight_layout()

interact(
    interactive_truth_slice,
    k=widgets.IntSlider(min=k_start, max=k_end - 1, step=1, value=0),
    slice_type=widgets.Dropdown(
        options=['xy', 'xz', 'yz'],
        value='xy',
        description='Slice type:'
    ),
    slice_index=widgets.IntSlider(min=0, max=7, step=1, value=4, description='Slice index:')
)

# %%

# %%
