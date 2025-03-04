"""
Visualization script for creating spatial error maps from GP model predictions.

This script loads previously trained GP models with optimal inducing points and
creates visualizations of prediction errors across space. It generates several
types of plots:

1. RMSE maps showing the spatial distribution of prediction errors
2. Maps showing the proportion of errors greater than 1°C
3. Maps showing the maximum error at each location
4. Time series plots for selected example locations

The script uses OpenStreetMap as a base layer for the spatial plots to provide
geographical context. Results are saved as high-resolution PNG and PDF files.
"""

import models
from kernels import *
import warnings
from train_and_eval import training, eval_model
import contextily as cx
import osmnx as ox
from jax.scipy.linalg import cho_factor, cho_solve, block_diag, expm
import jax.numpy as jnp
from bayesnewton.utils import (
    scaled_squared_euclid_dist,
    softplus,
    softplus_inv,
    rotation_matrix,
)
import bayesnewton
import objax
import numpy as np
import pickle
import time
import sys
from scipy.cluster.vq import kmeans2
from jax.lib import xla_bridge
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import jax
from jax import config
from tqdm import tqdm

from pyproj import Transformer

from tensorflow_probability.substrates.jax.math import bessel_ive
import argparse

from scipy.stats import qmc

import os
import uncertainty_toolbox as uct

# Set GPU device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Parse command line arguments
parser = argparse.ArgumentParser(description="Gaussian Process example")

parser.add_argument("--N_runs", nargs="?", default=10, type=int,
                   help="Number of runs to analyze")
parser.add_argument(
    "--N_sites_to_try",
    nargs="+",
    default=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    type=int,
    help="Number of inducing points to analyze"
)
parser.add_argument(
    "--dataset",
    default="phoenix",
    type=str,
    choices=["phoenix", "arizona", "flagstaff", "tuscon", "miniphoenix"],
    help="Dataset to analyze"
)
parser.add_argument("--use_random", default=False, type=bool,
                   help="Whether to use random points instead of optimal points")
parser.add_argument("--idx", default=1, type=int,
                   help="Index of the model configuration to visualize")

args = parser.parse_args()

# Enable 64-bit precision for better numerical stability
config.update("jax_enable_x64", True)

# Configuration settings
mean_field = False  # Whether to use mean-field approximation
parallel = True     # Whether to use parallel computation
N_t = 24 * 30       # Number of time points to use (24 hours * 30 days)
N_t_start = 61 * 24 # Starting time index (61 days * 24 hours)
moving_points = False  # Whether to use moving points

use_optimal_points = not args.use_random  # Use optimal points unless random is specified

np.random.seed(1)  # Set random seed for reproducibility

# Load previously saved results from get_N_optimal.py
with open(
    f"../results/N_fixed_optimal_Z_{args.dataset}_subbandmix_0.1_{'_'.join([str(n) for n in args.N_sites_to_try])}",
    "rb",
) as f:
    z_opt = pickle.load(f)

with open(
    f"../results/N_fixed_optimal_nat1_{args.dataset}_subbandmix_0.1_{'_'.join([str(n) for n in args.N_sites_to_try])}",
    "rb",
) as f:
    nat1s = pickle.load(f)

with open(
    f"../results/N_fixed_optimal_nat2_{args.dataset}_subbandmix_0.1_{'_'.join([str(n) for n in args.N_sites_to_try])}",
    "rb",
) as f:
    nat2s = pickle.load(f)

with open(
    f"../results/N_fixed_optimal_kernel_hypers_{args.dataset}_subbandmix_0.1_{'_'.join([str(n) for n in args.N_sites_to_try])}",
    "rb",
) as f:
    kernel_hypers = pickle.load(f)

with open(
    f"../results/N_fixed_optimal_post_covs_{args.dataset}_subbandmix_0.1_{'_'.join([str(n) for n in args.N_sites_to_try])}",
    "rb",
) as f:
    variational_covs = pickle.load(f)

with open(
    f"../results/N_fixed_optimal_post_means_{args.dataset}_subbandmix_0.1_{'_'.join([str(n) for n in args.N_sites_to_try])}",
    "rb",
) as f:
    variational_means = pickle.load(f)

# Select the model configuration to visualize based on the idx argument
i = len(z_opt) - args.idx
print("##########################")
print(f"######## i = {i} #########")

# Get the optimal inducing points for this configuration
optimal_points = np.array(z_opt[i])
N_obs_pts = optimal_points.shape[0]
print(f"######## N_obs = {N_obs_pts} #########")
print("##########################")

# Load temperature data for Phoenix
air_temp_timeseries = np.load("../data/WRF_data_2013_phoenix.npz")[
    "air_temp_timeseries"
]
N_sites = air_temp_timeseries.shape[1]

# Select random points for example time series plots
random_idxs = [int(np.random.rand() * N_sites) for _ in range(3)]

# Extract model parameters
nat1 = nat1s[i][:N_t]
posterior_variance = nat2s[i][:N_t]
nat2 = posterior_variance
k_hyper = kernel_hypers[i]
vcov = variational_covs[i]
vmean = variational_means[i]

# Plot variational covariance
plt.clf()
plt.plot(vcov[:, 0, 0])
plt.savefig("vcov_t.png")
plt.clf()
plt.imshow(vcov[0])
plt.savefig("vcov.png")

# Standardize data for GPs
mus = np.mean(air_temp_timeseries, axis=(0, 1))
stds = np.std(air_temp_timeseries, axis=(0, 1))
# Don't change time
mus[0] = 0
stds[0] = 1
air_temp_timeseries = (air_temp_timeseries - mus) / stds

print("MUS", mus)
print("STDS", stds)

# Plot optimal inducing points
plt.clf()
plt.scatter(optimal_points[:, 0], optimal_points[:, 1])
plt.savefig(f"optimal_pts_{N_obs_pts}.png")
plt.clf()

# Find the nearest point in the training data for each optimal point
if use_optimal_points:
    sample_points = []
    for optimal_pt in optimal_points:
        sample_points.append(
            np.argmin(
                (air_temp_timeseries[0, :, 1] - optimal_pt[0]) ** 2
                + (air_temp_timeseries[0, :, 2] - optimal_pt[1]) ** 2
            )
        )
    sample_points = np.array(sample_points)
else:
    success = False

    # while not success:
    #     # Select points from the Latin Hypercube
    #     lh = qmc.LatinHypercube(d=2)
    #     lh_samples = lh.random(N_obs_pts)
    #     # print(np.min(air_temp_timeseries[0, :, 1:3], axis=0), np.max(air_temp_timeseries[0, :, 1:3], axis=0))
    #     sample_z = qmc.scale(
    #         lh_samples,
    #         np.min(air_temp_timeseries[0, :, 1:3], axis=0),
    #         np.max(air_temp_timeseries[0, :, 1:3], axis=0),
    #     )

    #     sample_points = []
    #     for z in sample_z:
    #         new_point = np.argmin(
    #             (air_temp_timeseries[0, :, 1] - z[0]) ** 2
    #             + (air_temp_timeseries[0, :, 2] - z[1]) ** 2
    #         )
    #         sample_points.append(new_point)

    #     sample_points = np.array(sample_points)

    #     success = len(np.unique(sample_points)) == len(sample_points)

    # Original Random
    sample_points = np.random.choice(N_sites, N_obs_pts, replace=False)

# Extract training data
X = air_temp_timeseries[N_t_start : N_t_start + N_t, sample_points, 0:3]
Y = air_temp_timeseries[N_t_start : N_t_start + N_t, sample_points, 3]

# Reshape data for GP model
X = X.reshape(N_t * N_obs_pts, 3)
Y = Y.reshape(N_t * N_obs_pts, 1)
X_t = air_temp_timeseries[N_t_start : N_t_start + N_t, :, 0:3].reshape(N_t * N_sites, 3)
Y_t = air_temp_timeseries[N_t_start : N_t_start + N_t, :, 3].reshape(N_t * N_sites, 1)

# Create spatiotemporal grid
t, R, Y = bayesnewton.utils.create_spatiotemporal_grid(X, Y)
t_t, R_t, Y_t = bayesnewton.utils.create_spatiotemporal_grid(X_t, Y_t)

# Calculate temperatures > 30°C for error analysis
temps_gt_30 = (mus[3] + stds[3] * air_temp_timeseries[:N_t, :, 3]).reshape(
    (-1, 1)
) > 30
N_rt = R_t.shape[1]

# Extract kernel hyperparameters
k_t_l, k_t_rf, k_t_v, k_tm_l, k_tm_v, k_s_v, k_s_l, k_l_v = k_hyper

# Create kernels with the optimal hyperparameters
kern_space = bayesnewton.kernels.Matern32(variance=k_s_v, lengthscale=k_s_l)
kern_time = bayesnewton.kernels.Sum(
    [
        SubbandMatern32(
            variance=k_t_v,
            lengthscale=k_t_l,
            radial_frequency=2 * np.pi / 24.0,
        ),
        bayesnewton.kernels.Matern32(variance=k_tm_v, lengthscale=k_tm_l),
    ]
)

# Create the GP model
if use_optimal_points:
    model = models.make_model(
        N_obs_pts,
        X,
        Y,
        R,
        t,
        k_l_v,
        kern_time,
        kern_space,
        opt_z=False,
        z_init=optimal_points,
    )
else:
    model = models.make_model(
        N_obs_pts,
        X,
        Y,
        R,
        t,
        k_l_v,
        kern_time,
        kern_space,
        opt_z=False,
        z_init=R[0],
    )

# Set the posterior variance from the saved values
model.posterior_variance.value = posterior_variance[
    : len(model.posterior_variance.value)
]

# Train the model (minimal training since we're using saved parameters)
model = training(model, verbose=False, lr_adam=0.1, iters=100, optimize_all=False)

# Get pseudo-likelihood parameters
pseudo_y, pseudo_var = model.compute_full_pseudo_lik()

# Run filtering and smoothing
_, (filter_mean, filter_cov) = model.filter(
    model.dt,
    model.kernel,
    pseudo_y,
    pseudo_var,
    parallel=model.parallel,
)
dt = np.concatenate([model.dt[1:], np.array([0.0])], axis=0)
smoother_mean, smoother_cov, gain = model.smoother(
    dt,
    model.kernel,
    filter_mean,
    filter_cov,
    return_full=True,
    parallel=model.parallel,
)

# Evaluate the model
(rmse, nlpd, errors_gt_1, temps_gt_30, posterior_mean, posterior_var) = eval_model(
    model,
    t_t,
    R_t,
    mus,
    stds,
    air_temp_timeseries,
    N_t,
    Y_t,
    N_minibatch=len(air_temp_timeseries),
    pseudo_lik_params=(
        pseudo_y,
        pseudo_var,
    ),
)

# Calculate error metrics for each location
rmses = np.sqrt(np.mean(((posterior_mean - Y_t.squeeze()) * stds[3]) ** 2, axis=0))
errs_gt_1 = np.sum(np.abs((posterior_mean - Y_t.squeeze()) * stds[3]) > 1.0, axis=0)
max_errs = np.max(np.abs((posterior_mean - Y_t.squeeze()) * stds[3]), axis=0)

# Plot number of errors by time
plt.clf()
plt.plot(
    (np.sum(np.abs((posterior_mean - Y_t.squeeze()) * stds[3]) > 1.0, axis=1))[
        : 24 * 5 + 7
    ]
)
plt.savefig(f"N_errors_by_time_{R.shape[1]}.png")

# Calculate uncertainty calibration metrics
print(
    uct.metrics.get_all_metrics(
        posterior_mean.flatten() * stds[3] + mus[3],
        stds[3] * np.sqrt(posterior_var).flatten(),
        stds[3] * np.asarray(Y_t).flatten() + mus[3],
    )
)

# Plot calibration curve
plt.clf()
f, ax = plt.subplots(1, 1, figsize=(4.2, 4.0))
uct.plot_calibration(
    posterior_mean.flatten() * stds[3] + mus[3],
    stds[3] * np.sqrt(posterior_var).flatten(),
    stds[3] * np.asarray(Y_t).flatten() + mus[3],
    ax=ax,
)
plt.savefig("MiscalibrationPlot.pdf")

# Function to truncate color map for better visualization
def truncate_colormap(cmapIn="jet", minval=0.0, maxval=1.0, n=100):
    """truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100)"""
    cmapIn = plt.get_cmap(cmapIn)

    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmapIn.name, a=minval, b=maxval),
        cmapIn(np.linspace(minval, maxval, n)),
    )

    return new_cmap

# Create RMSE map
plt.clf()
f, ax = plt.subplots(1, 1, figsize=(4.2, 4.0))
errs_cmap = truncate_colormap(cmapIn="bwr", minval=0.5, maxval=1.0, n=100)
gdf = ox.geocode_to_gdf({"city": "Phoenix"})

# Convert normalized coordinates back to lat/lon
xs, ys = (
    R_t[0, :, 1] * stds[2] + mus[2],
    R_t[0, :, 0] * stds[1] + mus[1],
)
ax.set_xlim(xs.min() - 0.1, xs.max() + 0.1)
ax.set_ylim(ys.min() - 0.05, ys.max() + 0.05)

# Add OpenStreetMap basemap
cx.add_basemap(ax, crs="WGS84", source="OpenStreetMap.HOT", attribution=False)

# Plot RMSE values
plt.scatter(xs, ys, c=rmses, cmap=errs_cmap, vmin=0.0, vmax=1.0, s=30)

# Format plot
plt.xlabel("Latitude (deg)", fontsize=8)
plt.ylabel("Longitude (deg)", fontsize=8)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=6.3)

# Plot inducing points
xs, ys = (
    R[0, :, 1] * stds[2] + mus[2],
    R[0, :, 0] * stds[1] + mus[1],
)
plt.scatter(xs, ys, facecolor="None", edgecolors="black", s=30)

# Mark example points
xs, ys = (
    R_t[0, np.array(random_idxs), 1] * stds[2] + mus[2],
    R_t[0, np.array(random_idxs), 0] * stds[1] + mus[1],
)
for i in range(len(random_idxs)):
    plt.scatter(xs[i], ys[i], marker=f"${i+1}$", color="black", s=27, lw=0.5)

plt.title("RMSE (deg C)", fontsize=10)

# Save RMSE map
if args.use_random:
    plt.savefig("spatiotemporal_rmse_map_uniform.png", bbox_inches="tight", dpi=300)
else:
    plt.savefig(
        f"spatiotemporal_rmse_map_{R.shape[1]}.png", bbox_inches="tight", dpi=300
    )

# Create map of errors > 1°C
plt.clf()
if args.idx == 1:
    f, ax = plt.subplots(1, 1, figsize=(4.2, 4.0))
else:
    f, ax = plt.subplots(1, 1, figsize=(4.0, 4.0))
errs_cmap = truncate_colormap(cmapIn="PuOr", minval=0.5, maxval=1.0, n=100)

# Set up map
xs, ys = (
    R_t[0, :, 1] * stds[2] + mus[2],
    R_t[0, :, 0] * stds[1] + mus[1],
)
ax.set_xlim(xs.min() - 0.1, xs.max() + 0.1)
ax.set_ylim(ys.min() - 0.05, ys.max() + 0.05)
cx.add_basemap(ax, crs="WGS84", source="OpenStreetMap.HOT", attribution=False)

# Plot proportion of errors > 1°C
scatter_out = plt.scatter(
    xs, ys, c=errs_gt_1 / 720, cmap=errs_cmap, vmin=0.0, vmax=0.5, s=30
)

# Format plot
plt.xlabel("Latitude (deg)", fontsize=8)
plt.ylabel("Longitude (deg)", fontsize=8)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
if args.idx == 1:
    cbar = plt.colorbar(scatter_out)
    cbar.ax.tick_params(labelsize=6.3)

# Plot inducing points
xs, ys = (
    R[0, :, 1] * stds[2] + mus[2],
    R[0, :, 0] * stds[1] + mus[1],
)
plt.scatter(xs, ys, facecolor="None", edgecolors="black", s=30)

xs, ys = (
    R_t[0, np.array(random_idxs), 1] * stds[2] + mus[2],
    R_t[0, np.array(random_idxs), 0] * stds[1] + mus[1],
)
if args.use_random:
    plt.savefig("spatiotemporal_err_gt_1_map_uniform.png", bbox_inches="tight", dpi=300)
else:
    plt.savefig(
        f"spatiotemporal_err_gt_1_map_{R.shape[1]}.png", bbox_inches="tight", dpi=300
    )

# Create standalone colorbar for use in publications
f, ax = plt.subplots(1, 1, figsize=(4.2, 4.0))
cbar = plt.colorbar(scatter_out)
cbar.ax.tick_params(labelsize=6.3)
ax.remove()
plt.savefig("plot_onlycbar.png", bbox_inches="tight", dpi=300)

# Create map of maximum errors
plt.clf()
f, ax = plt.subplots(1, 1, figsize=(4.2, 4.0))
errs_cmap = truncate_colormap(cmapIn="bwr", minval=0.5, maxval=1.0, n=100)

# Set up map
xs, ys = (
    R_t[0, :, 1] * stds[2] + mus[2],
    R_t[0, :, 0] * stds[1] + mus[1],
)
ax.set_xlim(xs.min() - 0.1, xs.max() + 0.1)
ax.set_ylim(ys.min() - 0.05, ys.max() + 0.05)
cx.add_basemap(ax, crs="WGS84", source="OpenStreetMap.HOT", attribution=False)

# Plot maximum errors
plt.scatter(xs, ys, c=max_errs, cmap=errs_cmap, vmin=0.0, vmax=6.0, s=30)

# Format plot
plt.xlabel("Latitude (deg)", fontsize=8)
plt.ylabel("Longitude (deg)", fontsize=8)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=6.3)

# Plot inducing points
xs, ys = (
    R[0, :, 1] * stds[2] + mus[2],
    R[0, :, 0] * stds[1] + mus[1],
)
plt.scatter(xs, ys, facecolor="None", edgecolors="black", s=30)

# Mark example points
xs, ys = (
    R_t[0, np.array(random_idxs), 1] * stds[2] + mus[2],
    R_t[0, np.array(random_idxs), 0] * stds[1] + mus[1],
)
for i in range(len(random_idxs)):
    plt.scatter(xs[i], ys[i], marker=f"${i+1}$", color="black", s=27, lw=0.5)

plt.title("Maximum Errors", fontsize=10)

# Save maximum errors map
if args.use_random:
    plt.savefig("spatiotemporal_max_err_uniform.png", bbox_inches="tight", dpi=300)
else:
    plt.savefig(
        f"spatiotemporal_max_err_map_{R.shape[1]}.png", bbox_inches="tight", dpi=300
    )

# Create time series plots for example points
# Set up the theme
sns.set_theme(context="paper", style="white", palette="colorblind")

# Create a single figure with a 2×3 grid of subplots
fig, axes = plt.subplots(
    nrows=2,
    ncols=3,
    sharex=True,
    sharey="row",  # all top subplots share one y-axis, all bottom share another
    figsize=(7, 4),
)
GMT_OFFSET = 7  # Time zone offset for Phoenix

# Create plots for each example point
for i, random_idx in enumerate(random_idxs):
    # Retrieve axes for top (ax1) and bottom (ax2) subplot in column i
    ax1 = axes[0, i]
    ax2 = axes[1, i]

    # Pick a random 3-day window
    start_day = np.random.randint(low=0, high=N_t // 24 - 4)
    end_day = start_day + 3

    # Get location information
    print("Rand Lat", R_t[0, random_idx, 1] * stds[2] + mus[2])
    print("Rand Long", R_t[1, random_idx, 0] * stds[1] + mus[1])

    # Extract posterior predictions for this location
    post_mean = (
        np.reshape(posterior_mean, (N_t, N_sites))[
            start_day * 24 + GMT_OFFSET : end_day * 24 + GMT_OFFSET, random_idx
        ]
        * stds[3]
        + mus[3]
    )
    post_std = stds[3] * np.sqrt(
        np.reshape(posterior_var, (N_t, N_sites))[
            start_day * 24 + GMT_OFFSET : end_day * 24 + GMT_OFFSET, random_idx
        ]
    )

    # Define colors for uncertainty bands
    c_light = "#DCBCBC"
    c_light_highlight = "#C79999"
    c_mid = "#B97C7C"
    c_mid_highlight = "#A25050"
    c_dark = "#8F2727"

    # Plot prediction with uncertainty bands
    for z, c in zip(
        [1.28, 0.84, 0.52, 0.25], [c_light, c_light_highlight, c_mid, c_mid_highlight]
    ):
        ax1.fill_between(
            np.arange(3 * 24),
            post_mean - z * post_std,
            post_mean + z * post_std,
            color=c,
        )
    ax1.plot(np.arange(3 * 24), post_mean, color=c_dark, label="GP Prediction")
    
    # Plot true data
    ax1.plot(
        Y_t[start_day * 24 + GMT_OFFSET : end_day * 24 + GMT_OFFSET, random_idx, 3]
        * stds[3]
        + mus[3],
        c="#a1c3f7",
        label="True",
    )

    # Add legend to first plot only
    if i == 0:
        ax1.legend(fontsize=8, loc="lower left")

    # Add title
    ax1.set_title(
        f"Example {i+1}\nAugust {start_day+1} to August {end_day+1}", fontsize=10
    )

    # Add y-label to leftmost plot only
    if i == 0:
        ax1.set_ylabel("Air Temp. (deg C)", fontsize=8)

    ax1.tick_params(labelsize=6)

    # Plot errors in bottom subplot
    errs = post_mean - (
        Y_t[start_day * 24 + GMT_OFFSET : end_day * 24 + GMT_OFFSET, random_idx, 3]
        * stds[3]
        + mus[3]
    )
    mask = np.ma.masked_less(abs(errs), 1.0) * np.sign(errs)
    colors = ["red" if abs(err) > 1 else "black" for err in errs]

    # Add reference lines at +/- 1°C
    ax2.plot(
        [0, (end_day - start_day) * 24],
        [-1, -1],
        linewidth=2,
        c="black",
        linestyle="--",
    )
    ax2.plot(
        [0, (end_day - start_day) * 24], [1, 1], linewidth=2, c="black", linestyle="--"
    )
    
    # Plot errors
    ax2.scatter(range((end_day - start_day) * 24), errs, s=10, c=colors)
    ax2.plot(range((end_day - start_day) * 24), errs, c="black")
    ax2.plot(range((end_day - start_day) * 24), mask, c="red")

    # Add y-label to leftmost plot only
    if i == 0:
        ax2.set_ylabel("Error (deg C)", fontsize=8)

    ax2.set_xlabel(f"Hours Since\nMidnight August {start_day+1}", fontsize=8)
    ax2.tick_params(labelsize=6)

# Adjust layout
plt.tight_layout()

# Save time series plots
if args.use_random:
    plt.savefig("example_timeseries_combined_uniform.pdf", bbox_inches="tight", dpi=300)
else:
    plt.savefig(
        f"example_timeseries_combined_{R.shape[1]}.pdf", bbox_inches="tight", dpi=300
    )
