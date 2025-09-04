import argparse
import os
import pickle
from collections import defaultdict

import bayesnewton
import contextily as cx
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from matplotlib.colors import to_hex, to_rgb
from matplotlib.transforms import Affine2D

from milsensors import data_helper

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="Gaussian Process example")

parser.add_argument("--N_runs", nargs="?", default=10, type=int)
parser.add_argument(
    "--N_sites_to_try",
    nargs="+",
    default=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    type=int,
)
parser.add_argument(
    "--dataset",
    default="phoenix",
    type=str,
    choices=[
        "phoenix",
        "arizona",
        "flagstaff",
        "tuscon",
        "miniphoenix",
        "arizona_refined",
        "restricted_phoenix",
    ],
)
parser.add_argument(
    "--data_file", default="../data/WRF_data_2013_and_2023_full.npz", type=str
)
parser.add_argument("--use_random", default=False, type=bool)
parser.add_argument("--N_runs_to_show", default=1, type=int)
parser.add_argument(
    "--spatial_kern",
    default="matern32",
    type=str,
    choices=["matern32", "sepmatern32", "deepmatern32"],
)
parser.add_argument("--noise_level", default=0.1, type=float)

args = parser.parse_args()

# config.update("jax_debug_nans", True)
# config.update("jax_enable_x64", True)
# config.update("jax_platform_name", "cpu")

mean_field = False
parallel = True

N_t = 24 * 30  # 92
N_t_start = 61 * 24  # 61 * 24  # 61 * 24  # 61 * 24
moving_points = False

use_optimal_points = not args.use_random

np.random.seed(1)


with open(
    f"results/N_fixed_optimal_Z_{args.dataset}_subbandmix_{args.noise_level}_{'_'.join([str(n) for n in args.N_sites_to_try])}",
    "rb",
) as f:
    z_opt = pickle.load(f)

# with open(
#     f"../results/N_fixed_optimal_Z_{args.dataset}_2_subbandmix_0.1_{'_'.join([str(n) for n in args.N_sites_to_try])}",
#     "rb",
# ) as f:
#     z_opt_2 = pickle.load(f)

with open(
    f"results/N_fixed_init_Z_{args.dataset}_subbandmix_{args.noise_level}_{'_'.join([str(n) for n in args.N_sites_to_try])}",
    "rb",
) as f:
    z_init = pickle.load(f)

with open(
    f"results/N_fixed_optimal_nat1_{args.dataset}_subbandmix_{args.noise_level}_{'_'.join([str(n) for n in args.N_sites_to_try])}",
    "rb",
) as f:
    nat1s = pickle.load(f)

with open(
    f"results/N_fixed_optimal_nat2_{args.dataset}_subbandmix_{args.noise_level}_{'_'.join([str(n) for n in args.N_sites_to_try])}",
    "rb",
) as f:
    nat2s = pickle.load(f)

with open(
    f"results/N_fixed_optimal_kernel_hypers_{args.dataset}_subbandmix_{args.noise_level}_{'_'.join([str(n) for n in args.N_sites_to_try])}",
    "rb",
) as f:
    kernel_hypers = pickle.load(f)

with open(
    f"results/N_fixed_optimal_post_covs_{args.dataset}_subbandmix_{args.noise_level}_{'_'.join([str(n) for n in args.N_sites_to_try])}",
    "rb",
) as f:
    variational_covs = pickle.load(f)

with open(
    f"results/N_fixed_optimal_post_means_{args.dataset}_subbandmix_{args.noise_level}_{'_'.join([str(n) for n in args.N_sites_to_try])}",
    "rb",
) as f:
    variational_means = pickle.load(f)


results_errors_gt_1 = []

pseudo_var_results = []
variational_var_results = []

pseudo_var_nlpd_results = []
variational_var_nlpd_results = []

for i in range(args.N_runs_to_show):
    print("##########################")
    print(f"######## i = {i} #########")

    optimal_points = np.array(z_opt[i])
    N_obs_pts = optimal_points.shape[0]
    print(f"######## N_obs = {N_obs_pts} #########")
    print("##########################")
    (
        X,
        Y,
        X_t,
        Y_t,
        air_temp_timeseries,
        N_t,
        mus,
        stds,
        N_sites,
        sample_points,
    ) = data_helper.create_dataset(
        0, -1, N_t, dataset=args.dataset, obs_noise=0.0, data_file=args.data_file
    )
    N_sites = air_temp_timeseries.shape[1]

    nat1 = nat1s[i][:N_t]
    posterior_variance = nat2s[i][:N_t]
    nat2 = posterior_variance
    # k_hyper = kernel_hypers[i]
    vcov = variational_covs[i]
    vmean = variational_means[i]

    # Training data
    X = X.reshape(N_t * N_sites, 3)
    Y = Y.reshape(N_t * N_sites, 1)

    t, R, Y = bayesnewton.utils.create_spatiotemporal_grid(X, Y)

plt.clf()

Zs = [np.array(z_init[i]) * stds[1:3] + mus[1:3] for i in range(args.N_runs_to_show)]

Zps_opt = [
    np.array(z_opt[i]) * stds[1:3] + mus[1:3] for i in range(args.N_runs_to_show)
]

# Save off modified optimal points in CSV
for i in range(args.N_runs_to_show):
    Z_opt = np.array(z_opt[i]) * stds[1:3] + mus[1:3]
    np.savetxt(
        f"results/2013_{args.dataset}_{args.N_sites_to_try[0]}_optimal_pts_{i}.csv",
        Z_opt,
        delimiter=",",
        fmt="%f",
    )

Zps = []

for Zp_opt in Zps_opt:
    sample_points = []
    for optimal_pt in Zp_opt:
        # print(optimal_pt)
        sample_points.append(
            np.argmin(
                (air_temp_timeseries[0, :, 1] * stds[1] + mus[1] - optimal_pt[0]) ** 2
                + (air_temp_timeseries[0, :, 2] * stds[2] + mus[2] - optimal_pt[1]) ** 2
            )
        )
    sample_points = np.array(sample_points)
    Zps.append(air_temp_timeseries[0, sample_points, 1:3] * stds[1:3] + mus[1:3])

shared_points = defaultdict(list)

# We assume Zps is a list of length 3:
#   Zps[0] are the points for group 0
#   Zps[1] are the points for group 1
#   Zps[2] are the points for group 2
for group_idx, Zp_array in enumerate(Zps):
    for pt in Zp_array:
        # pt is something like [lat, lon] or [y, x]
        # Round to a few decimal places to avoid floating-point "almost matches"
        coord_rounded = (round(pt[0], 6), round(pt[1], 6))
        shared_points[coord_rounded].append(group_idx)


# ---------------------------------------------------------------------
# 2) Define a helper function that draws one coordinate with either
#    single, half, or triple wedges, depending on how many groups share it.
# ---------------------------------------------------------------------
def plot_shared_point(ax, lat, lon, groups, colors, radius=0.05, angle_deg=45):
    """
    Draws a circle at (lat, lon) with diagonal stripes for each group color.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    lat, lon : float
        The coordinate (latitude, longitude) or (y, x).
    groups : list of int
        Which group indices share this coordinate (e.g. [0], [0,1], or [0,1,2]).
    colors : list of str
        A list of color hex codes, e.g. ['#540d6e', '#ee4266', '#ffd23f'].
    radius : float
        The circle radius for each point.
    angle_deg : float
        The rotation angle (in degrees) for the diagonal stripes.
    """

    x = lon
    y = lat
    n = len(groups)

    # If there's only 1 group, just draw a circle in that color
    if n == 1:
        circ = patches.Circle(
            (x, y),
            radius=radius,
            facecolor=colors[groups[0]],
            edgecolor="black",
            lw=0.5,
            zorder=5,
        )
        ax.add_patch(circ)
        return

    # Draw an outline circle (facecolor='none') for clipping and boundary
    circle_outline = patches.Circle(
        (x, y),
        radius=radius,
        facecolor="none",
        edgecolor="black",
        lw=0.5,
        zorder=5,
    )
    ax.add_patch(circle_outline)

    # Each diagonal stripe covers a fraction of the bounding box,
    # which is width = 2*radius, height = 2*radius,
    # from (x - radius, y - radius) to (x + radius, y + radius).
    # We'll stack n stripes in the vertical direction, then rotate them.

    stripe_height = (2 * radius) / n

    # Create a rotation transform around the center (x, y)
    rotation = Affine2D().rotate_deg_around(x, y, angle_deg) + ax.transData

    for i, group_idx in enumerate(groups):
        # The bottom-left corner of the i-th stripe in the *unrotated* frame:
        rect_x = x - radius  # left edge
        rect_y = y - radius + i * stripe_height  # bottom edge
        rect_w = 2 * radius
        rect_h = stripe_height

        # Create the rectangle (unrotated)
        rect = patches.Rectangle(
            (rect_x, rect_y),
            rect_w,
            rect_h,
            facecolor=colors[group_idx],
            edgecolor="none",
            zorder=6,  # Above the circle outline
        )

        # Apply the rotation around (x, y)
        rect.set_transform(rotation)

        # Clip the rectangle to the circle
        rect.set_clip_path(circle_outline)

        ax.add_patch(rect)


# Step 3: Plotting
f, ax = plt.subplots(1, 1, figsize=(3.0, 3.0))

gdf = ox.geocode_to_gdf({"city": "Phoenix"})
xs, ys = (
    air_temp_timeseries[0, :, 2] * stds[2] + mus[2],
    air_temp_timeseries[0, :, 1] * stds[1] + mus[1],
)
ax.set_xlim(xs.min() - 0.1, xs.max() + 0.1)
ax.set_ylim(ys.min() - 0.05, ys.max() + 0.05)
cx.add_basemap(ax, crs="WGS84", source="OpenStreetMap.HOT", attribution=False)

plt.scatter(xs, ys, c="#36413E", s=2, alpha=0.3)

plt.scatter(
    [],
    [],
    color="black",
    marker="s",
    alpha=0.5,
    label="Initial Points",
    facecolor="None",
)
plt.scatter([], [], edgecolor="black", facecolor="None", label="Optimized Points")


# Original colors
fun_colors = ["#0072B2", "#E69F00", "#CC79A7"]


# Function to brighten a color
def brighten_color(hex_color, factor=1.1):
    rgb = to_rgb(hex_color)
    brightened_rgb = [
        min(1.0, c * factor) for c in rgb
    ]  # Ensure values don't exceed 1.0
    return to_hex(brightened_rgb)


# Brighten each color
colors = [brighten_color(color) for color in fun_colors]

# Plot the quiver lines (arrows) from Z -> Zp as you had:
for j, (Z, Zp) in enumerate(zip(Zs, Zps)):
    delta = Zp - Z
    dx = delta[:, 0]
    dy = delta[:, 1]

    # Original points
    plt.scatter(
        Z[:, 1],
        Z[:, 0],
        edgecolor="black",  # colors[j],
        facecolor=colors[j],  # "None",
        marker="s",
        s=10,
        lw=0.5,
        # label=f"Initial group {j}" if j == 0 else None,
    )

    # Arrows
    plt.quiver(
        Z[:, 1],
        Z[:, 0],
        dy,
        dx,
        angles="xy",
        scale_units="xy",
        scale=1,
        color=colors[j],
        width=0.01,
    )

# Z_fixed = np.zeros((2)) * stds[1:3] + mus[1:3]
# plt.scatter(Z_fixed[1], Z_fixed[0], edgecolor="black", facecolor="red", marker="*", s=10, lw=0.5, label='Fixed Point')

# ---------------------------------------------------------------------
# 4) Finally, plot the shared (perturbed) points with custom wedges.
# ---------------------------------------------------------------------
for coord, group_list in shared_points.items():
    lat, lon = coord  # recall we stored them as (lat, lon)
    plot_shared_point(ax, lat, lon, group_list, colors, radius=0.01)

# Make the legend text smaller
plt.legend(loc="lower left", fontsize=6.3)

# Make axis labels smaller (e.g., size=8)
plt.xlabel("Latitude (deg)", fontsize=8)
plt.ylabel("Longitude (deg)", fontsize=8)

# Make tick labels smaller (e.g., size=6)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)


# Display the plot
plt.savefig(f"plot_pertubation_{args.spatial_kern}.png", bbox_inches="tight", dpi=300)
