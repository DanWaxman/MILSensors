"""
Analysis script for evaluating optimal inducing point configurations.

This script loads previously saved optimal inducing point configurations and
evaluates their performance on temperature prediction tasks. It compares
different methods for using the inducing points:
1. Using the pseudo-likelihood parameters directly
2. Using the variational parameters

The script visualizes results and saves performance metrics for further analysis.
"""

import models
from kernels import SubbandMatern32
from train_and_eval import training, eval_model
import bayesnewton
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import qmc
import argparse
import os

# Set GPU device
# Should maybe do this via command line argument...
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Parse command line arguments
parser = argparse.ArgumentParser(description="Gaussian Process example")

parser.add_argument(
    "--N_runs", nargs="?", default=10, type=int, help="Number of runs to analyze"
)
parser.add_argument(
    "--N_sites_to_try",
    nargs="+",
    default=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    type=int,
    help="Number of inducing points to analyze",
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
        "restricted_phoenix",
    ],
    help="Dataset to analyze",
)
parser.add_argument(
    "--use_random",
    default=False,
    type=bool,
    help="Whether to use random points instead of optimal points",
)
parser.add_argument(
    "--use_lhs",
    default=False,
    type=bool,
    help="Whether to use LHS instead of random points",
)

args = parser.parse_args()

# Configuration settings
mean_field = False  # Whether to use mean-field approximation
parallel = True  # Whether to use parallel computation
N_t = 24 * 30  # Number of time points to use (24 hours * 30 days)
N_t_start = 61 * 24  # Starting time index (61 days * 24 hours)
moving_points = False  # Whether to use moving points

use_optimal_points = (
    not args.use_random
)  # Use optimal points unless random is specified
use_lhs = args.use_lhs  # Use LHS instead of random points

np.random.seed(1)  # Set random seed for reproducibility

# Load previously saved results from get_N_optimal.py

# Load optimal inducing points
with open(
    f"../results/N_fixed_optimal_Z_{args.dataset}_subbandmix_0.1_{'_'.join([str(n) for n in args.N_sites_to_try])}",
    "rb",
) as f:
    z_opt = pickle.load(f)

# Load natural parameters (first natural parameter - pseudo observations)
with open(
    f"../results/N_fixed_optimal_nat1_{args.dataset}_subbandmix_0.1_{'_'.join([str(n) for n in args.N_sites_to_try])}",
    "rb",
) as f:
    nat1s = pickle.load(f)

# Load natural parameters (second natural parameter - pseudo precision)
with open(
    f"../results/N_fixed_optimal_nat2_{args.dataset}_subbandmix_0.1_{'_'.join([str(n) for n in args.N_sites_to_try])}",
    "rb",
) as f:
    nat2s = pickle.load(f)

# Load kernel hyperparameters
with open(
    f"../results/N_fixed_optimal_kernel_hypers_{args.dataset}_subbandmix_0.1_{'_'.join([str(n) for n in args.N_sites_to_try])}",
    "rb",
) as f:
    kernel_hypers = pickle.load(f)

# Load variational parameters (posterior covariances)
with open(
    f"../results/N_fixed_optimal_post_covs_{args.dataset}_subbandmix_0.1_{'_'.join([str(n) for n in args.N_sites_to_try])}",
    "rb",
) as f:
    variational_covs = pickle.load(f)

# Load variational parameters (posterior means)
with open(
    f"../results/N_fixed_optimal_post_means_{args.dataset}_subbandmix_0.1_{'_'.join([str(n) for n in args.N_sites_to_try])}",
    "rb",
) as f:
    variational_means = pickle.load(f)

# Initialize results storage
results_errors_gt_1 = []  # Errors greater than 1°C
pseudo_var_results = []  # RMSE results using pseudo-likelihood parameters
variational_var_results = []  # RMSE results using variational parameters
pseudo_var_nlpd_results = []  # NLPD results using pseudo-likelihood parameters
variational_var_nlpd_results = []  # NLPD results using variational parameters

# Analyze each set of inducing points
for i in range(len(z_opt)):
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

    # Extract natural parameters and kernel hyperparameters
    nat1 = nat1s[i][:N_t]
    posterior_variance = nat2s[i][:N_t]
    nat2 = posterior_variance
    k_hyper = kernel_hypers[i]
    vcov = variational_covs[i]
    vmean = variational_means[i]

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

    # Create a visualization of the optimal inducing points
    plt.clf()
    plt.scatter(optimal_points[:, 0], optimal_points[:, 1])
    plt.savefig(f"optimal_pts_{N_obs_pts}.png")
    plt.clf()

    if use_optimal_points:
        sample_points = []
        # Find the nearest point in the training data for each optimal point
        for optimal_pt in optimal_points:
            # print(optimal_pt)
            sample_points.append(
                np.argmin(
                    (air_temp_timeseries[0, :, 1] - optimal_pt[0]) ** 2
                    + (air_temp_timeseries[0, :, 2] - optimal_pt[1]) ** 2
                )
            )

        sample_points = np.array(sample_points)
    else:
        success = False

        if use_lhs:
            while not success:
                # Select points from the Latin Hypercube
                lh = qmc.LatinHypercube(d=2)
                lh_samples = lh.random(N_obs_pts)
                sample_z = qmc.scale(
                    lh_samples,
                    np.min(air_temp_timeseries[0, :, 1:3], axis=0),
                    np.max(air_temp_timeseries[0, :, 1:3], axis=0),
                )

                sample_points = []
                for z in sample_z:
                    new_point = np.argmin(
                        (air_temp_timeseries[0, :, 1] - z[0]) ** 2
                        + (air_temp_timeseries[0, :, 2] - z[1]) ** 2
                    )
                    sample_points.append(new_point)

                sample_points = np.array(sample_points)

                success = len(np.unique(sample_points)) == len(sample_points)

        # Original Random
        sample_points = np.random.choice(N_sites, N_obs_pts, replace=False)

    X = air_temp_timeseries[N_t_start : N_t_start + N_t, sample_points, 0:3]
    Y = air_temp_timeseries[N_t_start : N_t_start + N_t, sample_points, 3]

    print("NAT2 SHAPE:", nat2.shape, sample_points)
    print(np.diag(nat2[0]))
    print(np.sum(np.diag(nat2[0]) > 0))
    # nat2 = nat2[:N_t, sample_points[:, None], sample_points].reshape(
    #     (N_t, N_obs_pts, N_obs_pts)
    # )
    # nat2 = jnp.repeat(jnp.mean(nat2, axis=0, keepdims=True), nat2.shape[0], axis=0)
    print("NAT2 SHAPE:", nat2.shape)
    print(np.sum(np.diag(nat2[0]) > 0))
    print("NAT1 SHAPE:", nat1.shape)
    mean_1 = air_temp_timeseries[N_t_start : N_t_start + N_t, sample_points, 3].reshape(
        (N_t, N_obs_pts, 1)
    )
    # nat1 = -2 * nat2 @ mean_1
    print("NAT1 SHAPE:", nat1.shape)

    X = X.reshape(N_t * N_obs_pts, 3)
    Y = Y.reshape(N_t * N_obs_pts, 1)
    X_t = air_temp_timeseries[N_t_start : N_t_start + N_t, :, 0:3].reshape(
        N_t * N_sites, 3
    )
    Y_t = air_temp_timeseries[N_t_start : N_t_start + N_t, :, 3].reshape(
        N_t * N_sites, 1
    )

    print("X:", X.shape)
    print("Y:", Y.shape)

    print(Y.shape)
    print("num data points =", Y.shape[0])

    t, R, Y = bayesnewton.utils.create_spatiotemporal_grid(X, Y)
    t_t, R_t, Y_t = bayesnewton.utils.create_spatiotemporal_grid(X_t, Y_t)
    print(X_t[:10, 0:2])
    print(Y_t[:10, 0])

    Nt = t.shape[0]
    print("num time steps =", Nt)
    Nr = R.shape[1]
    print("num spatial points =", Nr, R_t.shape[1])
    N = Y.shape[0] * Y.shape[1] * Y.shape[2]
    print("num data points =", N)

    temps_gt_30 = (mus[3] + stds[3] * air_temp_timeseries[:N_t, :, 3]).reshape(
        (-1, 1)
    ) > 30  #  (30 + 273)
    N_rt = R_t.shape[1]
    print(np.sum(temps_gt_30), N_t * N_rt, N_t * N_rt / np.sum(temps_gt_30))

    import matplotlib.pyplot as plt

    plt.clf()
    plt.plot(nat1.squeeze()[:, 3], label="Variational Pseudovar")
    plt.plot(Y.squeeze()[:, 3], label="Y")
    # plt.plot(pseudo_y.squeeze()[:, 3], label="Current Pseudovar")
    plt.ylim([-2, 2])
    plt.legend()
    plt.savefig("means_.png")  # , bbox_inches="tight")

    plt.clf()
    plt.plot(nat1.squeeze()[:, 3] - Y.squeeze()[:, 3], label="Variational Pseudovar")
    plt.savefig(f"means_diff_{N_obs_pts}.png")  # , bbox_inches="tight")

    k_t_l, k_t_rf, k_t_v, k_tm_l, k_tm_v, k_s_v, k_s_l, k_l_v = k_hyper
    len_space = 4.5133
    # Create the model for interpolation
    print("k_t_v", k_t_v)
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

    print("R0 SHAPE", R[0, ...].shape, "NUM_Z", N_obs_pts)

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

    model.posterior_variance.value = posterior_variance[
        : len(model.posterior_variance.value)
    ]

    print("Time Kernel:")
    print(
        kern_time.kernel0.lengthscale,
        kern_time.kernel0.radial_frequency,
        kern_time.kernel0.variance,
        kern_time.kernel1.lengthscale,
        kern_time.kernel1.variance,
    )
    print("Space Kernel:")
    print(kern_space.variance, kern_space.lengthscale)

    model = training(model, verbose=False, lr_adam=0.1, iters=100, optimize_all=False)

    print(
        kern_time.kernel0.variance,
        kern_time.kernel0.lengthscale,
        kern_time.kernel1.variance,
        kern_time.kernel1.lengthscale,
    )
    print(kern_space.variance, kern_space.lengthscale)

    # Eval models
    print("y_shapes", model.posterior_mean.value.shape, Y.shape)
    pseudo_y, pseudo_var = model.compute_full_pseudo_lik()
    # print(np.diag(pseudo_var[5]))
    # print(np.diag(posterior_variance[5]))

    print(model.kernel.z.value)
    print(optimal_points)

    plt.clf()
    plt.plot(Y.squeeze()[:, 3], label="Y")
    plt.plot(model.posterior_mean.value.squeeze()[:, 3], label="posterior_mean")
    plt.legend()
    plt.savefig("posterior_mean.png")

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
    print("Filter Shape", filter_mean.shape, filter_cov.shape)
    print("Smoother Shape", smoother_mean.shape, smoother_cov.shape)

    print("nat1_diffs", np.mean(nat1.squeeze() - pseudo_y.squeeze(), axis=0))
    plt.clf()
    plt.plot(
        nat1.squeeze()[:, 3] - pseudo_y.squeeze()[:, 3], label="Variational Pseudovar"
    )

    diff = (nat1.squeeze() - pseudo_y.squeeze()).reshape(-1, 24, N_obs_pts).mean(1)
    print("diff", diff)
    with open(
        "../results/diff",
        "wb",
    ) as handle:
        pickle.dump(diff, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.savefig(f"means_diff_2_{N_obs_pts}.png")

    print("Time Kernel:")
    print(
        kern_time.kernel0.lengthscale,
        kern_time.kernel0.radial_frequency,
        kern_time.kernel0.variance,
        kern_time.kernel1.lengthscale,
        kern_time.kernel1.variance,
    )
    print("Space Kernel:")
    print(kern_space.variance, kern_space.lengthscale)

    # Evaluate model using pseudo-likelihood parameters
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
    print("posterior mean shape", posterior_mean.shape)
    print("With pseudo var")
    print("RMSE", rmse, " | NLPD", nlpd)
    pseudo_var_results.append(rmse)
    pseudo_var_nlpd_results.append(nlpd)

    print("pseudovar_shape", pseudo_var.shape)
    print("pseudo_y shape", pseudo_y.shape)
    print("Y shape", Y.shape)

    plt.clf()
    vm, vc = model.get_variational_params()

    print(f"t_t {t_t.shape} R_t {R_t.shape} N_t {N_t} Y_t {Y_t.shape}")

    # Evaluate model using variational parameters
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
        variational_params=(vm, vcov),
    )
    variational_var_results.append(rmse)
    variational_var_nlpd_results.append(nlpd)
    print("With variational")
    print("RMSE", rmse, " | NLPD", nlpd)

    # Plot inducing point time series
    plt.clf()
    plt.errorbar(
        np.arange(240),
        pseudo_y.squeeze()[:240, 0],
        np.sqrt(pseudo_var.squeeze()[:240, 0, 0]),
        fmt="o",
    )
    plt.plot(Y[:240, 0])
    plt.savefig("inducing_ts.png")

    # Calculate error statistics
    errors_gt_1 = (
        stds[3]
        * np.abs(
            np.reshape(posterior_mean, (N_t, N_rt))
            - air_temp_timeseries[N_t_start : N_t_start + N_t, :, 3]
        ).reshape((-1, 1))
        > 1
    )
    print(np.sum(errors_gt_1) / len(errors_gt_1))
    print(np.sum(errors_gt_1 * temps_gt_30) / np.sum(temps_gt_30))

    results_errors_gt_1.append(errors_gt_1 / np.sum(errors_gt_1))

# Print final results
print(results_errors_gt_1)
print(pseudo_var_nlpd_results)
print(pseudo_var_results)
print(variational_var_nlpd_results)
print(variational_var_results)
