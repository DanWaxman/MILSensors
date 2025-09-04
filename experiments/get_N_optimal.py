"""
Experiment script to find the optimal inducing points for GP models.

This script runs experiments to determine the optimal inducing points
for spatiotemporal Gaussian Process models. It trains models with different numbers
of inducing points and evaluates their performance on test data.

The script:
1. Loads temperature data for a specified region
2. Creates and trains GP models with different numbers of inducing points
3. Evaluates model performance using RMSE and NLPD metrics
4. Saves results for further analysis

Command line arguments allow customization of the experiment parameters.
"""

# It's important this runs at the top and not in the main block
# because of a pecularity in how jax allocates devices
import argparse
import os
import pickle

from scipy.cluster.vq import kmeans2

# Make sure to include cublas and cuda_runtime in the LD_LIBRARY_PATH
# On DLS0, you can do this with:
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/anaconda3/lib/python3.7/site-packages/nvidia/cuda_runtime/lib:/usr/local/anaconda3/lib/python3.7/site-packages/nvidia/cublas/lib
parser = argparse.ArgumentParser(description="Gaussian Process example")
parser.add_argument("--N_t", nargs="?", default=24 * 92, type=int)
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
        "urban_corridor",
        "miniphoenix",
        "restricted_phoenix",
    ],
)
parser.add_argument("--data_file", default="data/WRF_data_2013", type=str)
parser.add_argument("--device_num", nargs="?", default="0", type=str)
parser.add_argument(
    "--spatial_kern",
    default="matern32",
    type=str,
    choices=["matern32", "sepmatern32"],
)
parser.add_argument(
    "--temp_kern",
    default="subbandmix",
    type=str,
    choices=["subbandmix", "matern32", "subband"],
)
parser.add_argument("--noise_level", default=0.1, type=float)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device_num

from jax import config
from tqdm import tqdm

config.update("jax_enable_x64", True)

import warnings

import bayesnewton
import jax.numpy as jnp
import numpy as np

from milsensors import data_helper, models
from milsensors.kernels import SubbandMatern32
from milsensors.train_and_eval import eval_model, training

N_sites_to_try = args.N_sites_to_try
N_runs = args.N_runs
dataset = args.dataset
data_file = args.data_file
N_t = args.N_t


def inference_loop(seed, N_inducing, N_t):
    """
    Run a single inference loop with a specific number of inducing points

    Args:
        seed: Random seed for reproducibility
        N_inducing: Number of inducing points to use
        N_t: Number of time points

    Returns:
        Various model parameters, metrics, and results from training and evaluation
    """
    print("Loading Data")
    # First, make the dataset
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
        seed, -1, N_t, dataset=dataset, obs_noise=0.0, data_file=data_file
    )

    # Next, make the spatiotemporal grid
    X, Y, R, t, X_t, Y_t, R_t, t_t = data_helper.make_grid(
        X, Y, X_t, Y_t, air_temp_timeseries, N_t, mus, stds, N_sites
    )

    print("MUS", mus)
    print("STDS", stds)

    print("Making Models")
    var_f = 0.75
    len_space = 2.65
    # Create the model for interpolation
    if args.spatial_kern == "sepmatern32":
        kern_space_x = bayesnewton.kernels.Matern32(
            variance=var_f, lengthscale=len_space
        )
        kern_space_y = bayesnewton.kernels.Matern32(
            variance=var_f, lengthscale=len_space
        )
        kern_space = bayesnewton.kernels.Separable([kern_space_x, kern_space_y])
    elif args.spatial_kern == "matern32":
        kern_space = bayesnewton.kernels.Matern32(variance=var_f, lengthscale=len_space)
    else:
        raise NotImplementedError

    # Configure temporal kernel based on command line argument
    if args.temp_kern == "subbandmix":
        kern_time = bayesnewton.kernels.Sum(
            [
                SubbandMatern32(
                    variance=0.5,
                    lengthscale=100.0,
                    radial_frequency=2 * np.pi / 24.0,
                ),
                bayesnewton.kernels.Matern32(variance=1.0, lengthscale=5.0),
            ]
        )
    elif args.temp_kern == "matern32":
        kern_time = bayesnewton.kernels.Matern12(variance=0.3, lengthscale=1.0)
    elif args.temp_kern == "subband":
        kern_time = SubbandMatern32(
            variance=0.6, lengthscale=24.0, radial_frequency=2 * np.pi / 24.0
        )
    else:
        raise NotImplementedError

    # Initialize inducing points using k-means
    z_init = jnp.array(kmeans2(R[0, ...], N_inducing, minit="points")[0])
    z_init_copy = z_init.copy()

    # Create the model
    model = models.make_model(
        N_inducing,
        X,
        Y,
        R,
        t,
        args.noise_level**2 / stds[3] ** 2,
        kern_time,
        kern_space,
        parallel=True,
        z_init=z_init,
    )

    print("X SHAPE", X.shape)

    for k, v in (model.vars()).items():
        print(k, v.shape)

    print("Training Models")
    model = training(model, verbose=True, iters=150, lr_adam=2e-1)

    print("Evaluating Models")
    # Evaluate model performance
    (rmse, nlpd, errors_gt_1, temps_gt_30, posterior_mean, posterior_var) = eval_model(
        model,
        t_t,
        R_t,
        mus,
        stds,
        air_temp_timeseries,
        N_t,
        Y_t,
    )

    print("RMSE", rmse, " | NLPD", nlpd)
    print(
        "Errors > 1°C: ",
        np.sum(errors_gt_1) / len(errors_gt_1),
        " | Errors > 1°C at temps > 30°C: ",
        np.sum(errors_gt_1 * temps_gt_30) / np.sum(temps_gt_30),
    )

    # Return all relevant data for analysis
    return (
        X,
        Y,
        R,
        t_t,
        X_t,
        Y_t,
        R_t,
        t_t,
        air_temp_timeseries,
        N_t,
        mus,
        stds,
        N_sites,
        model,
        rmse,
        nlpd,
        errors_gt_1,
        temps_gt_30,
        posterior_mean,
        posterior_var,
        sample_points,
        z_init_copy,
    )


if __name__ == "__main__":
    # Initialize storage for results
    rmses = []
    nlpds = []
    zs = []
    errors_gt_1s = []
    errors_gt_1_at_30s = []
    nat1s = []
    nat2s = []
    kernel_params = []
    post_covs = []
    post_means = []
    z_inits = []

    # Run experiments for each number of inducing points and each random seed
    for n in tqdm(range(N_runs)):
        for n_sites in tqdm(N_sites_to_try, leave=False):
            print(n_sites)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                (
                    X,
                    Y,
                    R,
                    t_t,
                    X_t,
                    Y_t,
                    R_t,
                    t_t,
                    air_temp_timeseries,
                    N_t,
                    mus,
                    stds,
                    N_sites,
                    model,
                    rmse,
                    nlpd,
                    errors_gt_1,
                    temps_gt_30,
                    posterior_mean,
                    posterior_var,
                    sample_points,
                    z_init,
                ) = inference_loop(n, n_sites, N_t)

            print("XT SHAPE", X_t.shape)

            # Store results
            z_inits.append(z_init)
            zs.append(np.asarray(model.kernel.z.value))

            pseudo_y, pseudo_var = model.compute_full_pseudo_lik()

            rmses.append(rmse)
            nlpds.append(nlpd)
            errors_gt_1s.append(np.sum(errors_gt_1) / len(errors_gt_1))
            errors_gt_1_at_30s.append(
                np.sum(errors_gt_1 * temps_gt_30) / np.sum(temps_gt_30)
            )
            nat1s.append(pseudo_y)
            nat2s.append(pseudo_var)

            # Store kernel parameters based on spatial kernel type
            if args.spatial_kern == "sepmatern32":
                kernel_params.append(
                    [
                        model.kernel.temporal_kernel.kernel0.lengthscale,
                        model.kernel.temporal_kernel.kernel0.radial_frequency,
                        model.kernel.temporal_kernel.kernel0.variance,
                        model.kernel.temporal_kernel.kernel1.lengthscale,
                        model.kernel.temporal_kernel.kernel1.variance,
                        model.kernel.spatial_kernel.kernel0.variance,
                        model.kernel.spatial_kernel.kernel0.lengthscale,
                        model.kernel.spatial_kernel.kernel1.variance,
                        model.kernel.spatial_kernel.kernel1.lengthscale,
                        model.likelihood.variance,
                    ]
                )
            else:
                kernel_params.append(
                    [
                        model.kernel.temporal_kernel.kernel0.lengthscale,
                        model.kernel.temporal_kernel.kernel0.radial_frequency,
                        model.kernel.temporal_kernel.kernel0.variance,
                        model.kernel.temporal_kernel.kernel1.lengthscale,
                        model.kernel.temporal_kernel.kernel1.variance,
                        model.kernel.spatial_kernel.variance,
                        model.kernel.spatial_kernel.lengthscale,
                        model.likelihood.variance,
                    ]
                )

            post_covs.append(model.get_variational_cov())
            post_means.append(model.get_variational_mean())

    # Reshape results for analysis
    errors_gt_1s = np.array(errors_gt_1s).reshape((N_runs, len(N_sites_to_try)))
    errors_gt_1_at_30s = np.array(errors_gt_1_at_30s).reshape(
        (N_runs, len(N_sites_to_try))
    )

    # If results doesn't exist, create the results directory
    if not os.path.exists("results"):
        os.makedirs("results")

    # Save results to files
    np.savez_compressed(
        f"results/N_fixed_optimal_{dataset}_{args.temp_kern}_{args.noise_level}_"
        + "_".join([str(n_sites) for n_sites in N_sites_to_try]),
        errors_gt_1s=errors_gt_1s,
        errors_gt_1_at_30s=errors_gt_1_at_30s,
    )

    # Save inducing points
    with open(
        f"results/N_fixed_optimal_Z_{dataset}_{args.temp_kern}_{args.noise_level}_"
        + "_".join([str(n_sites) for n_sites in N_sites_to_try]),
        "wb",
    ) as handle:
        pickle.dump(zs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save initial inducing points
    with open(
        f"results/N_fixed_init_Z_{dataset}_{args.temp_kern}_{args.noise_level}_"
        + "_".join([str(n_sites) for n_sites in N_sites_to_try]),
        "wb",
    ) as handle:
        pickle.dump(z_inits, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save natural parameters
    with open(
        f"results/N_fixed_optimal_nat1_{dataset}_{args.temp_kern}_{args.noise_level}_"
        + "_".join([str(n_sites) for n_sites in N_sites_to_try]),
        "wb",
    ) as handle:
        pickle.dump(nat1s, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(
        f"results/N_fixed_optimal_nat2_{dataset}_{args.temp_kern}_{args.noise_level}_"
        + "_".join([str(n_sites) for n_sites in N_sites_to_try]),
        "wb",
    ) as handle:
        pickle.dump(nat2s, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save kernel hyperparameters
    with open(
        f"results/N_fixed_optimal_kernel_hypers_{dataset}_{args.temp_kern}_{args.noise_level}_"
        + "_".join([str(n_sites) for n_sites in N_sites_to_try]),
        "wb",
    ) as handle:
        pickle.dump(kernel_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save posterior covariances
    with open(
        f"results/N_fixed_optimal_post_covs_{dataset}_{args.temp_kern}_{args.noise_level}_"
        + "_".join([str(n_sites) for n_sites in N_sites_to_try]),
        "wb",
    ) as handle:
        pickle.dump(post_covs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save posterior means
    with open(
        f"results/N_fixed_optimal_post_means_{dataset}_{args.temp_kern}_{args.noise_level}_"
        + "_".join([str(n_sites) for n_sites in N_sites_to_try]),
        "wb",
    ) as handle:
        pickle.dump(post_means, handle, protocol=pickle.HIGHEST_PROTOCOL)
