"""
Utilities for MIL-based sensor design with spatiotemporal Gaussian Processes.

This module provides simple, high-level helpers to:
- build default temporal/spatial kernels used in the paper
- train an ST-VGP model on simulator-derived data
- extract inducing points and variational covariance for downstream analysis
"""

from typing import Dict, Optional, Tuple

import bayesnewton
import jax.numpy as jnp
import numpy as np
from scipy.cluster.vq import kmeans2

from milsensors import data_helper, models
from milsensors.kernels import SubbandMatern32
from milsensors.train_and_eval import eval_model, training


def build_default_kernels(
    spatial_kern: str = "matern32",
    temp_kern: str = "subbandmix",
) -> Tuple[bayesnewton.kernels.Kernel, bayesnewton.kernels.Kernel]:
    """Construct default temporal and spatial kernels.

    spatial_kern in {"matern32", "sepmatern32"}; temp_kern in {"subbandmix", "matern32", "subband"}.
    """
    var_f = 0.75
    len_space = 2.65

    if spatial_kern == "sepmatern32":
        kern_space_x = bayesnewton.kernels.Matern32(
            variance=var_f, lengthscale=len_space
        )
        kern_space_y = bayesnewton.kernels.Matern32(
            variance=var_f, lengthscale=len_space
        )
        kern_space = bayesnewton.kernels.Separable([kern_space_x, kern_space_y])
    elif spatial_kern == "matern32":
        kern_space = bayesnewton.kernels.Matern32(variance=var_f, lengthscale=len_space)
    else:
        raise NotImplementedError

    if temp_kern == "subbandmix":
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
    elif temp_kern == "matern32":
        kern_time = bayesnewton.kernels.Matern12(variance=0.3, lengthscale=1.0)
    elif temp_kern == "subband":
        kern_time = SubbandMatern32(
            variance=0.6, lengthscale=24.0, radial_frequency=2 * np.pi / 24.0
        )
    else:
        raise NotImplementedError

    return kern_time, kern_space


def init_inducing_points(R: jnp.ndarray, num_inducing: int) -> jnp.ndarray:
    """Initialize inducing points with k-means on the first time slice of R."""
    return jnp.array(kmeans2(R[0, ...], num_inducing, minit="points")[0])


def extract_inducing_and_variational(
    model: models.MarkovVariationalGP,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return z, variational_mean, variational_cov, pseudo_y, pseudo_var."""
    z = np.asarray(model.kernel.z.value)
    pseudo_y, pseudo_var = model.compute_full_pseudo_lik()
    variational_mean = model.get_variational_mean()
    variational_cov = model.get_variational_cov()
    return z, variational_mean, variational_cov, pseudo_y, pseudo_var


def train_and_extract(
    seed: int,
    num_inducing: int,
    num_time_steps: int,
    dataset: str = "phoenix",
    data_file: str = "data/WRF_data_2013",
    spatial_kern: str = "matern32",
    temp_kern: str = "subbandmix",
    noise_level: float = 0.1,
    iters: int = 150,
    lr_adam: float = 0.2,
    parallel: bool = True,
    kernel_time: Optional[bayesnewton.kernels.Kernel] = None,
    kernel_space: Optional[bayesnewton.kernels.Kernel] = None,
    z_fixed: Optional[jnp.ndarray] = None,
    opt_z: bool = True,
) -> Dict[str, object]:
    """Train a model and return inducing points and variational quantities.

    Returns a dict with keys: model, z_init, z, variational_mean, variational_cov,
    pseudo_y, pseudo_var, rmse, nlpd, errors_gt_1, temps_gt_30, posterior_mean,
    posterior_var, mus, stds, N_sites, sample_points.
    """
    X, Y, X_t, Y_t, air_temp_ts, N_t, mus, stds, N_sites, sample_points = (
        data_helper.create_dataset(
            seed,
            -1,
            num_time_steps,
            dataset=dataset,
            obs_noise=0.0,
            data_file=data_file,
        )
    )
    X, Y, R, t, X_t, Y_t, R_t, t_t = data_helper.make_grid(X, Y, X_t, Y_t)

    # Use provided kernels when given; otherwise build defaults
    if (kernel_time is None) or (kernel_space is None):
        def_time, def_space = build_default_kernels(
            spatial_kern=spatial_kern, temp_kern=temp_kern
        )
        if kernel_time is None:
            kernel_time = def_time
        if kernel_space is None:
            kernel_space = def_space
    z_init = init_inducing_points(R, num_inducing)

    model = models.make_model(
        num_inducing,
        X,
        Y,
        R,
        t,
        noise_level**2 / stds[3] ** 2,
        kernel_time,
        kernel_space,
        parallel=parallel,
        z_init=z_init,
        z_fixed=z_fixed,
        opt_z=opt_z,
    )

    model = training(model, verbose=True, iters=iters, lr_adam=lr_adam)

    rmse, nlpd, errors_gt_1, temps_gt_30, posterior_mean, posterior_var = eval_model(
        model,
        t_t,
        R_t,
        mus,
        stds,
        air_temp_ts,
        N_t,
        Y_t,
    )

    z, variational_mean, variational_cov, pseudo_y, pseudo_var = (
        extract_inducing_and_variational(model)
    )

    return {
        "model": model,
        "z_init": z_init,
        "z": z,
        "variational_mean": variational_mean,
        "variational_cov": variational_cov,
        "pseudo_y": pseudo_y,
        "pseudo_var": pseudo_var,
        "rmse": rmse,
        "nlpd": nlpd,
        "errors_gt_1": errors_gt_1,
        "temps_gt_30": temps_gt_30,
        "posterior_mean": posterior_mean,
        "posterior_var": posterior_var,
        "mus": mus,
        "stds": stds,
        "N_sites": N_sites,
        "sample_points": sample_points,
    }
