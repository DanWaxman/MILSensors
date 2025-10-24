"""
Training and evaluation utilities for Gaussian Process models.

This module provides functions for training Gaussian Process models using a combination
of Newton's method for variational parameters and Adam for hyperparameters, as well as
functions for evaluating model performance on test data.

The module includes:
- training: Function to train a GP model with optional batch training and early stopping
- eval_model: Function to evaluate a trained model on test data, computing metrics like RMSE and NLPD
"""

import time
from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np
import objax
from tqdm import tqdm


def training(
    model: object,
    iters: int = 150,
    lr_adam: float = 0.1,
    lr_newton: float = 1.0,
    verbose: bool = True,
    batch_size: Optional[int] = None,
    N: Optional[int] = None,
    early_stopping: bool = False,
    optimize_all: bool = True,
    verbose_iter: int = 50,
) -> object:
    """Train a Gaussian Process model using a combination of Newton's method and Adam.

    :param model: The GP model to train
    :param iters: Maximum number of training iterations
    :param lr_adam: Learning rate for Adam optimizer (for hyperparameters)
    :param lr_newton: Learning rate for Newton's method (for variational parameters)
    :param verbose: Whether to print training progress
    :param batch_size: Size of mini-batches (if None, use full dataset)
    :param N: Total number of data points (required if using mini-batches)
    :param early_stopping: Whether to stop training if loss increases
    :param optimize_all: Whether to optimize all parameters (if False, kernel parameters are fixed)
    :param verbose_iter: How often to print training progress
    :returns: The trained model
    """
    # Get model variables, but not kernel parameters
    if optimize_all:
        model_vars = model.vars()
    else:
        model_vars = objax.VarCollection(
            {
                k: v
                for k, v in model.vars().items()
                if ("kernel" not in k.lower() and "variance" not in k.lower())
            }
        )
    opt_hypers = objax.optimizer.Adam(model_vars)
    energy = objax.GradValues(model.energy, model_vars)

    if batch_size is None or N is None:

        @objax.Function.with_vars(model_vars + opt_hypers.vars())
        def train_op():
            # perform inference and update variational params
            model.inference(lr=lr_newton)
            dE, E = energy()  # compute energy and its gradients w.r.t. hypers
            opt_hypers(lr_adam, dE)
            return E

    else:

        @objax.Function.with_vars(model_vars + opt_hypers.vars())
        def train_op():
            batch = np.random.permutation(N)[:batch_size]
            model.inference(
                lr=lr_newton, batch_ind=batch
            )  # perform inference and update variational params
            dE, E = energy(
                batch_ind=batch
            )  # compute energy and its gradients w.r.t. hypers
            opt_hypers(lr_adam, dE)
            return E

    train_op = objax.Jit(train_op)

    t0 = time.time()
    last_loss = 1e10
    if verbose:
        print("############\nStarting Training Loop\n############")
    for i in range(1, iters + 1):
        loss = train_op()
        if verbose and i % verbose_iter == 0:
            print("iter %2d: energy: %1.4f" % (i, loss[0]))
            # Commented out kernel parameter printing

        if loss[0] > last_loss and early_stopping:
            print("Early stopping!")
            break

        last_loss = loss[0]

    t1 = time.time()
    if verbose:
        print("optimisation time: %2.2f secs" % (t1 - t0))
        avg_time_taken = (t1 - t0) / iters
        print("average iter time: %2.2f secs" % avg_time_taken)

    return model


def eval_model(
    model: object,
    t_t: jnp.ndarray,
    R_t: jnp.ndarray,
    mus: np.ndarray,
    stds: np.ndarray,
    air_temp_timeseries: np.ndarray,
    N_t: int,
    Y_t: np.ndarray,
    N_minibatch: int = 200,
    pseudo_lik_params: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    variational_params: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a trained GP model on test data.

    :param model: The trained GP model
    :param t_t: Test time points
    :param R_t: Test spatial locations
    :param mus: Mean values for normalization
    :param stds: Standard deviations for normalization
    :param air_temp_timeseries: Original temperature time series data
    :param N_t: Number of test time points
    :param Y_t: Test observations
    :param N_minibatch: Size of mini-batches for prediction
    :param pseudo_lik_params: Optional pre-computed pseudo-likelihood parameters
    :param variational_params: Optional pre-computed variational parameters
    :returns: Tuple containing (rmse, nlpd, errors_gt_1, temps_gt_30, posterior_mean, posterior_var)
        where rmse is root mean squared error, nlpd is negative log predictive density,
        errors_gt_1 is boolean array indicating errors greater than 1°C,
        temps_gt_30 is boolean array indicating temperatures greater than 30°C,
        posterior_mean is predicted mean values, and posterior_var is predicted variance values
    """
    N_rt = R_t.shape[1]

    posterior_mean = []
    posterior_var = []
    nlpd = 0.0
    for batch_idx in tqdm(range(int(np.ceil(t_t.shape[0] / N_minibatch))), leave=False):
        s = batch_idx * N_minibatch
        e = s + N_minibatch
        p_m, p_v = model.predict_y(
            X=t_t[s:e, ...],
            R=R_t[s:e, ...],
            pseudo_lik_params=pseudo_lik_params,
            variational_params=variational_params,
        )
        posterior_mean.append(p_m)
        posterior_var.append(p_v)

        nlpd += model.negative_log_predictive_density(
            X=t_t[s:e, ...],
            R=R_t[s:e, ...],
            Y=Y_t[s:e, ...],
            variational_params=variational_params,
        )
    posterior_mean = np.concatenate(posterior_mean)
    posterior_var = np.concatenate(posterior_var)

    # Calculate errors greater than 1°C
    errors_gt_1 = (
        stds[3]
        * np.abs(
            np.reshape(posterior_mean, (N_t, N_rt)) - air_temp_timeseries[:N_t, :, 3]
        ).reshape((-1, 1))
        > 1
    )
    # Calculate temperatures greater than 30°C
    temps_gt_30 = (mus[3] + stds[3] * air_temp_timeseries[:N_t, :, 3]).reshape(
        (-1, 1)
    ) > 30

    # Calculate RMSE
    rmse = stds[3] * np.sqrt(
        np.mean((np.squeeze(Y_t) - np.squeeze(posterior_mean)) ** 2)
    )

    return rmse, nlpd, errors_gt_1, temps_gt_30, posterior_mean, posterior_var
