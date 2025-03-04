"""
Gaussian Process model implementations for spatiotemporal data.

This module provides custom Gaussian Process model implementations that extend
the BayesNewton library, specifically designed for spatiotemporal modeling.
It includes:
- MarkovVariationalGP: A variational GP model using filtering and smoothing, based on the BayesNewton implementation, but allowing for prediction-time specification of variational parameters and pseudo-likelihood parameters.
- Helper functions to create different types of GP models with various kernels
"""

import bayesnewton
import objax
from scipy.cluster.vq import kmeans2
import math
import numpy as np
import kernels
import jax.numpy as jnp
import jax
from bayesnewton.utils import transpose, diag


class MarkovVariationalGP(
    bayesnewton.inference.VariationalInference,
    bayesnewton.basemodels.MarkovGaussianProcess,
):
    """
    Markov variational Gaussian process: a VGP where the posterior is computed via
    (spatio-temporal) filtering and smoothing [1]. Based on the BayesNewton implementation, but allowing for prediction-time specification of variational parameters and pseudo-likelihood parameters.
    
    Args:
        kernel: A kernel object
        likelihood: A likelihood object
        X: Inputs
        Y: Observations
        R: Spatial inputs
        parallel: Boolean determining whether to run parallel filtering

    References:
        [1] Chang, Wilkinson, Khan, Solin: Fast Variational Learning in State Space Gaussian Process Models, MLSP 2020
    """

    def __init__(self, kernel, likelihood, X, Y, R=None, parallel=None):
        super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel)

    def predict_y(
        self, X, R=None, cubature=None, pseudo_lik_params=None, variational_params=None
    ):
        """
        Predict y at new test locations X
        
        Args:
            X: Test input locations
            R: Test spatial locations
            cubature: Optional cubature method for non-Gaussian likelihoods
            pseudo_lik_params: Optional pre-computed pseudo-likelihood parameters
            variational_params: Optional pre-computed variational parameters
            
        Returns:
            mean_y: Predicted mean values
            var_y: Predicted variance values
        """
        mean_f, var_f = self.predict(
            X,
            R,
            pseudo_lik_params=pseudo_lik_params,
            variational_params=variational_params,
        )
        mean_f = mean_f.reshape(mean_f.shape[0], -1, 1)
        if not isinstance(
            self.likelihood, bayesnewton.likelihoods.MultiLatentLikelihood
        ):
            var_f = var_f.reshape(var_f.shape[0], -1, 1)
        mean_y, var_y = jax.vmap(self.likelihood.predict, (0, 0, None))(
            mean_f, var_f, cubature
        )
        return jnp.squeeze(mean_y), jnp.squeeze(var_y)

    def predict(self, X=None, R=None, pseudo_lik_params=None, variational_params=None):
        """
        Predict at new test locations X
        
        Args:
            X: Test input locations
            R: Test spatial locations
            pseudo_lik_params: Optional pre-computed pseudo-likelihood parameters
            variational_params: Optional pre-computed variational parameters
            
        Returns:
            test_mean: Predicted mean values
            test_var: Predicted variance values
        """
        if X is None:
            X = self.X
        elif len(X.shape) < 2:
            X = X[:, None]
        if R is None:
            R = X[:, 1:]
        X = X[:, :1]  # take only the temporal component

        H = self.kernel.measurement_model()

        if variational_params is None:
            if pseudo_lik_params is None:
                pseudo_y, pseudo_var = self.compute_full_pseudo_lik()
            else:
                pseudo_y, pseudo_var = (
                    pseudo_lik_params  # this deals with the posterior sampling case
                )
            _, (filter_mean, filter_cov) = self.filter(
                self.dt,
                self.kernel,
                pseudo_y,
                pseudo_var,
                mask=self.mask_pseudo_y,  # mask has no effect here (loglik not used)
                parallel=self.parallel,
            )
            dt = jnp.concatenate([self.dt[1:], jnp.array([0.0])], axis=0)
            smoother_mean, smoother_cov, gain = self.smoother(
                dt,
                self.kernel,
                filter_mean,
                filter_cov,
                return_full=True,
                parallel=self.parallel,
            )

            # add dummy states at either edge
            inf = 1e10 * jnp.ones_like(self.X[0, :1])
            X_aug = jnp.block([[-inf], [self.X[:, :1]], [inf]])

            # predict the state distribution at the test time steps:
            state_mean, state_cov = self.temporal_conditional(
                X_aug, X, smoother_mean, smoother_cov, gain, self.kernel
            )

            # extract function values from the state:
            if self.spatio_temporal:
                # TODO: if R is fixed, only compute B, C once
                B, C = self.kernel.spatial_conditional(X, R, predict=True)
                W = B @ H
                print((H @ state_mean).shape)
                test_mean = W @ state_mean
                test_var = W @ state_cov @ transpose(W) + C
            else:
                test_mean, test_var = H @ state_mean, H @ state_cov @ transpose(H)
        else:
            variational_mean, variational_cov = variational_params
            if self.spatio_temporal:
                # TODO: if R is fixed, only compute B, C once
                B, C = self.kernel.spatial_conditional(X, R, predict=True)
                test_mean = B @ variational_mean
                test_var = B @ variational_cov @ transpose(B) + C
            else:
                test_mean, test_var = H @ state_mean, H @ state_cov @ transpose(H)

        # Deal with spatio-temporal case (discard spatial covariance)
        if (
            self.spatio_temporal
        ):
            test_var = diag(test_var)
        return jnp.squeeze(test_mean), jnp.squeeze(test_var)

    def get_variational_cov(self, X=None, R=None):
        """Get the variational covariance at locations X, R"""
        return self.get_variational_params(X, R)[1]

    def get_variational_mean(self, X=None, R=None):
        """Get the variational mean at locations X, R"""
        return self.get_variational_params(X, R)[0]

    def get_variational_params(self, X=None, R=None):
        """
        Get the variational parameters at locations X, R
        
        Args:
            X: Input locations
            R: Spatial locations
            
        Returns:
            variational_mean: Mean of the variational distribution
            variational_cov: Covariance of the variational distribution
        """
        if X is None:
            X = self.X
        elif len(X.shape) < 2:
            X = X[:, None]
        if R is None:
            R = X[:, 1:]
        X = X[:, :1]  # take only the temporal component

        pseudo_y, pseudo_var = self.compute_full_pseudo_lik()

        _, (filter_mean, filter_cov) = self.filter(
            self.dt,
            self.kernel,
            pseudo_y,
            pseudo_var,
            mask=self.mask_pseudo_y,  # mask has no effect here (loglik not used)
            parallel=self.parallel,
        )
        dt = jnp.concatenate([self.dt[1:], jnp.array([0.0])], axis=0)
        smoother_mean, smoother_cov, gain = self.smoother(
            dt,
            self.kernel,
            filter_mean,
            filter_cov,
            return_full=True,
            parallel=self.parallel,
        )

        # add dummy states at either edge
        inf = 1e10 * jnp.ones_like(self.X[0, :1])
        X_aug = jnp.block([[-inf], [self.X[:, :1]], [inf]])

        # predict the state distribution at the test time steps:
        state_mean, state_cov = self.temporal_conditional(
            X_aug, X, smoother_mean, smoother_cov, gain, self.kernel
        )

        # extract function values from the state:
        H = self.kernel.measurement_model()
        return H @ state_mean, H @ state_cov @ transpose(H)

    def negative_log_predictive_density(
        self, X, Y, R=None, cubature=None, variational_params=None
    ):
        """
        Calculate the negative log predictive density at test locations
        
        Args:
            X: Test input locations
            Y: Test observations
            R: Test spatial locations
            cubature: Optional cubature method for non-Gaussian likelihoods
            variational_params: Optional pre-computed variational parameters
            
        Returns:
            nlpd: Negative log predictive density
        """
        predict_mean, predict_var = self.predict(
            X, R, variational_params=variational_params
        )
        if Y.ndim < 2:
            Y = Y.reshape(-1, 1)
        if isinstance(self.likelihood, bayesnewton.likelihoods.MultiLatentLikelihood):
            predict_mean = predict_mean[..., None]
        elif (predict_mean.ndim > 1) and (
            predict_mean.shape[1] != Y.shape[1]
        ):  # multi-latent case
            predict_mean, predict_var = predict_mean[..., None], predict_var[
                ..., None
            ] * np.eye(predict_var.shape[1])
        else:
            predict_mean, predict_var = predict_mean.reshape(
                -1, 1, 1
            ), predict_var.reshape(-1, 1, 1)
        log_density = jax.vmap(self.likelihood.log_density, (0, 0, 0, None))(
            Y.reshape(predict_mean.shape[0], -1, 1), predict_mean, predict_var, cubature
        )
        return -jnp.nanmean(log_density)


def make_model(
    N_obs_pts,
    X,
    Y,
    R,
    t,
    var_y,
    kern_time,
    kern_space,
    parallel=False,
    z_fixed=None,
    opt_z=True,
    z_init=None,
):
    """
    Create a spatiotemporal GP model
    
    Args:
        N_obs_pts: Number of observation points
        X: Input locations
        Y: Observations
        R: Spatial locations
        t: Time points
        var_y: Observation noise variance
        kern_time: Temporal kernel
        kern_space: Spatial kernel
        parallel: Whether to use parallel filtering
        z_fixed: Fixed inducing points
        opt_z: Whether to optimize inducing point locations
        z_init: Initial inducing point locations
        
    Returns:
        model: The constructed GP model
    """
    num_z_space = N_obs_pts

    if z_init is None:
        z_train = jnp.array(kmeans2(R[0, ...], num_z_space, minit="points")[0])
    else:
        z_train = z_init

    if z_fixed is None:
        kern = bayesnewton.kernels.SpatioTemporalKernel(
            temporal_kernel=kern_time,
            spatial_kernel=kern_space,
            z=z_train,
            sparse=True,
            opt_z=opt_z,
            conditional="Full",
        )
    else:
        kern = kernels.SpatioTemporalPartialOptKernel(
            temporal_kernel=kern_time,
            spatial_kernel=kern_space,
            z_fixed=z_fixed,
            z_train=z_train,
            sparse=True,
            conditional="Full",
        )

    lik = bayesnewton.likelihoods.Gaussian(variance=var_y)
    # lik = bayesnewton.likelihoods.StudentsT(scale=math.sqrt(var_y))

    model = MarkovVariationalGP(
        kernel=kern, likelihood=lik, X=t, R=R, Y=Y, parallel=parallel
    )

    return model


def make_model_sep_matern(N_obs_pts, X, Y, R, t, var_y):
    """
    Create a GP model with separable Matern kernels
    
    Args:
        N_obs_pts: Number of observation points
        X: Input locations
        Y: Observations
        R: Spatial locations
        t: Time points
        var_y: Observation noise variance
        
    Returns:
        model: The constructed GP model with separable Matern kernels
    """
    var_f = 1.0

    len_space = 0.2
    len_time = 1.0

    kern_time = bayesnewton.kernels.Matern12(
        variance=var_f,
        lengthscale=len_time,
    )

    kern_space_x = bayesnewton.kernels.Matern32(variance=var_f, lengthscale=len_space)
    kern_space_y = bayesnewton.kernels.Matern32(variance=var_f, lengthscale=len_space)
    kern_space = bayesnewton.kernels.Separable([kern_space_x, kern_space_y])

    return make_model(
        N_obs_pts, X, Y, R, t, var_y, kern_time=kern_time, kern_space=kern_space
    )


def make_model_matern(N_obs_pts, X, Y, R, t, var_y):
    """
    Create a GP model with Matern kernels
    
    Args:
        N_obs_pts: Number of observation points
        X: Input locations
        Y: Observations
        R: Spatial locations
        t: Time points
        var_y: Observation noise variance
        
    Returns:
        model: The constructed GP model with Matern kernels
    """
    var_f = 1.0

    len_space = 0.2
    len_time = 1.0

    kern_time = bayesnewton.kernels.Matern12(
        variance=var_f,
        lengthscale=len_time,
    )

    kern_space = bayesnewton.kernels.Matern32(variance=var_f, lengthscale=len_space)

    return make_model(
        N_obs_pts, X, Y, R, t, var_y, kern_time=kern_time, kern_space=kern_space
    )
