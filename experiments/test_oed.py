import argparse

import bayesnewton
import data_helper
import models
import numpy as np
from bayes_oed import get_posterior_predictive_logdet, get_posterior_predictive_tr
from kernels import SubbandMatern32
from scipy.cluster.vq import kmeans2
from train_and_eval import training

parser = argparse.ArgumentParser(description="Gaussian Process example")


##### Load the full dataset
N_t = 24 * 92  # number of time instants to consider
N_t_start = 0  # offset
seed = 0
N_inducing = 5  # number of inducing points

X, Y, X_test, Y_test, air_temp_timeseries, N_t, mus, stds, N_sites, sample_points = (
    data_helper.create_dataset(
        seed,
        -1,
        N_t,
        dataset="phoenix",
        obs_noise=0.0,
        data_file="../data/WRF_data_2013_phoenix.npz",
    )
)

X, Y, R, t, X_t, Y_t, R_t, t_t = data_helper.make_grid(
    X, Y, X_test, Y_test, air_temp_timeseries, N_t, mus, stds, N_sites
)

R_full = np.copy(R)


####### Apply k means with N_inducing centroids and select the closest grid points
centroids = kmeans2(R[0, ...], N_inducing, minit="points")[0]

sample_points = []
# Find the nearest point in the training data for each optimal point
for centroid in centroids:
    print(centroid)
    sample_points.append(
        np.argmin((R[0, :, 0] - centroid[0]) ** 2 + (R[0, :, 1] - centroid[1]) ** 2)
    )

sample_points = np.array(sample_points)


# Select the first month
X = air_temp_timeseries[N_t_start : N_t_start + 24 * 30, sample_points, 0:3]
Y = air_temp_timeseries[N_t_start : N_t_start + 24 * 30, sample_points, 3]

X = X.reshape(24 * 30 * N_inducing, 3)
Y = Y.reshape(24 * 30 * N_inducing, 1)
print("N_sites", N_sites)
X_t = air_temp_timeseries[-24 * 30 :, :, 0:3].reshape(24 * 30 * N_sites, 3)
Y_t = air_temp_timeseries[-24 * 30 :, :, 3].reshape(24 * 30 * N_sites, 1)

t, R, Y = bayesnewton.utils.create_spatiotemporal_grid(X, Y)
t_t, R_t, Y_t = bayesnewton.utils.create_spatiotemporal_grid(X_t, Y_t)


###### Make the models
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

kern_space = bayesnewton.kernels.Matern32(variance=0.75, lengthscale=2.65)


model = models.make_model(
    N_inducing,
    X,
    Y,
    R,
    t,
    0.1 / stds[3],
    kern_time,
    kern_space,
    opt_z=False,
    z_init=R[0],
)


#### Train the models

model = training(model, verbose=True, verbose_iter=20, lr_adam=0.1, iters=20)


#### Make predictions on the third month

inputs_to_try = R_full[0]  # this should be provided by the optimization routine
times = 24 * 30


def objective_trace(inputs_to_try):  # MINIMIZE!
    R_full_for_prediction = np.stack([inputs_to_try] * times, axis=0)
    # trace = get_posterior_predictive_tr(model,np.arange(24*92 - times, 24*92), R_full_for_prediction)

    trace = get_posterior_predictive_tr(
        model, np.arange(24 * 30, 24 * 30 + 24 * 30), R_full_for_prediction
    )

    print(np.sum(trace))


def objective_logdet(inputs_to_try):  # MAXIMIZE!
    R_full_for_prediction = np.stack([inputs_to_try] * times, axis=0)
    # trace = get_posterior_predictive_tr(model,np.arange(24*92 - times, 24*92), R_full_for_prediction)

    logdet = get_posterior_predictive_logdet(
        model, np.arange(24 * 30, 24 * 30 + 24 * 30), R_full_for_prediction
    )

    print(np.sum(logdet))


for k in range(5):
    inputs_to_try = -2.5 + 5 * np.random.rand(N_inducing, 2)
    print(inputs_to_try)
