import os
from milsensors import models
from milsensors.kernels import SubbandMatern32
from milsensors.train_and_eval import training
import bayesnewton
from milsensors import data_helper
import numpy as np

import argparse
import re

import jax

parser = argparse.ArgumentParser()
parser.add_argument(
    "--datafile",
    type=str,
    required=True,
    help="Path to the .npz file containing seed and indices_init_train",
)
args = parser.parse_args()


# Try to switch to MPS if available
if "mps" in jax.devices()[0].platform:
    print("Using MPS device.")
else:
    print("MPS device not available, using default device.")
    # Set JAX to use the MPS device if available
    if any(device.platform == "mps" for device in jax.devices()):
        jax.config.update("jax_platform_name", "mps")
        print("Switched to MPS device.")
    else:
        print("MPS device not available, using default device.")
print("Running with parameters:", args)

# Extract N_init and N_to_add from the datafile name
filename = os.path.basename(args.datafile)
match = re.search(r"N_init_(\d+)_N_added_(\d+)", filename)
if match:
    N_init = int(match.group(1))
    N_to_add = int(match.group(2))
else:
    raise ValueError("Could not extract N_init and N_added from the datafile name.")


if "mes" in filename.lower():
    method_type = "mes"
elif "mpv" in filename.lower():
    method_type = "mpv"
else:
    raise ValueError("Could not determine method type from the datafile name.")


data = np.load(f"results/{args.datafile}")
seed = int(data["seed"])
indices_init_train = list(data["indices_init_train"])

print("Running with", f"N_init: {N_init}, N_to_add: {N_to_add}, seed: {seed}")


####### REFIT with all N_init and N_to_add points
print(
    "Retrain GP with all points and evaluate on August"
)  # Select data for the last month (August 2013)
N_t = 24 * 92  # number of time instants to consider (We load all the dataset again)
N_t_start = 0  # offset

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


assert len(indices_init_train) == N_init + N_to_add, (
    "Number of initial training points does not match N_init + N_to_add"
)
print(
    "Number of Initial training points after adding new points:",
    len(indices_init_train),
)

# Remove duplicated indices from indices_init_train
indices_init_train = list(sorted(set(indices_init_train)))
print(
    "Number of Initial training points after removing duplicates:",
    len(indices_init_train),
)


X = air_temp_timeseries[24 * 92 - 24 * 30 :, indices_init_train, 0:3]
Y = air_temp_timeseries[24 * 92 - 24 * 30 :, indices_init_train, 3]

assert X.shape[0] == 24 * 30, "X should have 24*30 time points"
print("Length of Time Series for Final Training:", X.shape)

N_total = len(indices_init_train)
X = X.reshape(24 * 30 * N_total, 3)
Y = Y.reshape(24 * 30 * N_total, 1)
X_t = air_temp_timeseries[24 * 92 - 24 * 30 :, :, 0:3].reshape(24 * 30 * N_sites, 3)
Y_t = air_temp_timeseries[24 * 92 - 24 * 30 :, :, 3].reshape(24 * 30 * N_sites, 1)
assert X_t.shape[0] == 24 * 30 * N_sites, "X_t should have 24*30 * N_sites time points"
print("Length of Time Series for Final Evaluation:", X_t.reshape(-1, N_sites, 3).shape)


t, R, Y = bayesnewton.utils.create_spatiotemporal_grid(X, Y)
t_t, R_t, Y_t = bayesnewton.utils.create_spatiotemporal_grid(X_t, Y_t)
# assert R[0,:, 0].shape[0] == N_total
if R[0].shape[0] != N_total:
    print(
        "Warning: Number of training points does not match N_total. This might because two optimized points were matched to the same grid."
    )


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
    N_total,
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
model = training(model, verbose=True, verbose_iter=20, lr_adam=0.1, iters=200)

mean, covar = model.predict(
    np.arange(2208.0 - 24 * 30, 2208), R_full[: 24 * 30], return_diag_cov_only=True
)

print("Length of Mean for evaluation:", mean.shape)
assert mean.shape == (24 * 30, N_sites), "Mean shape does not match expected shape"


rmse = stds[3] * np.sqrt(np.mean((np.squeeze(Y_t) - np.squeeze(mean)) ** 2))
nlpd = model.negative_log_predictive_density(X=t_t, R=R_t, Y=Y_t)

final_pts = R[0] * stds[1:3] + mus[1:3]
print(final_pts)

# store rmse, nlpd and the locations of the initial training points and the new points
os.makedirs("results", exist_ok=True)
np.savez(
    f"results/error_{method_type}_phoenix_N_init_{N_init}_N_added_{N_to_add}_seed_{seed}.npz",
    rmse=rmse,
    nlpd=nlpd,
    indices_init_train=indices_init_train,
    final_pts=final_pts,
)
