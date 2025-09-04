import sys
import os
from milsensors import models
from milsensors.kernels import SubbandMatern32
from milsensors.train_and_eval import training
import bayesnewton
from scipy.cluster.vq import kmeans2
from milsensors import data_helper
import numpy as np

from milsensors.bayes_oed import get_posterior_predictive_logdet
from tqdm import trange
from scipy.spatial import ConvexHull  # only needed once
import jax
import jax.numpy as jnp
import argparse

parser = argparse.ArgumentParser(description="Maximum Entropy Sampling (MES)")
parser.add_argument("--N_init", nargs="?", default=5, type=int)
parser.add_argument("--N_to_add", nargs="?", default=5, type=int)
parser.add_argument("--seed", nargs="?", default=0, type=int)
args = parser.parse_args()

if jax.lib.xla_bridge.get_backend().platform == "cpu":
    print("CPU backend detected.")
elif jax.lib.xla_bridge.get_backend().platform == "gpu":
    print("GPU backend detected.")
elif jax.lib.xla_bridge.get_backend().platform == "tpu":
    print("TPU backend detected.")
elif jax.lib.xla_bridge.get_backend().platform == "mps":
    print("MPS backend detected.")
else:
    print(f"Unknown backend: {jax.lib.xla_bridge.get_backend().platform}")

# Try to switch to MPS if available
if "mps" in jax.devices()[0].platform:
    print("Using MPS device.")
else:
    print("MPS device not available, using default device.")

print("Running with parameters:", args)

N_inducing = args.N_init  # number of initial points from June 2013 to train GP on
N_to_add = args.N_to_add  # number of spatial locations to add using either Maximum Entropy Sampling (logdet) or Minumum Predictive Variance (trace)
seed = args.seed  # seed for random number generation


rng_gen = np.random.default_rng(seed)  # random number generator

##### Load the full dataset
N_t = 24 * 92  # number of time instants to consider
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
print("Total number of sites:", R_full[0].shape[0])


####### Apply k means with N_inducing centroids and select the closest grid points
centroids = kmeans2(R_full[0, ...], N_inducing, minit="points")[0]

sample_points = []
for centroid in centroids:
    sample_points.append(  # Find the nearest point in the training data for each optimal point
        np.argmin(
            (R_full[0, :, 0] - centroid[0]) ** 2 + (R_full[0, :, 1] - centroid[1]) ** 2
        )
    )
indices_init_train = (
    sample_points.copy()
)  # store the indices of the initial training points
sample_points = np.array(sample_points)
init_train_points = R[0][
    sample_points
].copy()  # get the actual points of the initial training points

assert len(sample_points) == N_inducing, (
    "Number of initial training points does not match N_inducing"
)
print("Number of Initial training points:", len(init_train_points))

# Select the first month
X = air_temp_timeseries[
    N_t_start : N_t_start + 24 * 30, sample_points, 0:3
]  # ONLY June 2013
Y = air_temp_timeseries[N_t_start : N_t_start + 24 * 30, sample_points, 3]

assert X.shape[0] == 24 * 30, "X should have 24*30 time points"
assert Y.shape[0] == 24 * 30, "Y should have 24*30 time points"

X = X.reshape(24 * 30 * N_inducing, 3)
Y = Y.reshape(24 * 30 * N_inducing, 1)

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
model = training(model, verbose=True, verbose_iter=20, lr_adam=0.1, iters=200)


#### Make predictions on the second month (we need this predictive model to evaluate the objective function)
# inputs_to_try = R_full[0]     # this should be provided by the optimization routine
times = 24 * 30


def objective_neglogdet(
    inputs_to_try,
):  # MINIMIZE! (equivalently, maximize logdet = maximize entropy)
    assert inputs_to_try.shape == (N_to_add, 2), (
        "inputs_to_try should have shape (N_sites, 2)"
    )
    # R_for_prediction = jnp.stack([inputs_to_try] * times, axis=0)
    R_for_prediction = jnp.broadcast_to(inputs_to_try, (times,) + inputs_to_try.shape)

    logdet = get_posterior_predictive_logdet(
        model, jnp.arange(24 * 30, 24 * 30 + 24 * 30), jnp.array(R_for_prediction)
    )
    return -1.0 * jnp.sum(logdet)


# ============================================== #
####### Apply k means with N_to_add centroids and select the closest grid points
centroids = kmeans2(R_full[0, ...], N_to_add, minit="points")[
    0
]  # this is to initialize the points to add

sample_points_otro = []
for centroid in centroids:
    sample_points_otro.append(  # Find the nearest point in the training data for each optimal point
        np.argmin(
            (R_full[0, :, 0] - centroid[0]) ** 2 + (R_full[0, :, 1] - centroid[1]) ** 2
        )
    )
sample_points_otro = np.array(sample_points_otro).reshape(-1)
assert len(sample_points_otro) == N_to_add, (
    "Number of points to add does not match N_to_add"
)
print("Number of points to add:", len(sample_points_otro))
# ================================================== #


####### FOR MES we don't have to retrain the model!!!


def convex_hull(points_np: np.ndarray) -> np.ndarray:
    """Return hull vertices in counter-clockwise order (shape: (H,2))."""
    hull = ConvexHull(points_np)
    return points_np[hull.vertices]  # SciPy guarantees CCW order


def _project_to_polygon(points: jnp.ndarray, hull: jnp.ndarray) -> jnp.ndarray:
    """
    Orthogonally project each 2-D point onto a convex polygon.

    points : (N,2)  JAX array
    hull   : (H,2)  JAX array of polygon vertices in CCW order
    returns: (N,2)  projected points
    """
    a = hull  # (H,2)
    b = jnp.roll(hull, -1, axis=0)  # next vertex for each edge (H,2)
    ab = b - a  # edge vectors               (H,2)
    ab2 = jnp.sum(ab * ab, axis=1)  # |ab|²                      (H,)

    def proj_one(p):
        # ------------ inside-test (all left turns for CCW polygon) ----------
        ap = p - a  # vectors from edge start to p (H,2)
        cross = ab[:, 0] * ap[:, 1] - ab[:, 1] * ap[:, 0]
        inside = jnp.all(cross >= 0)

        # ------------ projection onto each edge-segment ---------------------
        t = jnp.clip(jnp.sum(ap * ab, axis=1) / ab2, 0.0, 1.0)  # (H,)
        cand = a + t[:, None] * ab  # (H,2)
        d2 = jnp.sum((cand - p) ** 2, axis=1)  # (H,)
        p_proj = cand[jnp.argmin(d2)]

        return jax.lax.cond(inside, lambda: p, lambda: p_proj)

    return jax.vmap(proj_one)(points)


# ---- projected gradient-descent driver --------------------------------------
def pgd(
    f,
    pts0_np: np.ndarray,  # initial (N,2) NumPy array
    pts_for_ch,
    lr: float = 1e-2,
    steps: int = 100,
):
    """
    Run projected GD on f over the convex hull of the starting points.

    Returns the final point array (JAX, shape (N,2)).
    """
    # freeze geometry
    hull_np = convex_hull(pts_for_ch)
    hull_jax = jnp.asarray(hull_np)

    # set up optimisation
    pts = jnp.asarray(pts0_np)  # mutable in loop
    val_grad = jax.jit(jax.value_and_grad(f))

    pbar = trange(steps)
    for _ in pbar:
        val, g = val_grad(pts)
        pts = pts - lr * g  # gradient step
        pts = _project_to_polygon(pts, hull_jax)  # projection step
        pbar.set_description(f"Loss: {val:.4f}")
    return pts


# -----------------------------------------------------------------------------


init_points = R_full[0][sample_points_otro]
assert init_points.shape[0] == N_to_add, "Initial points shape does not match N_to_add"
final_pts = pgd(
    objective_neglogdet,
    jnp.asarray(init_points),
    pts_for_ch=R_full[0],
    lr=1e-4,
    steps=200,
)

assert final_pts.shape[0] == N_to_add, "Number of points to add does not match N_to_add"


####### REFIT with all N_init and N_to_add points
print("WE DON'T Retrain GP and evaluate in this script")
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

# add the new points to the initial training points
for centroid in final_pts:
    # print(centroid)
    indices_init_train.append(
        np.argmin(
            (R_full[0, :, 0] - centroid[0]) ** 2 + (R_full[0, :, 1] - centroid[1]) ** 2
        )
    )
indices_init_train = np.array(indices_init_train)
assert len(indices_init_train) == N_inducing + N_to_add, (
    "Number of initial training points does not match N_inducing + N_to_add"
)
print(
    "Number of Initial training points after adding new points:",
    len(indices_init_train),
)


# Save the seed and indices_init_train to a data file
os.makedirs("results", exist_ok=True)
np.savez(
    f"results/mes_phoenix_N_init_{args.N_init}_N_added_{N_to_add}_seed_{seed}.npz",
    seed=seed,
    indices_init_train=indices_init_train,
)


sys.exit(0)


# # Select data for the last month (August 2013)
# X = air_temp_timeseries[ 24*92 - 24*30:, indices_init_train, 0:3]
# Y = air_temp_timeseries[ 24*92 - 24*30:, indices_init_train, 3]

# assert X.shape[0] == 24*30, "X should have 24*30 time points"
# print("Length of Time Series for Final Training:", X.shape)

# N_inducing = len(indices_init_train)
# X = X.reshape(24*30 * N_inducing, 3)
# Y = Y.reshape(24*30 * N_inducing, 1)
# X_t = air_temp_timeseries[24*92 - 24*30:, :, 0:3].reshape(
#     24*30 * N_sites, 3
# )
# Y_t = air_temp_timeseries[24*92 - 24*30:, :, 3].reshape(
#     24*30 * N_sites, 1
# )
# assert X_t.shape[0] == 24*30 * N_sites, "X_t should have 24*30 * N_sites time points"
# print("Length of Time Series for Final Evaluation:", X_t.shape)


# t, R, Y = bayesnewton.utils.create_spatiotemporal_grid(X, Y)
# t_t, R_t, Y_t = bayesnewton.utils.create_spatiotemporal_grid(X_t, Y_t)
# assert R[0,:, 0].shape[0] == N_inducing

# kern_time = bayesnewton.kernels.Sum(
#             [
#                 SubbandMatern32(
#                     variance=0.5,
#                     lengthscale=100.0,
#                     radial_frequency=2 * np.pi / 24.0,
#                 ),
#                 bayesnewton.kernels.Matern32(variance=1.0, lengthscale=5.0),
#             ]
#         )

# kern_space = bayesnewton.kernels.Matern32(variance=0.75, lengthscale=2.65)

# model = models.make_model(
#     N_inducing,
#     X,
#     Y,
#     R,
#     t,
#     0.1 / stds[3],
#     kern_time,
#     kern_space,
#     opt_z=False,
#     z_init=R[0],
# )


# #### Train the models
# model = training(model, verbose=True, verbose_iter=20, lr_adam=0.1, iters=20)

# mean, covar = model.predict(np.arange(2208.0 - 24*30, 2208), R_full[:24*30], return_diag_cov_only=True)

# print("Length of Mean for evaluation:", mean.shape)
# assert mean.shape == (24*30, N_sites), "Mean shape does not match expected shape"


# rmse = stds[3] * np.sqrt(
#     np.mean((np.squeeze(Y_t) - np.squeeze(mean)) ** 2)
# )
# nlpd = model.negative_log_predictive_density(X=t_t, R=R_t, Y=Y_t)

# # store rmse, nlpd and the locations of the initial training points and the new points
# os.makedirs("results", exist_ok=True)
# np.savez(
#     f"results/mes_phoenix_N_init_{args.N_init}_N_added_{N_to_add}_seed_{seed}.npz",
#     rmse=rmse,
#     nlpd=nlpd,
#     indices_init_train=indices_init_train,
#     final_pts=final_pts,
# )
