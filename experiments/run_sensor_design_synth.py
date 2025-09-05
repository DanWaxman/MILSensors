"""
Example: MIL-based sensor design on the synthetic dataset with visualization.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from milsensors.sensor_design import train_and_extract

DATA_PATH = "data/synthetic_temp_timeseries.npz"


def ensure_dataset():
    if not os.path.exists(DATA_PATH):
        from data.generate_synthetic_temperature import main as gen

        gen()


def reshape_to_grid(vec: np.ndarray, lats: np.ndarray, lons: np.ndarray):
    ulat = np.unique(lats)
    ulon = np.unique(lons)
    ny, nx = ulat.size, ulon.size
    return vec.reshape(ny, nx), ulat, ulon


def main():
    os.makedirs("experiments/results", exist_ok=True)
    ensure_dataset()

    artifacts = train_and_extract(
        seed=0,
        num_inducing=10,
        num_time_steps=24 * 4,
        dataset="phoenix",
        data_file=DATA_PATH,
        spatial_kern="matern32",
        temp_kern="subbandmix",
        noise_level=0.05,
        iters=250,
        parallel=False,
    )

    arr = np.load(DATA_PATH)["air_temp_timeseries"]
    N_t, N_sites = arr.shape[0], arr.shape[1]
    times = arr[:, 0, 0]
    lats = arr[0, :, 1]
    lons = arr[0, :, 2]

    Z_std = artifacts["z"]
    mus, stds = artifacts["mus"], artifacts["stds"]
    Z_lat = mus[1] + stds[1] * Z_std[:, 0]
    Z_lon = mus[2] + stds[2] * Z_std[:, 1]

    pm = artifacts["posterior_mean"].reshape(-1, N_sites)
    pv = artifacts["posterior_var"].reshape(-1, N_sites)
    pm_C = mus[3] + stds[3] * pm
    pv_C = (stds[3] ** 2) * pv

    # Plot inducing locations over site grid
    plt.figure(figsize=(6, 5))
    plt.scatter(lons, lats, s=8, c="lightgray", label="Sites")
    plt.scatter(Z_lon, Z_lat, s=40, c="crimson", marker="x", label="Inducing Z")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Sites and learned inducing points")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("experiments/results/synth_sites_inducing.png", dpi=150)

    # Spatial scatter at one time slice (irregular domain)
    t_plot = int(min(12, pm_C.shape[0] - 1))
    truth = arr[t_plot, :, 3]
    pred = pm_C[t_plot, :]
    err = pred - truth

    fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    sc0 = axs[0].scatter(lons, lats, c=truth, s=25, cmap="viridis")
    axs[0].set_title("Truth (C)")
    axs[0].set_xlabel("Lon")
    axs[0].set_ylabel("Lat")
    fig.colorbar(sc0, ax=axs[0])

    sc1 = axs[1].scatter(lons, lats, c=pred, s=25, cmap="viridis")
    axs[1].set_title("Prediction (C)")
    axs[1].set_xlabel("Lon")
    fig.colorbar(sc1, ax=axs[1])

    sc2 = axs[2].scatter(lons, lats, c=err, s=25, cmap="coolwarm")
    axs[2].set_title("Error (C)")
    axs[2].set_xlabel("Lon")
    fig.colorbar(sc2, ax=axs[2])
    plt.savefig("experiments/results/synth_spatial_scatter.png", dpi=150)

    # Time series at a representative site with uncertainty band
    site_idx = N_sites // 2
    truth_ts = arr[: pm_C.shape[0], site_idx, 3]
    pred_ts = pm_C[:, site_idx]
    std_ts = np.sqrt(np.maximum(pv_C[:, site_idx], 0.0))

    plt.figure(figsize=(8, 4))
    plt.plot(times[: pred_ts.shape[0]], truth_ts, label="Truth", lw=1.5)
    plt.plot(times[: pred_ts.shape[0]], pred_ts, label="Prediction", lw=1.5)
    plt.fill_between(
        times[: pred_ts.shape[0]],
        pred_ts - 2.0 * std_ts,
        pred_ts + 2.0 * std_ts,
        color="C1",
        alpha=0.2,
        label="±2σ",
    )
    plt.xlabel("Time (hours)")
    plt.ylabel("Temperature (C)")
    plt.title("Site time series with uncertainty")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("experiments/results/synth_time_series.png", dpi=150)

    print("Saved visualizations to experiments/results/")
    print("Inducing points shape:", Z_std.shape)
    print("RMSE (C):", artifacts["rmse"])
    print("NLPD:", float(artifacts["nlpd"]))
    plt.show()


if __name__ == "__main__":
    main()
