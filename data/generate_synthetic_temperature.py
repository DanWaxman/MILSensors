"""
Synthetic spatiotemporal temperature dataset generator.

Saves an npz file with array "air_temp_timeseries" of shape (N_t, N_sites, 4):
  [:, :, 0] = time index (hours since start)
  [:, :, 1] = latitude
  [:, :, 2] = longitude
  [:, :, 3] = temperature (deg C)

The construction mimics diurnal cycles with spatially varying amplitude/phase,
slow baseline, and correlated spatiotemporal residuals.
"""

from typing import Tuple

import numpy as np


def gaussian_kernel_1d(sigma: float, truncate: float = 3.0) -> np.ndarray:
    radius = max(1, int(truncate * sigma))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-(x**2) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k


def gaussian_smooth2d(x: np.ndarray, sigma_y: float, sigma_x: float) -> np.ndarray:
    ky = gaussian_kernel_1d(sigma_y)
    kx = gaussian_kernel_1d(sigma_x)
    # separable convolution with reflect padding
    ypad = np.pad(x, ((ky.size // 2, ky.size // 2), (0, 0)), mode="reflect")
    y = np.apply_along_axis(lambda v: np.convolve(v, ky, mode="valid"), 0, ypad)
    xpad = np.pad(y, ((0, 0), (kx.size // 2, kx.size // 2)), mode="reflect")
    z = np.apply_along_axis(lambda v: np.convolve(v, kx, mode="valid"), 1, xpad)
    return z


def make_grid(ny: int, nx: int, lat_bounds: Tuple[float, float], lon_bounds: Tuple[float, float]):
    lats = np.linspace(lat_bounds[0], lat_bounds[1], ny)
    lons = np.linspace(lon_bounds[0], lon_bounds[1], nx)
    Lon, Lat = np.meshgrid(lons, lats)
    coords = np.stack([Lat.ravel(), Lon.ravel()], axis=-1)
    return Lat, Lon, coords


def rbfs(coords: np.ndarray, centers: np.ndarray, lengthscale: float) -> np.ndarray:
    # coords: [N, 2], centers: [K, 2]
    diffs = coords[:, None, :] - centers[None, :, :]
    sq = np.sum(diffs**2, axis=-1)
    Phi = np.exp(-0.5 * sq / (lengthscale**2))
    return Phi  # [N, K]


def generate_dataset(
    days: int = 14,
    ny: int = 28,
    nx: int = 28,
    lat_bounds: Tuple[float, float] = (33.3, 33.7),
    lon_bounds: Tuple[float, float] = (-112.2, -111.8),
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    N_t = 24 * days
    Lat, Lon, coords = make_grid(ny, nx, lat_bounds, lon_bounds)
    N_sites = ny * nx

    # Spatially smooth fields via RBF bases
    K = 6
    centers = np.column_stack(
        [
            rng.uniform(lat_bounds[0], lat_bounds[1], size=K),
            rng.uniform(lon_bounds[0], lon_bounds[1], size=K),
        ]
    )
    Phi = rbfs(coords, centers, lengthscale=0.07)  # ~8-10 km characteristic

    def smooth_field(base_scale: float, offset: float = 0.0):
        w = rng.normal(0.0, 1.0, size=K)
        f = Phi @ w
        f = (f - f.mean()) / (f.std() + 1e-8)
        return offset + base_scale * f

    # Baseline mean, diurnal amplitude, phase, and second harmonic amplitude
    mu = smooth_field(base_scale=3.0, offset=30.0)  # ~30 C baseline
    A1 = np.clip(smooth_field(base_scale=2.5, offset=4.0), 0.5, None)  # primary daily cycle
    phi = smooth_field(base_scale=0.6, offset=0.0)  # spatial phase (radians)
    A2 = np.clip(smooth_field(base_scale=0.8, offset=0.8), 0.0, None)  # second harmonic

    # Optional heatwave bump (temporal sigmoid window) with spatial footprint
    heat_spatial = np.clip(smooth_field(base_scale=1.5, offset=0.0), 0.0, None)
    tgrid = np.arange(N_t, dtype=float)
    center = 24.0 * (days / 2.0)
    width = 24.0 * 2.0
    heat_temporal = 1.0 / (1.0 + np.exp(-(tgrid - center + width) / 8.0)) - 1.0 / (
        1.0 + np.exp(-(tgrid - center - width) / 8.0)
    )
    heat_temporal = heat_temporal / (heat_temporal.max() + 1e-8)

    # Spatiotemporal residuals via AR(1) with spatially correlated innovations
    rho = 0.92
    sigma_eta = 0.4
    eta_sigma_y, eta_sigma_x = 1.0, 1.0  # grid cells
    w_prev = np.zeros((ny, nx))

    air_temp_timeseries = np.zeros((N_t, N_sites, 4), dtype=float)

    for t in range(N_t):
        # residual innovation
        noise = rng.normal(0.0, sigma_eta, size=(ny, nx))
        noise = gaussian_smooth2d(noise, eta_sigma_y, eta_sigma_x)
        w = rho * w_prev + noise
        w_prev = w

        # diurnal components
        angle = 2.0 * np.pi * (t % 24) / 24.0
        angle2 = 2.0 * angle

        # sitewise temperature
        temp = (
            mu
            + A1 * np.cos(angle + phi)
            + 0.5 * A2 * np.cos(angle2)
            + w.ravel()
            + 0.2 * rng.normal(0.0, 1.0, size=N_sites)
            + 1.5 * heat_temporal[t] * heat_spatial
        )

        # fill record
        air_temp_timeseries[t, :, 0] = float(t)
        air_temp_timeseries[t, :, 1] = coords[:, 0]
        air_temp_timeseries[t, :, 2] = coords[:, 1]
        air_temp_timeseries[t, :, 3] = temp

    # Create an irregular domain mask on the large grid
    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds
    lat_n = (coords[:, 0] - lat_min) / (lat_max - lat_min)
    lon_n = (coords[:, 1] - lon_min) / (lon_max - lon_min)

    region_L = (lon_n > 0.12) & (lat_n > 0.12) & ((lon_n < 0.35) | (lat_n < 0.35))
    region_circle = (lon_n - 0.70) ** 2 + (lat_n - 0.40) ** 2 < 0.12 ** 2
    hole = (lon_n - 0.50) ** 2 + (lat_n - 0.75) ** 2 < 0.08 ** 2
    mask = (region_L | region_circle) & (~hole)

    # Fallback in case mask is too small
    if mask.sum() < max(50, int(0.2 * N_sites)):
        rng = np.random.default_rng(seed)
        keep = rng.choice(N_sites, size=max(100, int(0.25 * N_sites)), replace=False)
        mask = np.zeros(N_sites, dtype=bool)
        mask[keep] = True

    # Apply mask to produce T x S' x 4 output
    air_temp_timeseries = air_temp_timeseries[:, mask, :]

    return air_temp_timeseries


def main():
    air_temp_timeseries = generate_dataset()
    np.savez_compressed(
        "data/synthetic_temp_timeseries.npz", air_temp_timeseries=air_temp_timeseries
    )
    print(
        "Saved synthetic dataset to data/synthetic_temp_timeseries.npz with shape",
        air_temp_timeseries.shape,
    )


if __name__ == "__main__":
    main()


