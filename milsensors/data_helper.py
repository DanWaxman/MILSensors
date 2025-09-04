"""
Helper functions for loading and processing temperature data.

This module provides functions to load WRF (Weather Research and Forecasting)
temperature data, create training and testing datasets, and prepare the data
for use with spatiotemporal Gaussian Process models.

The module includes:
- create_dataset: Function to load data and create training/testing splits
- make_grid: Function to convert data into spatiotemporal grid format
"""

import numpy as np
import pandas as pd
import bayesnewton


def create_dataset(
    random_seed,
    N_obs_pts,
    N_t,
    dataset="phoenix",
    obs_noise=0.0,
    moving_points=False,
    data_file="data/WRF_data_2013.csv",
    N_fixed=0,
    N_t_start=0,
):
    """
    Load temperature data and create training/testing datasets.

    Args:
        random_seed: Random seed for reproducibility
        N_obs_pts: Number of observation points to sample (-1 for all available)
        N_t: Number of time points to use
        dataset: Region to use ('phoenix', 'flagstaff', 'tuscon', 'arizona')
        obs_noise: Amount of observation noise to add
        moving_points: Whether to use different random points at each time step
        data_file: Path to the data file
        N_fixed: Number of fixed observation locations
        N_t_start: Starting time index

    Returns:
        X: Training inputs (time, lat, lon)
        Y: Training targets (temperature)
        X_t: Test inputs
        Y_t: Test targets
        air_temp_timeseries: Full temperature time series
        N_t: Number of time points
        mus: Mean values for normalization
        stds: Standard deviations for normalization
        N_sites: Number of spatial locations
        sample_points: Indices of sampled points
    """
    # Load data from CSV or NPZ file
    if data_file.endswith("csv"):
        WRF_data_raw = pd.read_csv(
            data_file,
            delimiter=r"\s+",
            header=None,
            names=[
                "Urban point identifier",
                "latitude",
                "longitude",
                "Land Use-Land Cover",
                "albedo",
                "emissivity",
                "Air Temperature at 2m",
                "Surface (Skin) Temperature",
                "Water Vapor Mixing Ratio",
                "Wind Speed",
                "Shortwave downwelling",
                "Longwave downwelling",
                "Pressure",
            ],
        )
        # No time axis by default, make one which gives hours since midnight June 1st
        # 24*92=2208 total time instances
        # the data is formatted so that each site is repeated N_t times
        WRF_data_raw["Time"] = np.concatenate(
            [np.arange(2208) for _ in range(WRF_data_raw.shape[0] // 2208)]
        )

        # Allow a dataset of all urban points in Arizona or just Phoenix
        if dataset == "phoenix":
            # Bounding rectangle for Phoenix
            WRF_df = WRF_data_raw[
                (WRF_data_raw["longitude"] > -112.324)
                & (WRF_data_raw["longitude"] < -111.925)
                & (WRF_data_raw["latitude"] > 33.29)
                & (WRF_data_raw["latitude"] < 33.9)
            ]

        elif dataset == "flagstaff":
            # Bounding rectangle for Flagstaff
            WRF_df = WRF_data_raw[
                (WRF_data_raw["longitude"] > -112.507)
                & (WRF_data_raw["longitude"] < -111.709)
                & (WRF_data_raw["latitude"] > 35.122)
                & (WRF_data_raw["latitude"] < 35.240)
            ]

        elif dataset == "tuscon":
            # Bounding rectangle for Tuscon
            WRF_df = WRF_data_raw[
                (WRF_data_raw["longitude"] > -87.931)
                & (WRF_data_raw["longitude"] < -87.934)
                & (WRF_data_raw["latitude"] > 41.854)
                & (WRF_data_raw["latitude"] < 41.857)
            ]

        elif dataset == "arizona":
            N_sites = 675
            WRF_df = WRF_data_raw

        elif dataset == "restricted_phoenix":
            rectangle_1_query = (
                (WRF_data_raw["longitude"] > -112.2)
                & (WRF_data_raw["longitude"] < -111.84)
                & (WRF_data_raw["latitude"] > 33.4)
                & (WRF_data_raw["latitude"] < 33.56)
            )

            rectangle_2_query = (
                (WRF_data_raw["longitude"] > -112.0)
                & (WRF_data_raw["longitude"] < -111.6)
                & (WRF_data_raw["latitude"] > 33.26)
                & (WRF_data_raw["latitude"] < 33.44)
            )

            WRF_df = WRF_data_raw[rectangle_1_query | rectangle_2_query]
        else:
            raise NotImplementedError

        N_sites = WRF_df.shape[0] // 2208

        # Set up time series
        # there are 314 sites within the Phoenix bounding box
        air_temp_timeseries = np.zeros((N_t, N_sites, 4))

        for t in range(N_t):
            # get time series of (time, lat, long, air temp at 2m)
            air_temp_timeseries[t, ...] = WRF_df.iloc[t::2208, [13, 1, 2, 6]]
    elif data_file.endswith("npz"):
        air_temp_timeseries = np.load(data_file)["air_temp_timeseries"]
        N_sites = air_temp_timeseries.shape[1]
    else:
        raise ValueError("Unrecognized file format (must be .csv or .npz)")

    # Standardize data for GPs
    mus = np.mean(air_temp_timeseries, axis=(0, 1))
    stds = np.std(air_temp_timeseries, axis=(0, 1))
    # Don't change time
    mus[0] = 0
    stds[0] = 1
    air_temp_timeseries = (air_temp_timeseries - mus) / stds

    if N_obs_pts == -1:
        N_obs_pts = N_sites - N_fixed
    assert N_obs_pts + N_fixed <= N_sites

    # Sample observation points
    if N_fixed > 0:
        # Use a fixed random seed to choose fixed locations
        np.random.seed(0)
        fixed_sites = np.random.choice(N_sites, N_fixed, replace=False)

        np.random.seed(random_seed)

        # Choose N_obs_pts random locations
        sample_points_idx = np.random.choice(
            N_sites - N_fixed, N_obs_pts, replace=False
        )
        sample_points = np.concatenate(
            [fixed_sites, np.delete(np.arange(N_sites), fixed_sites)[sample_points_idx]]
        )
    else:
        np.random.seed(random_seed)

        sample_points = np.random.choice(N_sites, N_obs_pts, replace=False)

    # Extract training data
    X = air_temp_timeseries[N_t_start : N_t + N_t_start, sample_points, 0:3]
    Y = air_temp_timeseries[N_t_start : N_t + N_t_start, sample_points, 3]

    # Add noise to temps if specified
    if obs_noise > 0:
        Y = Y + np.random.randn(*Y.shape) * obs_noise / stds[3]

    # If choosing random, new points at each time step
    if moving_points:
        for t_i in range(N_t):
            sample_points = np.random.choice(N_sites, N_obs_pts, replace=False)
            X[t_i, ...] = air_temp_timeseries[t_i, sample_points, 0:3]
            Y[t_i, ...] = air_temp_timeseries[t_i, sample_points, 3]

    # X includes (time, lat, long)
    # Y includes air temp at 2m

    # Reshape training data
    X = X.reshape(N_t * N_obs_pts, 3)
    Y = Y.reshape(N_t * N_obs_pts, 1)

    # Create test data (all sites)
    X_t = air_temp_timeseries[N_t_start : N_t + N_t_start, :, 0:3].reshape(
        N_t * N_sites, 3
    )
    Y_t = air_temp_timeseries[N_t_start : N_t + N_t_start, :, 3].reshape(
        N_t * N_sites, 1
    )

    return X, Y, X_t, Y_t, air_temp_timeseries, N_t, mus, stds, N_sites, sample_points


def make_grid(X, Y, X_t, Y_t, air_temp_timeseries, N_t, mus, stds, N_sites):
    """
    Convert data into spatiotemporal grid format for GP models.

    Args:
        X: Training inputs
        Y: Training targets
        X_t: Test inputs
        Y_t: Test targets
        air_temp_timeseries: Full temperature time series
        N_t: Number of time points
        mus: Mean values for normalization
        stds: Standard deviations for normalization
        N_sites: Number of spatial locations

    Returns:
        X: Original training inputs
        Y: Original training targets
        R: Spatial training inputs
        t: Temporal training inputs
        X_t: Original test inputs
        Y_t: Original test targets
        R_t: Spatial test inputs
        t_t: Temporal test inputs
    """
    t, R, Y = bayesnewton.utils.create_spatiotemporal_grid(X, Y)
    t_t, R_t, Y_t = bayesnewton.utils.create_spatiotemporal_grid(X_t, Y_t)

    return X, Y, R, t, X_t, Y_t, R_t, t_t
