"""
Data processing script for WRF (Weather Research and Forecasting) temperature data.

This script processes raw WRF temperature data files for Arizona from summer 2020 
(June, July, August) and creates several datasets for different geographic regions:
- Full dataset (all points)
- Urban corridor (points with specific land use/land cover codes)
- Phoenix metropolitan area
- Mini Phoenix (smaller subset of Phoenix)
- Flagstaff
- Tucson

Each dataset is saved as a compressed NPZ file containing temperature time series data
with dimensions [time_steps, points, features], where features include:
- Time index
- Latitude
- Longitude
- Temperature (T2)

The script handles data from multiple months, sorts it, and organizes it into a 
consistent format for further analysis.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

print("Loading Raw Data")
# Load raw data files for June, July, and August 2020
df_june = pd.read_csv(
    "WRF_INNER_DOMAIN_DOE_LULC_T2_JUNE_2020_AZ_1km.data",
    sep="\s+",
    header=None,
)
df_july = pd.read_csv(
    "WRF_INNER_DOMAIN_DOE_LULC_T2_JULY_2020_AZ_1km.data",
    sep="\s+",
    header=None,
)
df_august = pd.read_csv(
    "WRF_INNER_DOMAIN_DOE_LULC_T2_AUGUST_2020_AZ_1km.data",
    sep="\s+",
    header=None,
)
# Combine all months and sort by first column (likely point ID)
df = pd.concat([df_june, df_july, df_august])
sorted_df = df.sort_values(by=0, kind="stable")

# Also load 2013 data (though not used in this script)
df_2013 = pd.read_csv("WRF_data_2013.csv")

print("Creating Full Dataset")

# Save to an array where each row is one point
# with point ID, LULC, Lat, Long, T2[0], ..., T2[2207]
N_pts = 146556
data = np.zeros((N_pts, 2212))

### Full Dataset ###
# Reshape data so each point has all its time series data in one row
for n in tqdm(range(N_pts)):
    data[n, :4] = sorted_df.iloc[92 * n, :4]  # Point metadata (ID, LULC, lat, lon)
    data[n, 4:] = np.concatenate(sorted_df.iloc[92 * n : 92 * (n + 1), 4:].to_numpy())  # Temperature data

# Reshape into time series format [time, point, features]
air_temp_timeseries = np.zeros((2208, N_pts, 4))
for t in tqdm(range(2208)):
    for n in range(N_pts):
        air_temp_timeseries[t, n, 0] = t  # Time index
        air_temp_timeseries[t, n, 1:] = data[n, np.array([2, 3, 4 + t])]  # Lat, Lon, Temp

# Save full dataset
np.savez_compressed("WRF_data_2020_v2.data", air_temp_timeseries=air_temp_timeseries)
print(f"Done creating full dataset with {N_pts} points.")

### Urban Corridor Dataset ###
print("Creating Urban Corridor Dataset")
# Select points with urban land use/land cover codes (13, 31, 32, 33)
urban_points = np.where(
    (data[:, 1] == 13) | (data[:, 1] == 31) | (data[:, 1] == 32) | (data[:, 1] == 33)
)[0]

urban_corridor_data = data[urban_points]
N_pts = urban_corridor_data.shape[0]

# Create time series format for urban corridor
air_temp_timeseries = np.zeros((2208, N_pts, 4))
for t in tqdm(range(2208)):
    for n in range(N_pts):
        air_temp_timeseries[t, n, 0] = t
        air_temp_timeseries[t, n, 1:] = urban_corridor_data[n, np.array([2, 3, 4 + t])]

# Save urban corridor dataset
np.savez_compressed(
    "WRF_data_2020_v2_urban_corridor",
    air_temp_timeseries=air_temp_timeseries,
)
print(f"Done creating Urban Corridor dataset with {N_pts} points.")

### Phoenix Dataset ###
print("Creating Phoenix Dataset")
# Select points within Phoenix geographic boundaries
phoenix_data = []
for n in tqdm(range(146556)):
    lat = data[n, 2]
    long = data[n, 3]
    if long > -112.324 and long < -111.925 and lat > 33.29 and lat < 33.9:
        phoenix_data.append(data[n])

phoenix_data = np.array(phoenix_data)
N_pts = phoenix_data.shape[0]

# Create time series format for Phoenix
air_temp_timeseries = np.zeros((2208, N_pts, 4))
for t in tqdm(range(2208)):
    for n in range(N_pts):
        air_temp_timeseries[t, n, 0] = t
        air_temp_timeseries[t, n, 1:] = phoenix_data[n, np.array([2, 3, 4 + t])]

# Save Phoenix dataset
np.savez_compressed("WRF_data_2020_v2_phoenix", air_temp_timeseries=air_temp_timeseries)
print(f"Done creating Phoenix dataset with {N_pts} points.")

### Mini Phoenix Dataset ###
print("Creating Mini Phoenix Dataset")
# Select points within a smaller Phoenix area
phoenix_data = []
for n in tqdm(range(146556)):
    lat = data[n, 2]
    long = data[n, 3]
    if long > -112.2 and long < -112.0 and lat > 33.4 and lat < 33.55:
        phoenix_data.append(data[n])

phoenix_data = np.array(phoenix_data)
N_pts = phoenix_data.shape[0]

# Create time series format for Mini Phoenix
air_temp_timeseries = np.zeros((2208, N_pts, 4))
for t in tqdm(range(2208)):
    for n in range(N_pts):
        air_temp_timeseries[t, n, 0] = t
        air_temp_timeseries[t, n, 1:] = phoenix_data[n, np.array([2, 3, 4 + t])]

# Save Mini Phoenix dataset
np.savez_compressed(
    "WRF_data_2020_v2_miniphoenix", air_temp_timeseries=air_temp_timeseries
)
print(f"Done creating Mini Phoenix dataset with {N_pts} points.")

### Flagstaff Dataset ###
print("Creating Flagstaff Dataset")
# Select points within Flagstaff geographic boundaries
flagstaff_data = []
for n in tqdm(range(146556)):
    lat = data[n, 2]
    long = data[n, 3]
    if long > -111.709 and long < -111.507 and lat > 35.122 and lat < 35.240:
        flagstaff_data.append(data[n])

flagstaff_data = np.array(flagstaff_data)
N_pts = flagstaff_data.shape[0]

# Create time series format for Flagstaff
air_temp_timeseries = np.zeros((2208, N_pts, 4))
for t in tqdm(range(2208)):
    for n in range(N_pts):
        air_temp_timeseries[t, n, 0] = t
        air_temp_timeseries[t, n, 1:] = flagstaff_data[n, np.array([2, 3, 4 + t])]

# Save Flagstaff dataset
np.savez_compressed(
    "WRF_data_2020_v2_flagstaff",
    air_temp_timeseries=air_temp_timeseries,
)
print(f"Done creating Flagstaff dataset with {N_pts} points.")

### Tucson Dataset ###
print("Creating Tucson Dataset")
# Select points within Tucson geographic boundaries
tucson_data = []
for n in tqdm(range(146556)):
    lat = data[n, 2]
    long = data[n, 3]
    if long > -111.058 and long < -110.708 and lat > 31.99 and lat < 32.321:
        tucson_data.append(data[n])

tucson_data = np.array(tucson_data)
N_pts = tucson_data.shape[0]

# Create time series format for Tucson
air_temp_timeseries = np.zeros((2208, N_pts, 4))
for t in tqdm(range(2208)):
    for n in range(N_pts):
        air_temp_timeseries[t, n, 0] = t
        air_temp_timeseries[t, n, 1:] = tucson_data[n, np.array([2, 3, 4 + t])]

# Save Tucson dataset
np.savez_compressed(
    "WRF_data_2020_v2_tucson",  # Fixed typo in filename from "tuscon" to "tucson"
    air_temp_timeseries=air_temp_timeseries,
)
print(f"Done creating Tucson dataset with {N_pts} points.")
