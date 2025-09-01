#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import re
import numpy as np
import pandas as pd
import pygrib
from datetime import timedelta
from config import (
    BASE_DIR, LAT_BOUNDS, LON_BOUNDS, INIT_TIMES, FORECAST_HOURS,
    GRID_SIZE
)

# -------------------- Utilities --------------------
def split_and_reshape_data(df, variable):
    """Split full dataframe into three lead-day dictionaries of 25-gridpoint frames (8 rows each)."""
    ranges = {
        1: list(range(15, 37, 3)),
        2: list(range(39, 61, 3)),
        3: list(range(63, 85, 3))
    }

    outputs = {}
    for day, steps in ranges.items():
        # Keep only columns corresponding to these forecast steps
        keep_patterns = [f"_{x:03d}$" for x in steps]
        cols_mask = df.columns.to_series().apply(lambda c: any(re.search(pat, c) for pat in keep_patterns))
        day_df = df.loc[:, cols_mask]

        # break into contiguous 25-column chunks (each chunk -> one grid snapshot)
        chunks = [day_df.iloc[:, i:i + GRID_SIZE].values for i in range(0, day_df.shape[1], GRID_SIZE)]

        if len(chunks) == 0:
            # create empty DataFrame with 8 timestamps (3-hourly) and 25 columns
            base_time = df.index[0] + pd.Timedelta(hours=min(steps))
            timestamps = pd.date_range(start=base_time, periods=8, freq='3h')
            result = pd.DataFrame(index=timestamps, columns=[f"{variable}_{i}" for i in range(GRID_SIZE)])
            outputs[day] = result
            continue

        base_time = df.index[0] + pd.Timedelta(hours=min(steps))
        timestamps = pd.date_range(start=base_time, periods=len(chunks), freq='3h')

        result = pd.DataFrame(index=timestamps, columns=[f"{variable}_{i}" for i in range(GRID_SIZE)])
        for i, chunk in enumerate(chunks):
            # chunk shape is (1,25) typically because df had single row per initialization
            if chunk.shape[0] >= 1:
                result.iloc[i] = chunk[0]
            else:
                result.iloc[i] = np.nan
        outputs[day] = result
    return outputs


# -------------------- Wind preprocessing --------------------
def preprocess_wind(variable, directory):
    """
    Preprocess instantaneous variables (U1000, V1000).
    Returns dict with three lead day DataFrames (25 columns each, 8 rows).
    """
    # timestamp corresponding to initialization
    ts_utc = pd.Timestamp.utcnow().normalize() + pd.Timedelta(hours=6)
    #ts_utc = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=6)
    
    # generate column names: for each forecast hour, each lat, lon
    lats = list(np.arange(23.5, 22.25, -0.25))
    lons = list(np.arange(72.0, 73.25, 0.25))
    cols = [f"{variable}_{lat}_{lon}_{fh:03d}" for fh in FORECAST_HOURS for lat in lats for lon in lons]

    df = pd.DataFrame(index=[ts_utc], columns=cols)

    for fh in FORECAST_HOURS:
        filename = f"gfs.{ts_utc.strftime('%Y%m%d')}.t06z.pgrb2.0p25.f{fh:03d}"
        filepath = os.path.join(directory, filename)
        try:
            grbs = pygrib.open(filepath)
            var_name_map = {
                "U1000": "U component of wind",
                "V1000": "V component of wind"
            }
            grb = grbs.select(name=var_name_map[variable])[0]
            data = grb.values
            grbs.close()

            grid = data[2:7, 2:7][::-1]  # 5x5 block
            flat = np.ravel(grid)

            if flat.shape[0] == GRID_SIZE:
                start_idx = FORECAST_HOURS.index(fh) * GRID_SIZE
                df.iloc[0, start_idx:start_idx + GRID_SIZE] = flat
            else:
                print(f"[WARN] Unexpected grid size in {filepath} for {variable}")
        except Exception as e:
            print(f"[ERROR] preprocess_wind: {filepath}: {e}")
            # leave NaNs

    # Convert from UTC to IST and pick 11:30 row
    df = df.shift(freq=pd.Timedelta(hours=5, minutes=30))
    df.index.name = 'DateTime'
    df = df[df.index.time == pd.Timestamp("11:30").time()]

    return split_and_reshape_data(df, variable)


# -------------------- Precipitation preprocessing --------------------
def preprocess_precipitation(variable, directory):
    """
    Preprocess precipitation (PRATE) to convert rate (kg/m2/s) to accumulated 3-hour values (mm).
    Returns dict with 3 lead-day DataFrames (8 rows x 25 columns).
    """
    latbounds = LAT_BOUNDS
    lonbounds = LON_BOUNDS
    time_from_ref = FORECAST_HOURS
    
    ts_06utc = pd.Timestamp.utcnow().normalize() + pd.Timedelta(hours=6)
    #ts_06utc = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=6)

    cols = [f"{variable}_{j}_{k}_{t:03d}" for t in time_from_ref for j in np.arange(23.5, 22.25, -0.25) for k in np.arange(72.0, 73.25, 0.25)]
    data_prec = pd.DataFrame(index=[ts_06utc], columns=cols)

    # temporary storage per initialization to build 3-hourly accumulations
    for time_step in data_prec.index:
        date_temp = pd.date_range(start=time_step + timedelta(hours=15), end=time_step + timedelta(hours=84), freq='3h')
        data_temp = pd.DataFrame(index=date_temp, columns=np.arange(GRID_SIZE), dtype=float)

        for time_lag in time_from_ref:
            filename = f'gfs.{time_step.year}{time_step.month:02d}{time_step.day:02d}.t{time_step.hour:02d}z.pgrb2.0p25.f{time_lag:03d}'
            filepath = os.path.join(directory, filename)
            print(f"[INFO] Accessing file: {filepath}")
            try:
                grbs = pygrib.open(filepath)
                grb = grbs.select(name='Precipitation rate')[0]
                temp = grb.values
                lats, lons = grb.latlons()
                grbs.close()

                reversed_arr = lats[:, 0][::-1]
                lons_reshaped = lons[0, :]

                latli = np.argmin(np.abs(reversed_arr - latbounds[1]))
                latui = np.argmin(np.abs(reversed_arr - latbounds[0]))
                lonli = np.argmin(np.abs(lons_reshaped - lonbounds[0]))
                lonui = np.argmin(np.abs(lons_reshaped - lonbounds[1]))

                data = temp[latli:latui, lonli:lonui][::-1]

                time = time_step + timedelta(hours=int(time_lag))
                time_prev = time - timedelta(hours=3)

                if data.size == GRID_SIZE:
                    rflat = np.ravel(data)
                    # kg/m2/s to mm over 3h -> multiply by 10800
                    if time_lag % 6 == 0:
                        # 6-hour accumulation, subtract previous 3-hr component if available
                        if time_prev in data_temp.index and not data_temp.loc[time_prev].isnull().all():
                            data_temp.loc[time] = rflat * 21600 - data_temp.loc[time_prev]
                        else:
                            data_temp.loc[time] = rflat * 21600
                    else:
                        # 3-hour accumulation
                        data_temp.loc[time] = rflat * 10800
                else:
                    print(f"[WARN] Precip crop produced {data.size} points (expected {GRID_SIZE}) in {filepath}")

            except Exception as e:
                print(f"[ERROR] {filename}: {e}")

        # flatten the data_temp into the main data_prec row (in chronological order)
        flattened = data_temp.values.flatten()
        # if lengths mismatch, pad/truncate
        required_len = len(cols)
        if flattened.size < required_len:
            flattened = np.concatenate([flattened, np.full(required_len - flattened.size, np.nan)])
        data_prec.loc[time_step] = flattened[:required_len]

    # timezone shift to IST
    data_prec = data_prec.shift(freq=pd.Timedelta(hours=5, minutes=30))
    data_prec.index.name = 'DateTime'
    extracted = data_prec[data_prec.index.time == pd.Timestamp("11:30").time()]

    return split_and_reshape_data(extracted, variable)


# -------------------- Aggregate & Save --------------------
def aggregate_and_save(daily_data, variable):
    """
    For each lead day (1/2/3) aggregate to daily stats and append to existing Excel file.
    If PREC -> sum over the 8 x 3-hour slots; else -> mean (for instantaneous variables).
    """
    today = pd.Timestamp.today().normalize()
    #today = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    targets = {
        1: today + pd.Timedelta(days=1, hours=23, minutes=30),
        2: today + pd.Timedelta(days=2, hours=23, minutes=30),
        3: today + pd.Timedelta(days=3, hours=23, minutes=30),
    }

    for day, df in daily_data.items():
        agg_func = np.sum if variable == "PREC" else np.mean

        # group rows by block of 8 (if df has 8 rows for 8 time steps)
        if df.shape[0] == 0:
            print(f"[WARN] No data for {variable} lead day {day}")
            grouped = pd.DataFrame(columns=df.columns)
        else:
            grouped = df.groupby(df.index.to_series().reset_index(drop=True).index // max(1, df.shape[0])).agg(agg_func)
            # set index to target date
            grouped.index = [targets[day]]
            grouped.index.name = "DateTime"

        filename = os.path.join(BASE_DIR, f"{variable}_Ahmedabad_LD_{day}_daily basis.xlsx")
        if os.path.exists(filename):
            try:
                old_df = pd.read_excel(filename, index_col='DateTime', parse_dates=True)
                # align columns if possible
                grouped = grouped.reindex(columns=old_df.columns, fill_value=0)
            except Exception as e:
                print(f"[WARNING] Could not read or align existing file: {e}")
                old_df = pd.DataFrame(columns=grouped.columns)
        else:
            old_df = pd.DataFrame(columns=grouped.columns)

        combined = pd.concat([old_df, grouped])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined.to_excel(filename)
        print(f"[INFO] Saved to: {filename}")
        print(combined.tail(2))


# -------------------- Dispatcher --------------------
def preprocess_variable(variable, directory):
    if variable == "PREC":
        return preprocess_precipitation(variable, directory)
    else:
        return preprocess_wind(variable, directory)

