import os
import re
import numpy as np
import pandas as pd
import pygrib
from datetime import timedelta
from config import (
    BASE_DIR, LAT_BOUNDS, LON_BOUNDS, INIT_TIMES, FORECAST_HOURS,
    GRID_SIZE, RANGES, GRIB_NAMES
)

# -------------------- Utilities --------------------
def split_and_reshape_data(df, variable):
    """Split full dataframe into three lead-day dictionaries of 25-gridpoint frames (8 rows each)."""
    outputs = {}
    for day, steps in RANGES.items():
        keep_patterns = [f"_{x:03d}$" for x in steps]
        cols_mask = df.columns.to_series().apply(
            lambda c: any(re.search(pat, c) for pat in keep_patterns)
        )
        day_df = df.loc[:, cols_mask]

        chunks = [
            day_df.iloc[:, i:i + GRID_SIZE].values
            for i in range(0, day_df.shape[1], GRID_SIZE)
        ]

        if len(chunks) == 0:
            base_time = df.index[0] + pd.Timedelta(hours=min(steps))
            timestamps = pd.date_range(start=base_time, periods=8, freq='3h')
            result = pd.DataFrame(
                index=timestamps,
                columns=[f"{variable}_{i}" for i in range(GRID_SIZE)]
            )
            outputs[day] = result
            continue

        base_time = df.index[0] + pd.Timedelta(hours=min(steps))
        timestamps = pd.date_range(start=base_time, periods=len(chunks), freq='3h')

        result = pd.DataFrame(
            index=timestamps,
            columns=[f"{variable}_{i}" for i in range(GRID_SIZE)]
        )
        for i, chunk in enumerate(chunks):
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
    ts_utc = pd.Timestamp.utcnow().normalize() + pd.Timedelta(hours=6)

    lats = list(np.arange(LAT_BOUNDS[0], LAT_BOUNDS[1], -0.25))
    lons = list(np.arange(LON_BOUNDS[0], LON_BOUNDS[1], 0.25))
    cols = [
        f"{variable}_{lat}_{lon}_{fh:03d}"
        for fh in FORECAST_HOURS
        for lat in lats
        for lon in lons
    ]

    df = pd.DataFrame(index=[ts_utc], columns=cols)

    for fh in FORECAST_HOURS:
        filename = f"gfs.{ts_utc.strftime('%Y%m%d')}.t06z.pgrb2.0p25.f{fh:03d}"
        filepath = os.path.join(directory, filename)
        try:
            grbs = pygrib.open(filepath)
            grb = grbs.select(name=GRIB_NAMES[variable])[0]   # <-- from config
            data = grb.values
            grbs.close()

            latli = np.argmin(np.abs(grb.latitudes[:, 0] - LAT_BOUNDS[0]))
            latui = np.argmin(np.abs(grb.latitudes[:, 0] - LAT_BOUNDS[1]))
            lonli = np.argmin(np.abs(grb.longitudes[0, :] - LON_BOUNDS[0]))
            lonui = np.argmin(np.abs(grb.longitudes[0, :] - LON_BOUNDS[1]))

            grid = data[min(latli, latui):max(latli, latui)+1,
                        min(lonli, lonui):max(lonli, lonui)+1][::-1]

            flat = np.ravel(grid)

            if flat.shape[0] == GRID_SIZE:
                start_idx = FORECAST_HOURS.index(fh) * GRID_SIZE
                df.iloc[0, start_idx:start_idx + GRID_SIZE] = flat
            else:
                print(f"[WARN] Unexpected grid size in {filepath} for {variable}")
        except Exception as e:
            print(f"[ERROR] preprocess_wind: {filepath}: {e}")

    df = df.shift(freq=pd.Timedelta(hours=5, minutes=30))
    df.index.name = 'DateTime'
    df = df[df.index.time == pd.Timestamp("11:30").time()]

    return split_and_reshape_data(df, variable)


# -------------------- Precipitation preprocessing --------------------
def preprocess_precipitation(variable, directory):
    """
    Preprocess precipitation (PREC) to convert rate (kg/m2/s) to accumulated 3-hour values (mm).
    Returns dict with 3 lead-day DataFrames (8 rows x 25 columns).
    """
    ts_06utc = pd.Timestamp.utcnow().normalize() + pd.Timedelta(hours=6)

    cols = [
        f"{variable}_{lat}_{lon}_{t:03d}"
        for t in FORECAST_HOURS
        for lat in np.arange(LAT_BOUNDS[0], LAT_BOUNDS[1], -0.25)
        for lon in np.arange(LON_BOUNDS[0], LON_BOUNDS[1], 0.25)
    ]
    data_prec = pd.DataFrame(index=[ts_06utc], columns=cols)

    for time_step in data_prec.index:
        date_temp = pd.date_range(
            start=time_step + timedelta(hours=15),
            end=time_step + timedelta(hours=84),
            freq='3h'
        )
        data_temp = pd.DataFrame(index=date_temp, columns=np.arange(GRID_SIZE), dtype=float)

        for time_lag in FORECAST_HOURS:
            filename = f'gfs.{time_step.year}{time_step.month:02d}{time_step.day:02d}.t{time_step.hour:02d}z.pgrb2.0p25.f{time_lag:03d}'
            filepath = os.path.join(directory, filename)
            print(f"[INFO] Accessing file: {filepath}")
            try:
                grbs = pygrib.open(filepath)
                grb = grbs.select(name=GRIB_NAMES[variable])[0]   # <-- from config
                temp = grb.values
                lats, lons = grb.latlons()
                grbs.close()

                reversed_arr = lats[:, 0][::-1]
                lons_reshaped = lons[0, :]

                latli = np.argmin(np.abs(reversed_arr - LAT_BOUNDS[1]))
                latui = np.argmin(np.abs(reversed_arr - LAT_BOUNDS[0]))
                lonli = np.argmin(np.abs(lons_reshaped - LON_BOUNDS[0]))
                lonui = np.argmin(np.abs(lons_reshaped - LON_BOUNDS[1]))

                data = temp[latli:latui, lonli:lonui][::-1]

                time = time_step + timedelta(hours=int(time_lag))
                time_prev = time - timedelta(hours=3)

                if data.size == GRID_SIZE:
                    rflat = np.ravel(data)
                    if time_lag % 6 == 0:
                        if time_prev in data_temp.index and not data_temp.loc[time_prev].isnull().all():
                            data_temp.loc[time] = rflat * 21600 - data_temp.loc[time_prev]
                        else:
                            data_temp.loc[time] = rflat * 21600
                    else:
                        data_temp.loc[time] = rflat * 10800
                else:
                    print(f"[WARN] Precip crop produced {data.size} points (expected {GRID_SIZE}) in {filepath}")
            except Exception as e:
                print(f"[ERROR] {filename}: {e}")

        flattened = data_temp.values.flatten()
        required_len = len(cols)
        if flattened.size < required_len:
            flattened = np.concatenate([flattened, np.full(required_len - flattened.size, np.nan)])
        data_prec.loc[time_step] = flattened[:required_len]

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
    targets = {
        1: today + pd.Timedelta(days=1, hours=23, minutes=30),
        2: today + pd.Timedelta(days=2, hours=23, minutes=30),
        3: today + pd.Timedelta(days=3, hours=23, minutes=30),
    }

    for day, df in daily_data.items():
        agg_func = np.sum if variable == "PREC" else np.mean

        if df.shape[0] == 0:
            print(f"[WARN] No data for {variable} lead day {day}")
            grouped = pd.DataFrame(columns=df.columns)
        else:
            grouped = df.groupby(df.index.to_series().reset_index(drop=True).index // max(1, df.shape[0])).agg(agg_func)
            grouped.index = [targets[day]]
            grouped.index.name = "DateTime"

        filename = os.path.join(BASE_DIR, f"{variable}_Ahmedabad_LD_{day}_daily basis.xlsx")
        if os.path.exists(filename):
            try:
                old_df = pd.read_excel(filename, index_col='DateTime', parse_dates=True)
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
