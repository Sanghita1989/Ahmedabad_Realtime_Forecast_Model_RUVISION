#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import re
import requests
import numpy as np
import pandas as pd
import pygrib
from datetime import datetime, timedelta, timezone

# ---------------------------- CONFIG ---------------------------- #

BASE_PATH = r"C:\Users\Angshudeep Majumdar\Downloads\Githubs_Code_RUVISION_Final"
LAT_BOUNDS = [22.25, 23.5]
LON_BOUNDS = [72.0, 73.25]
FORECAST_HOURS = list(range(15, 85, 3))
GRID_SIZE = 25
INIT_TIMES = [6]
VARIABLES = ['U1000', 'V1000', 'PREC']

# ---------------------- UTILITY FUNCTIONS ----------------------- #

def create_output_folder(variable):
    path = os.path.join(BASE_PATH, variable)
    os.makedirs(path, exist_ok=True)
    return path

def generate_url(current_date, init_hour, forecast_hour, variable):
    base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
    var_flags = {
        "U1000": "var_UGRD=on&lev_1000_mb=on",
        "V1000": "var_VGRD=on&lev_1000_mb=on",
        "PREC": "var_PRATE=on&lev_surface=on"
    }
    var_flag = var_flags[variable]
    return (
        f"{base_url}?dir=%2Fgfs.{current_date.strftime('%Y%m%d')}%2F{init_hour:02d}%2Fatmos&"
        f"file=gfs.t{init_hour:02d}z.pgrb2.0p25.f{forecast_hour:03d}&{var_flag}&"
        f"subregion=&toplat=47&leftlon=55&rightlon=105&bottomlat=0"
    )

def download_grib_files(variable):
    output_dir = create_output_folder(variable)
    current_date= datetime.now(timezone.utc)
    current_date = current_date - timedelta(days=1)

    for init_time in INIT_TIMES:
        for fh in FORECAST_HOURS:
            url = generate_url(current_date, init_time, fh, variable)
            filename = f"gfs.{current_date.strftime('%Y%m%d')}.t{init_time:02d}z.pgrb2.0p25.f{fh:03d}"
            filepath = os.path.join(output_dir, filename)

            if not os.path.exists(filepath):
                response = requests.get(url)
                if response.status_code == 200:
                    with open(filepath, "wb") as f:
                        f.write(response.content)
                    print(f"Downloaded: {filename}")
                else:
                    print(f"Failed to download ({response.status_code}): {url}")

    return output_dir

def extract_data_grid(filepath, variable):
    try:
        fh = int(re.search(r'f(\d{3})$', filepath).group(1))
        grbs = pygrib.open(filepath)
        var_name_map = {
            "U1000": "U component of wind",
            "V1000": "V component of wind",
        }
        grb = grbs.select(name=var_name_map[variable])[0]
        data = grb.values
        grbs.close()

        grid = data[2:7, 2:7][::-1]
        flat = np.ravel(grid)
        return flat
    except Exception as e:
        print(f"[ERROR] Failed to extract data from {filepath}: {e}")
        return np.full(GRID_SIZE, np.nan)

def preprocess_wind(variable, directory):
    #ts_utc = pd.Timestamp.utcnow().normalize() + pd.Timedelta(hours=6)
    ts_utc = pd.Timestamp.utcnow().normalize()- pd.Timedelta(days=1) + pd.Timedelta(hours=6)

    df = pd.DataFrame(index=[ts_utc], columns=[f"{variable}_{lat}_{lon}_{fh:03d}"
                                               for fh in FORECAST_HOURS
                                               for lat in np.arange(23.5, 22.25, -0.25)
                                               for lon in np.arange(72.0, 73.25, 0.25)])

    for fh in FORECAST_HOURS:
        filename = f"gfs.{ts_utc.strftime('%Y%m%d')}.t06z.pgrb2.0p25.f{fh:03d}"
        filepath = os.path.join(directory, filename)
        flat = extract_data_grid(filepath, variable)
        if flat.shape[0] == GRID_SIZE:
            for i, val in enumerate(flat):
                df.iloc[0, i + (FORECAST_HOURS.index(fh) * GRID_SIZE)] = val

    df = df.shift(freq=pd.Timedelta(hours=5, minutes=30))
    df.index.name = 'DateTime'
    df = df[df.index.time == pd.Timestamp("11:30").time()]
    return split_and_reshape_data(df, variable)


def preprocess_precipitation(variable, directory):
    latbounds = LAT_BOUNDS
    lonbounds = LON_BOUNDS
    time_from_ref = FORECAST_HOURS

    #ts_06utc = pd.Timestamp.utcnow().normalize() + pd.Timedelta(hours=6)
    ts_06utc = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=6)

    data_prec = pd.DataFrame(index=[ts_06utc], columns=[
        f"{variable}_{j}_{k}_{t:03d}"
        for t in time_from_ref
        for j in np.arange(23.5, 22.25, -0.25)
        for k in np.arange(72.0, 73.25, 0.25)
    ])

    counter = 0  # add counter if needed

    for time_step in data_prec.index:
        year = time_step.year
        month = time_step.month
        day = time_step.day
        ref_time = time_step.hour

        date_temp = pd.date_range(start=time_step + timedelta(hours=15),
                              end=time_step + timedelta(hours=84), freq='3h')
        data_temp = pd.DataFrame(index=date_temp, columns=np.arange(25))

        for time_lag in time_from_ref:
            filename = f'gfs.{year}{month:02d}{day:02d}.t{ref_time:02d}z.pgrb2.0p25.f{time_lag:03d}'
            filepath = os.path.join(directory, filename)
            print(f"[INFO] Accessing file: {filepath}")  # 👈 This line will print the full path

            try:
                grbs = pygrib.open(filepath)
                grb = grbs.select(name='Precipitation rate')[0]
                temp = grb.values
                lats, lons = grb.latlons()
                lats_reshaped = lats[:,0]  # Reshape latitudes to (189,)
                reversed_arr = lats_reshaped[::-1]
                lons_reshaped = lons[0,:]  # Reshape longitudes to (,201)
                lats=reversed_arr
                lons=lons_reshaped

                # latitude lower and upper index
                latli = np.argmin( np.abs( reversed_arr - latbounds[1] ) )
                latui = np.argmin( np.abs( reversed_arr - latbounds[0] ) ) 

                # longitude lower and upper index
                lonli = np.argmin( np.abs( lons_reshaped- lonbounds[0] ) )
                lonui = np.argmin( np.abs( lons_reshaped - lonbounds[1] ) )  

                data = temp[latli:latui, lonli:lonui][::-1]
                time = time_step + timedelta(hours=int(time_lag))
                time_prev = time - timedelta(hours=3)

                if data.size == 25:
                    if time_lag % 6 == 0:
                        if time_prev in data_temp.index:
                            data_temp.loc[time][0:25] = (np.ravel(data)*21600) - np.ravel(data_temp.loc[time_prev][0:25])
                    elif time_lag % 3 == 0:
                        data_temp.loc[time][0:25] = (np.ravel(data)*10800)

            except Exception as e:
                print(f"[ERROR] {filename}: {e}")

        data_prec.loc[time_step][0:600] = np.ravel(data_temp)

        counter += 1
        if counter % 100 == 0:
            print(f'Loop {counter} Done!')

    print("\n[DEBUG] Raw data_prec before timezone shift and filtering:")
    print(data_prec)

    data_prec = data_prec.shift(freq=pd.Timedelta(hours=5, minutes=30))
    data_prec.index.name = 'DateTime'
    extracted = data_prec[data_prec.index.time == pd.Timestamp("11:30").time()]

    return split_and_reshape_data(extracted, variable)


def split_and_reshape_data(df, variable):
    ranges = {
        1: list(range(15, 37, 3)),
        2: list(range(39, 61, 3)),
        3: list(range(63, 85, 3))
    }
    outputs = {}
    for day, steps in ranges.items():
        cols = ~df.columns.str.contains('|'.join([f"{x:03d}" for x in FORECAST_HOURS if x not in steps]))
        day_df = df.loc[:, cols]
        chunks = [day_df.iloc[:, i:i + 25].values for i in range(0, day_df.shape[1], 25)]
        base_time = df.index[0] + pd.Timedelta(hours=min(steps))
        timestamps = pd.date_range(start=base_time, periods=8, freq='3h')
        result = pd.DataFrame(index=timestamps, columns=[f"{variable}_{i}" for i in range(25)])
        for i, chunk in enumerate(chunks):
            result.iloc[i] = chunk[0]
        outputs[day] = result
    return outputs

def aggregate_and_save(daily_data, variable):
    #today = pd.Timestamp.today().normalize()
    today = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)

    targets = {
        1: today + pd.Timedelta(days=1, hours=23, minutes=30),
        2: today + pd.Timedelta(days=2, hours=23, minutes=30),
        3: today + pd.Timedelta(days=3, hours=23, minutes=30),
    }

    for day, df in daily_data.items():
        agg_func = np.sum if variable == "PREC" else np.mean
        grouped = df.groupby(df.index.to_series().reset_index(drop=True).index // 8).agg(agg_func)
        grouped.index = [targets[day]]
        grouped.index.name = "DateTime"

        filename = os.path.join(BASE_PATH, f"{variable}_Ahmedabad_Lead Day {day}_daily basis.xlsx")

        if os.path.exists(filename):
            try:
                old_df = pd.read_excel(filename, index_col='DateTime', parse_dates=True)
                grouped.columns = old_df.columns
            except Exception as e:
                print(f"[WARNING] Could not read or align existing file: {e}")
                old_df = pd.DataFrame(columns=grouped.columns)
        else:
            old_df = pd.DataFrame(columns=grouped.columns)

        combined = pd.concat([old_df, grouped])
        combined = combined[~combined.index.duplicated(keep='last')]

        combined.to_excel(filename)
        print(f"[INFO] Saved to: {filename}")
        print(combined)

# ---------------------- MAIN EXECUTION ----------------------- #

def preprocess_variable(variable, directory):
    if variable == "PREC":
        return preprocess_precipitation(variable, directory)
    else:
        return preprocess_wind(variable, directory)

def main():
    for variable in VARIABLES:
        print(f"Processing: {variable}")
        output_dir = download_grib_files(variable)
        daily_data = preprocess_variable(variable, output_dir)
        aggregate_and_save(daily_data, variable)

if __name__ == "__main__":
    main()


# In[ ]:




