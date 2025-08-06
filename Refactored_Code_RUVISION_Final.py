#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import re
import glob
import requests
import numpy as np
import pandas as pd
import pygrib
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta, timezone

# ---------------------------- CONFIG ---------------------------- #

BASE_DIR = r"C:\Users\Angshudeep Majumdar\Downloads\Githubs_Code_RUVISION_Final"
PICKLE_DIR = os.path.join(BASE_DIR, 'Pickle')
PLOT_DIR = os.path.join(BASE_DIR, 'Plot')

#Ahmedabad City based Forecast
LAT_BOUNDS = [22.25, 23.5]
LON_BOUNDS = [72.0, 73.25]
#Taking 06 hrs UTC of the Current Date as Initialisation hrs or 11:30 AM IST
INIT_TIMES = [6]
#Forecast Starts from 2:30 AM of next day and continuing upto 72 hrs or 3 days, ending at 23:30 hrs
FORECAST_HOURS = list(range(15, 85, 3))
GRID_SIZE = 25
# Vertical and Horrizontal Components of winds at 1000 mbar pressure level and Precipitation rate_GFS variables as input to model
VARIABLES = ['U1000', 'V1000', 'PREC']
LEAD_DAYS = [1, 2, 3]

GFS_FILES = {
    day: (f'PREC_Ahmedabad_LD_{day}_daily basis.xlsx',
          f'U1000_Ahmedabad_LD_{day}_daily basis.xlsx',
          f'V1000_Ahmedabad_LD_{day}_daily basis.xlsx')
    for day in LEAD_DAYS
}

OBS_FILES = {
    1: os.path.join(BASE_DIR, 'IMD_2015-2024_Daily Data_0.25 resolution Rain_OBS_LD_1.xlsx'),
    2: os.path.join(BASE_DIR, 'IMD_2015-2024_Daily Data_0.25 resolution Rain_OBS_LD_2.xlsx'),
    3: os.path.join(BASE_DIR, 'IMD_2015-2024_Daily Data_0.25 resolution Rain_OBS_LD_3.xlsx'),
}

JJAS_RANGES = {
    1: ('2025-06-01', '2025-09-28'),
    2: ('2025-06-02', '2025-09-29'),
    3: ('2025-06-03', '2025-09-30'),
}
#Quantile Regression threshold values
THRESHOLDS = {'80p': 0.2, '85p': 0.15, '90p': 0.1, '95p': 0.05, '99p': 0.01}
#Current Date
TODAY_STR = datetime.now().strftime('%Y-%m-%d')

# ------------------------- GFS PROCESSING ------------------------ #

#Create folders in names of variables where the downloaded files will be stored
def create_output_folder(variable):
    path = os.path.join(BASE_DIR, variable)
    os.makedirs(path, exist_ok=True)
    return path
    
#Building a custom download link for specific variable at a specific level, region, forecast hrs and initialization.
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
#Download variable data initialized on current date and store in the directory
def download_grib_files(variable):
    output_dir = create_output_folder(variable)
    current_date = datetime.now(timezone.utc)
    #current_date = current_date - timedelta(days=1)
    
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

#Check what is there in the grib file and extract information
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


#Preprocessing Data where the variables are instantaneous like wind components, temperature or RH
def preprocess_wind(variable, directory):
    #For row index DateTime on current date at 06 hrs initialization
    ts_utc = pd.Timestamp.utcnow().normalize() + pd.Timedelta(hours=6)
    #ts_utc = pd.Timestamp.utcnow().normalize()- pd.Timedelta(days=1) + pd.Timedelta(hours=6)
    
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
    #Convert from UTC to IST
    df = df.shift(freq=pd.Timedelta(hours=5, minutes=30))
    df.index.name = 'DateTime'
    df = df[df.index.time == pd.Timestamp("11:30").time()]
    return split_and_reshape_data(df, variable)

#Preprocessing Data where the variables are accumulative in nature like Precipitation Rate converted to Acc Prec  

def preprocess_precipitation(variable, directory):
    latbounds = LAT_BOUNDS
    lonbounds = LON_BOUNDS
    time_from_ref = FORECAST_HOURS

    #For row index DateTime on current date at 06 hrs initialization
    ts_06utc = pd.Timestamp.utcnow().normalize() + pd.Timedelta(hours=6)
    #ts_06utc = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=6)

    #Create Dataframe
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

        #Create time index and create a temporary DataFrame (data_temp) to store values at 3-hour intervals
        date_temp = pd.date_range(start=time_step + timedelta(hours=15),
                              end=time_step + timedelta(hours=84), freq='3h')
        data_temp = pd.DataFrame(index=date_temp, columns=np.arange(25))
        
        #file name format
        for time_lag in time_from_ref:
            filename = f'gfs.{year}{month:02d}{day:02d}.t{ref_time:02d}z.pgrb2.0p25.f{time_lag:03d}'
            filepath = os.path.join(directory, filename)
            print(f"[INFO] Accessing file: {filepath}")  # 👈 This line will print the full path

            try:
                #Find what are there inside grib file
                grbs = pygrib.open(filepath)
                grb = grbs.select(name='Precipitation rate')[0]
                temp = grb.values
                lats, lons = grb.latlons()
                #Flatten/Reshape Lats and Lons
                lats_reshaped = lats[:,0]  # Reshape latitudes to (189,)
                reversed_arr = lats_reshaped[::-1]
                lons_reshaped = lons[0,:]  # Reshape longitudes to (,201)
                lats=reversed_arr
                lons=lons_reshaped

                # latitude lower and upper index 
                #Find Index Ranges for Cropping
                latli = np.argmin( np.abs( reversed_arr - latbounds[1] ) )
                latui = np.argmin( np.abs( reversed_arr - latbounds[0] ) ) 

                # longitude lower and upper index
                lonli = np.argmin( np.abs( lons_reshaped- lonbounds[0] ) )
                lonui = np.argmin( np.abs( lons_reshaped - lonbounds[1] ) )  
                
                #Extract and Flip Region Data
                data = temp[latli:latui, lonli:lonui][::-1]
                
                #Align Data with Forecast Time
                time = time_step + timedelta(hours=int(time_lag))
                time_prev = time - timedelta(hours=3)
                
                #conversion of precipitation rate data (kg/m²/s) into accumulated rainfall (mm) for each 3-hourly time step and each of 25 grid points
                if data.size == 25:
                    #Handle 6-hour Accumulation Forecast, kg/m2/s to kg/m2; So divide by 6*60*60= 21600 
                    if time_lag % 6 == 0:
                        if time_prev in data_temp.index:
                            #This 6-hour value includes the last 3 hours, which are already captured (at time_prev) if using 3-hourly data.
                            data_temp.loc[time][0:25] = (np.ravel(data)*21600) - np.ravel(data_temp.loc[time_prev][0:25])
                            
                    #Handle 3 hr forecast lags
                    elif time_lag % 3 == 0:
                        data_temp.loc[time][0:25] = (np.ravel(data)*10800)

            except Exception as e:
                print(f"[ERROR] {filename}: {e}")
        #Store Flattened Result in Main DataFrame
        data_prec.loc[time_step][0:600] = np.ravel(data_temp)

        counter += 1
        if counter % 100 == 0:
            print(f'Loop {counter} Done!')

    print("\n[DEBUG] Raw data_prec before timezone shift and filtering:")
    print(data_prec)
    #Convert from UTC to IST
    data_prec = data_prec.shift(freq=pd.Timedelta(hours=5, minutes=30))
    data_prec.index.name = 'DateTime'
    extracted = data_prec[data_prec.index.time == pd.Timestamp("11:30").time()]
    
    return split_and_reshape_data(extracted, variable)

#Dividing data_prec data into 3 lead days; 15-36 hrs= Lead Day 1, 39-60 hrs= Lead Day 2 and 63-84 hrs= Lead Day 3
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

#If var=prec we sum lead hours to get lead day and if not, we take average of lead hour forecasts
#Concat with historical dataset
#save in directory
def aggregate_and_save(daily_data, variable):
    today = pd.Timestamp.today().normalize()
    #today = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    
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

        filename = os.path.join(BASE_DIR, f"{variable}_Ahmedabad_LD_{day}_daily basis.xlsx")

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

def preprocess_variable(variable, directory):
    if variable == "PREC":
        return preprocess_precipitation(variable, directory)
    else:
        return preprocess_wind(variable, directory)

# ----------------------- PCA & CQM MODELING ----------------------- #


#Pick excel files from directory updated every day and combine U1000, V1000, Prec for each lead day
#Normalize and Conduct PCA
#Saved transformed data in pickle format into directory
def preprocess_and_save_pca(lday):
    prec_file, u_file, v_file = GFS_FILES[lday]
    prec = pd.read_excel(os.path.join(BASE_DIR, prec_file)).set_index('DateTime')
    u = pd.read_excel(os.path.join(BASE_DIR, u_file)).set_index('DateTime')
    v = pd.read_excel(os.path.join(BASE_DIR, v_file)).set_index('DateTime')

    df = pd.concat([u, v, prec], axis=1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled, columns=df.columns)

    pca = PCA(0.99)
    transformed = pca.fit_transform(df_scaled)
    transformed_df = pd.DataFrame(transformed, index=df.index)

    filename = f'transformed_dataX_LD_{lday}_U1000_V1000_PREC.pkl'
    transformed_df.to_pickle(os.path.join(PICKLE_DIR, filename))
    
#Load Pickle File and Observed Data 
def load_data(pkl_path, obs_path):
    X_full = pd.read_pickle(pkl_path)
    Y = pd.read_excel(obs_path).set_index('DateTime')
    Y.index = pd.to_datetime(Y.index)
    X = X_full.iloc[:len(Y)]
    X.index = Y.index
    return X, Y['Prec_23.0_72.5'], X_full

#Divide into train and test data
def split_data(X, Y, split_index=3269):
    return X.iloc[:split_index], X.iloc[split_index+3:], Y.iloc[:split_index], Y.iloc[split_index+3:]

    #If Rain>0, Y_train converted to a binary classification target
    #GLM Model fitted and predicted on X_train 
    def binary_glm(X_train, Y_train):
    y_binary = (Y_train > 0).astype(int)
    glm = sm.GLM(y_binary, sm.add_constant(X_train), family=sm.families.Binomial()).fit()
    return glm.predict(sm.add_constant(X_train))

#Building a CQM pipeline for probabilistic rainfall prediction.
#Binary classification (to detect "rain or no rain")
#Quantile regression (to predict different levels of rainfall)
#Filtering of data to account for the zero-inflated nature of rainfall data

def quantile_regression(X, Y, q):
    return sm.QuantReg(Y, sm.add_constant(X)).fit(q=q).params

def reconstruct_rain(X, coefs):
    return X @ coefs[1:] + coefs[0]
    
#get rain probability (P(rain > 0))
def cqm_pipeline(X_train, Y_train, X_test, thresholds):
    yfit_train = binary_glm(X_train, Y_train)
    coefs_dict = {}
    rain_dict = {}
    
    #Each label is with corresponding probability thresholds like like '80p': 0.2, '85p': 0.15, '90p': 0.1, '95p': 0.05, '99p': 0.01
    for label, t in thresholds.items():
        q_val = float(label.replace('p', '')) / 100
        mask1 = yfit_train > t
        #Select only the training data where rain is likely (probability > threshold)
        X1, Y1 = X_train[mask1], Y_train[mask1]
        if X1.empty or Y1.empty:
            continue
        #Perform 1st stage Quantile regression and reconstructs or target value using the learned model coefficients
        coef1 = quantile_regression(X1, Y1, q_val)
        rain1 = reconstruct_rain(X1, coef1)
        #Remove rain values equals to or lesser than 0
        mask2 = rain1 > 0
        #Select only the training data where rain is likely (probability > threshold)
        X2, Y2 = X1[mask2], Y1[mask2]
        if X2.empty or Y2.empty:
            continue
        #Perform 2nd stage Quantile regression and reconstructs or target value using the learned model coefficients
        coef2 = quantile_regression(X2, Y2, q_val)
        rain_test = reconstruct_rain(X_test, coef2)
        coefs_dict[label] = coef2
        rain_dict[label] = np.maximum(rain_test, 0)

    return coefs_dict, rain_dict
    
#Performs quantile-based rainfall forecasting for a given lead time using a trained Censored Quantile Regression pipeline
#Data Prediction for JJAS months for 3 lead days (Real time)
#Save results and export in Excel for various Quantiles
#extracted 80p quantile value for each lead day

def forecast_future_rain(lday):
    pattern = rf'transformed_dataX_LD_{lday}_(.+)\.pkl'
    obs_path = OBS_FILES[lday]
    jjas_start, jjas_end = pd.Timestamp(JJAS_RANGES[lday][0]), pd.Timestamp(JJAS_RANGES[lday][1])

    for pkl in glob.glob(os.path.join(PICKLE_DIR, f'transformed_dataX_LD_{lday}_*.pkl')):
        var_match = re.search(pattern, os.path.basename(pkl))
        if not var_match:
            continue

        var_comb = var_match.group(1)
        out_dir = os.path.join(PLOT_DIR, var_comb)
        os.makedirs(out_dir, exist_ok=True)

        X, Y, X_full = load_data(pkl, obs_path)
        X_train, X_test, Y_train, Y_test = split_data(X, Y)
        coefs2, _ = cqm_pipeline(X_train, Y_train, X_test, THRESHOLDS)

        X_future = X_full.iloc[len(Y):]
        if X_future.empty:
            continue

        future_dates = pd.date_range(start=Y.index[-1] + pd.Timedelta(days=1), periods=len(X_future), freq='D')
        X_future.index = future_dates

        rain_future = {
            label: np.maximum(reconstruct_rain(X_future, coefs), 0)
            for label, coefs in coefs2.items()
        }

        mask_jjas = (X_future.index >= jjas_start) & (X_future.index <= jjas_end)
        rain_jjas = {
            label: pd.Series(data, index=X_future.index)[mask_jjas]
            for label, data in rain_future.items()
        }

        df_jjas = pd.DataFrame(rain_jjas)
        df_jjas.index = df_jjas.index.date

        excel_path = os.path.join(out_dir, f'CQM_QuantileForecast_{var_comb}_2025JJAS_LD_{lday}.xlsx')
        df_jjas.to_excel(excel_path, index_label='Date')
        print(f"Saved forecast to: {excel_path}")

        if '80p' in df_jjas.columns:
            yield df_jjas['80p'].iloc[-1:], lday
            
#Plot future lead days taking current date as initialization

def plot_last_3_days_bar(pred_series_list):
    combined = pd.concat([s for s, _ in pred_series_list])
    combined.index = [f"{i.strftime('%d-%b')}" for (s, lday), i in zip(pred_series_list, combined.index)]

    fig, ax = plt.subplots(figsize=(15, 8))
    bars = ax.bar(combined.index, combined.values, color='blue')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height * 0.5, f'{height:.1f}', ha='center', va='center', fontsize=12)

    ax.set_title(f"3-day Rainfall Forecast (Statistical Downscaling) - Ahmedabad - {TODAY_STR}", fontsize=14)
    ax.set_ylabel("Rainfall (mm)")
    ax.set_xlabel("Date")
    plt.xticks(rotation=45)

    save_path = os.path.join(PLOT_DIR, f'forecast_3_Lead_Days_{TODAY_STR}_Initialisation.png')
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Bar plot saved to: {save_path}")

# ----------------------------- MAIN FUNCTION----------------------------- #

def main():
    # Step 1: Download & Process GFS Data
    for variable in VARIABLES:
        print(f"[INFO] Processing Variable: {variable}")
        output_dir = download_grib_files(variable)
        daily_data = preprocess_variable(variable, output_dir)
        aggregate_and_save(daily_data, variable)

    # Step 2: PCA & Pickle
    for lday in LEAD_DAYS:
        preprocess_and_save_pca(lday)

    # Step 3: Forecast + Plot
    bar_series = list(forecast_future_rain(lday) for lday in LEAD_DAYS)
    flat_series = [item for sublist in bar_series for item in sublist]
    plot_last_3_days_bar(flat_series)

# ------------------------- EXECUTE ------------------------- #

if __name__ == "__main__":
    main()


# In[ ]:





