#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from config import VARIABLES, LEAD_DAYS, TODAY_STR, PLOT_DIR
from download import download_grib_files
from preprocessing import preprocess_variable, aggregate_and_save
from modeling import preprocess_and_save_pca, forecast_future_rain
from postprocessing import plot_last_3_days_bar
import os

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

if __name__ == "__main__":
    main()

