#!/usr/bin/env python
# coding: utf-8

import os
from config import parse_args, build_config
from download import download_grib_files
from preprocessing import preprocess_variable, aggregate_and_save
from modeling import preprocess_and_save_pca, forecast_future_rain
from postprocessing import plot_last_3_days_bar


def main():
    # Load config from arguments
    args = parse_args()
    config = build_config(args)

    # Step 1: Download & Process GFS Data
    for variable in config["VARIABLES"]:
        print(f"[INFO] Processing Variable: {variable}")
        output_dir = download_grib_files(config, variable)
        daily_data = preprocess_variable(variable, output_dir, config)
        aggregate_and_save(daily_data, variable, config)

    # Step 2: PCA & Pickle
    for lday in config["LEAD_DAYS"]:
        preprocess_and_save_pca(lday, config)

    # Step 3: Forecast + Plot
    pred_series_list = []
    for lday in config["LEAD_DAYS"]:
        for result in forecast_future_rain(lday, config):
            pred_series_list.append(result)

    if pred_series_list:  # only plot if we actually have forecasts
        plot_last_3_days_bar(pred_series_list, config)
    else:
        print("[WARNING] No forecasts available to plot.")


if __name__ == "__main__":
    main()

