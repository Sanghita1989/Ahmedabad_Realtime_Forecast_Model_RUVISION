#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import os
from datetime import datetime, timedelta

# ---------------------------- CONFIG ----------------------------
#BASE_DIR = "/home/subhojit/Githubs_Code_RUVISION_Final" 



def parse_args():
    parser = argparse.ArgumentParser(description="Rainfall Forecasting Pipeline for Ahmedabad")

    # Base directory for input/output
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory for input/output files")

    # Geographic domain
    parser.add_argument("--lat1", type=float, default=22.25, help="Lower latitude bound")
    parser.add_argument("--lat2", type=float, default=23.5, help="Upper latitude bound")
    parser.add_argument("--lon1", type=float, default=72.0, help="Lower longitude bound")
    parser.add_argument("--lon2", type=float, default=73.25, help="Upper longitude bound")
    parser.add_argument("--grid_size", type=int, default=25, help="Grid size for extraction (number of grid points)")

    # Initialization times and forecast hours
    parser.add_argument("--init_times", type=int, nargs="+", default=[6], help="Initialization hours UTC")
    parser.add_argument("--forecast_hours", type=int, nargs="+", default=list(range(15, 85, 3)), help="Forecast hours list")

    # Variables and lead days
    parser.add_argument("--variables", type=str, nargs="+", default=["U1000", "V1000", "PREC"], help="Variables to process")
    parser.add_argument("--lead_days", type=int, nargs="+", default=[1,2,3], help="Lead days for forecasting")

    # JJAS period
    parser.add_argument("--jjas_start", type=str, default="2025-06-01", help="JJAS start date")
    parser.add_argument("--jjas_end", type=str, default="2025-09-28", help="JJAS end date")

    return parser.parse_args()

#OBSERVED VARIABLES
def build_config(args):
    obs_files = {}
    for ld in args.lead_days:
        pattern = os.path.join(args.base_dir, f"IMD_2015-2024_Daily Data_0.25 resolution Rain_OBS_LD_{ld}.xlsx")
        matched_files = glob.glob(pattern)
        if matched_files:
            obs_files[ld] = matched_files[0]  # Take the first match
        else:
            raise FileNotFoundError(f"No OBS file found for Lead Day {ld} in {args.base_dir}")
    return {
        "BASE_DIR": args.base_dir,
        "PICKLE_DIR": os.path.join(args.base_dir, "Pickle"),
        "PLOT_DIR": os.path.join(args.base_dir, "Plot"),
        "LAT_BOUNDS": [args.lat1, args.lat2],
        "LON_BOUNDS": [args.lon1, args.lon2],
        "INIT_TIMES": args.init_times,
        "FORECAST_HOURS": args.forecast_hours,
        "GRID_SIZE": args.grid_size,
        "VARIABLES": args.variables,
        "LEAD_DAYS": args.lead_days,
        "JJAS_RANGES": {day: (args.jjas_start, args.jjas_end) for day in args.lead_days},
        "THRESHOLDS": {"80p": 0.2, "85p": 0.15, "90p": 0.1, "95p": 0.05, "99p": 0.01},
        "SUBREGION": {"toplat": 47, "bottomlat": 0, "leftlon": 55, "rightlon": 105},
        #Download GFS Data
        "BASE_URL": "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl",
        "VAR_FLAGS": {"U1000": "var_UGRD=on&lev_1000_mb=on",
                      "V1000": "var_VGRD=on&lev_1000_mb=on",
                      "PREC": "var_PRATE=on&lev_surface=on"},
        "TODAY_STR": datetime.now().strftime("%Y-%m-%d")
    }
