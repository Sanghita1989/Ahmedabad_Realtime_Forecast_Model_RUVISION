#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import os
from datetime import datetime, timedelta

# ---------------------------- CONFIG ----------------------------
#BASE_DIR = "/home/subhojit/Githubs_Code_RUVISION_Final" 

def parse_args():
    parser = argparse.ArgumentParser(description="Rainfall Forecasting Pipeline")

    # Base dirs
    parser.add_argument("--base_dir", type=str,
                        default="/home/subhojit/Githubs_Code_RUVISION_Final",
                        help="Base directory for input/output files")

    # City name (for file naming)
    parser.add_argument("--city", type=str, default="Ahmedabad",
                        help="City name for labeling files (e.g., Ahmedabad, Mumbai)")

    # Geographic domain (for processing, not download)
    parser.add_argument("--lat1", type=float, default=22.25, help="Lower latitude bound")
    parser.add_argument("--lat2", type=float, default=23.5, help="Upper latitude bound")
    parser.add_argument("--lon1", type=float, default=72.0, help="Lower longitude bound")
    parser.add_argument("--lon2", type=float, default=73.25, help="Upper longitude bound")

    # Model setup
    parser.add_argument("--init_times", type=int, nargs="+", default=[6],
                        help="Initialization hours (UTC)")
    parser.add_argument("--forecast_hours", type=int, nargs="+",
                        default=list(range(15, 85, 3)),
                        help="Forecast hours list")
    parser.add_argument("--grid_size", type=int, default=25,
                        help="Grid size for extraction")
    parser.add_argument("--variables", type=str, nargs="+",
                        default=["U1000", "V1000", "PREC"],
                        help="Variables to process")
    parser.add_argument("--lead_days", type=int, nargs="+", default=[1, 2, 3],
                        help="Lead days for forecasting")

    # JJAS bounds
    parser.add_argument("--jjas_start", type=str, default="2025-06-01", help="JJAS start date")
    parser.add_argument("--jjas_end", type=str, default="2025-09-30", help="JJAS end date")

    return parser.parse_args()


def build_config(args):
    # ---------------- OBS Files ---------------- #
    obs_files = {}
    for ld in args.lead_days:
        filename = f"OBS_LD_{ld}.xlsx"
        filepath = os.path.join(args.base_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Expected OBS file not found: {filepath}")
        obs_files[ld] = filepath

    # ---------------- GFS Files ---------------- #
    gfs_files = {
        ld: {
            var: os.path.join(
                args.base_dir,
                f"{var}_{args.city}_LD_{ld}_daily basis.xlsx"   # ✅ uses city + variable dynamically
            )
            for var in args.variables
        }
        for ld in args.lead_days
    }

    # ---------------- Final Config ---------------- #
    return {
        "BASE_DIR": args.base_dir,
        "CITY": args.city,
        "PICKLE_DIR": os.path.join(args.base_dir, "Pickle"),
        "PLOT_DIR": os.path.join(args.base_dir, "Plot"),

        # Domain
        "LAT_BOUNDS": [args.lat1, args.lat2],
        "LON_BOUNDS": [args.lon1, args.lon2],

        # Model setup
        "INIT_TIMES": args.init_times,
        "FORECAST_HOURS": args.forecast_hours,
        "GRID_SIZE": args.grid_size,
        "VARIABLES": args.variables,
        "LEAD_DAYS": args.lead_days,

        # Data files
        "OBS_FILES": obs_files,
        "GFS_FILES": gfs_files,

        # JJAS period
        "JJAS_RANGES": {day: (args.jjas_start, args.jjas_end) for day in args.lead_days},

        # Probability thresholds
        "THRESHOLDS": {"80p": 0.2, "85p": 0.15, "90p": 0.1, "95p": 0.05, "99p": 0.01},

        # GFS download setup
        "SUBREGION": {"toplat": 47, "bottomlat": 0, "leftlon": 55, "rightlon": 105},
        "BASE_URL": "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl",
        "VAR_FLAGS": {
            "U1000": "var_UGRD=on&lev_1000_mb=on",
            "V1000": "var_VGRD=on&lev_1000_mb=on",
            "PREC": "var_PRATE=on&lev_surface=on",
        },

        # Runtime info
        "TODAY_STR": datetime.now().strftime("%Y-%m-%d")
    }

