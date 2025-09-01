#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import re
import requests
import pygrib
import numpy as np
from datetime import datetime, timedelta, timezone
from config import BASE_DIR, INIT_TIMES, FORECAST_HOURS, GRID_SIZE

def create_output_folder(variable):
    path = os.path.join(BASE_DIR, variable)
    os.makedirs(path, exist_ok=True)
    return path

def generate_url(current_date, init_hour, forecast_hour, variable):
    base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
    var_flags = {
        "U1000": "var_UGRD=on&lev_1000_mb=on",
        "V1000": "var_VGRD=on&lev_1000_mb=on",
        "PREC": "var_PRATE=on&lev_surface=on"
    }
    return (
        f"{base_url}?dir=%2Fgfs.{current_date.strftime('%Y%m%d')}%2F{init_hour:02d}%2Fatmos&"
        f"file=gfs.t{init_hour:02d}z.pgrb2.0p25.f{forecast_hour:03d}&{var_flags[variable]}&"
        f"subregion=&toplat=47&leftlon=55&rightlon=105&bottomlat=0"
    )

def download_grib_files(variable):
    output_dir = create_output_folder(variable)
    current_date = datetime.now(timezone.utc)
    #current_date = datetime.now(timezone.utc) - timedelta(days=1)

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

