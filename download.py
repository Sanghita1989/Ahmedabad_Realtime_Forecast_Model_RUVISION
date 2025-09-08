#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import re
import requests
import pygrib
import numpy as np
from datetime import datetime, timedelta, timezone
from .config import build_config

def create_output_folder(base_dir,variable):
    path = os.path.join(base_dir, variable)
    os.makedirs(path, exist_ok=True)
    return path
    
def generate_url(config, current_date, init_hour, forecast_hour, variable):
    var_flag = config["VAR_FLAGS"][variable]
    return (f"{config['BASE_URL']}?dir=%2Fgfs.{current_date.strftime('%Y%m%d')}%2F{init_hour:02d}%2Fatmos&"
            f"file=gfs.t{init_hour:02d}z.pgrb2.0p25.f{forecast_hour:03d}&{var_flag}&"
            f"subregion=&toplat={config['SUBREGION']['toplat']}&leftlon={config['SUBREGION']['leftlon']}&"
            f"rightlon={config['SUBREGION']['rightlon']}&bottomlat={config['SUBREGION']['bottomlat']}")
    
def download_grib_files(config,variable, current_date):
    output_dir = create_output_folder(config["BASE_DIR"], variable)
    
    for init_time in config["INIT_TIMES"]:
        for fh in config["FORECAST_HOURS"]:
            url = generate_url(config, current_date, init_time, fh, variable)
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





