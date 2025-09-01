#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from datetime import datetime, timedelta

# ---------------------------- CONFIG ----------------------------

BASE_DIR = input("Enter the base directory for your project: ").strip()

# Ensure it exists
if not os.path.exists(BASE_DIR):
    raise FileNotFoundError(f"❌ The path {BASE_DIR} does not exist.")

#BASE_DIR = "/home/subhojit/Githubs_Code_RUVISION_Final"   
PICKLE_DIR = os.path.join(BASE_DIR, 'Pickle')
PLOT_DIR = os.path.join(BASE_DIR, 'Plot')

# Ahmedabad City based Forecast
LAT_BOUNDS = [22.25, 23.5]
LON_BOUNDS = [72.0, 73.25]

# Initialization and forecast hours
INIT_TIMES = [6]  # 06 UTC (11:30 AM IST)
FORECAST_HOURS = list(range(15, 85, 3))  # 15–84 hrs

GRID_SIZE = 25
VARIABLES = ['U1000', 'V1000', 'PREC']
LEAD_DAYS = [1, 2, 3]

# File names
GFS_FILES = {
    day: (
        f'PREC_Ahmedabad_LD_{day}_daily basis.xlsx',
        f'U1000_Ahmedabad_LD_{day}_daily basis.xlsx',
        f'V1000_Ahmedabad_LD_{day}_daily basis.xlsx'
    )
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

# Quantile Regression thresholds
THRESHOLDS = {'80p': 0.2, '85p': 0.15, '90p': 0.1, '95p': 0.05, '99p': 0.01}

# Current Date (yesterday’s run)
TODAY_STR = datetime.now().strftime('%Y-%m-%d')
#TODAY_STR = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')


# In[ ]:




