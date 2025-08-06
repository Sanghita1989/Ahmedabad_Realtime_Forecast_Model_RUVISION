🌧️ Ahmedabad Realtime Forecast Model — RUVISION
👉 Overview
This repository runs an operational rainfall forecasting model for Ahmedabad using GFS reanalysis data and observed IMD rainfall. The model has two major components:

Data Download & Processing

Forecast Modeling (Lead Day 1–3) & Plotting

📌 Variables of Interest
Code	Description
U1000	U-component of wind at 1000 mb
V1000	V-component of wind at 1000 mb
PREC	Precipitation rate from GFS

Observed Rainfall: IMD gridded daily rainfall at 0.25° resolution
📖 Reference: Pai et al., MAUSAM (2014)

Training Period: 2015–2023

Testing / Realtime Forecast: 2024–present

⏱️ Initialization Details
Initialization Hour: 06 UTC (11:30 AM IST)

Forecast Hours:

Lead Day 1: 15–36 hrs

Lead Day 2: 39–60 hrs

Lead Day 3: 63–84 hrs

Grid Size: 5x5 region over Ahmedabad

Forecast Method: Two-stage Censored Quantile Regression (CQR)

🎯 Forecast Objective
To forecast daily rainfall in Ahmedabad for:

Tomorrow

Day after tomorrow

Third day from initialization

🧠 Model Setup Workflow
1. Data Collection
Literature study for variable selection

Download historical GFS data (0.25°) from NCEP/NOMADS

Restrict lat/lon to Ahmedabad city region

2. Data Processing
Save historical data (for each lead day) into Excel

Preprocessing functions:

preprocess_wind() for wind (U1000, V1000)

preprocess_precipitation() for precipitation (rate → accumulation)

Convert GFS hourly to daily:

Precipitation: sum hourly values

Wind: average hourly values

Merge realtime data with past Excel files

3. Modeling
For each lead day:

Combine U1000, V1000, PREC

Normalize & apply PCA

Save transformed features in .pkl

Run GLM-based CQR for quantile forecasting (10th, 50th, 80th percentiles)

⚙️ Realtime Operations
Create base path & folders for each variable

Construct download URLs from NOMADS

Download GFS .grb2 data (06z run)

Preprocess and split into 3 lead days

Append new data to historical Excel

Run quantile regression forecast

Output Excel with daily quantile predictions

🚀 Automation
Code refactored & modularized

Deployed on HPC cluster

Scheduled daily at 5 PM IST

Output forecasts hosted on: clipre.ai

🛠️ Additional Tasks
✅ GitHub Code Review

✅ Experimented with 100+ variable combinations

✅ Scoring Metrics: Accuracy, F1, Recall, HSS, CQVSS

✅ Selected best-performing variable set


