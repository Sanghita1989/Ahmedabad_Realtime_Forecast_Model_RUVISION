#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import json
import pandas as pd
from datetime import datetime

# 📂 Paths
base_folder = r"C:\Users\Angshudeep Majumdar\Downloads\Githubs_Code_RUVISION_Final\IMD_API"
output_excel = os.path.join(base_folder, "Rainfall_Data.xlsx")

rows = []

# 🔍 Collect all date folders
for subfolder in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, subfolder)
    if os.path.isdir(folder_path):
        try:
            # Ensure folder name is a valid date
            folder_date = datetime.strptime(subfolder, "%Y-%m-%d").date()  # <-- strictly date
            json_file = os.path.join(folder_path, "city_weather_forecast.json")

            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    data = json.load(f)

                if isinstance(data, list) and len(data) > 0:
                    record = data[0]
                    rainfall = record.get("Past_24_hrs_Rainfall", None)
                    station = record.get("Station_Name", None)
                else:
                    rainfall, station = None, None

                rows.append([folder_date, station, rainfall])
        except Exception as e:
            print(f"⚠️ Skipped {subfolder}: {e}")

# 📝 Build dataframe
df_new = pd.DataFrame(rows, columns=["Date", "Station_Name", "Past_24_hrs_Rainfall"]).sort_values("Date")

if not df_new.empty:
    # take only latest date’s data
    last_date = df_new["Date"].max()
    df_latest = df_new[df_new["Date"] == last_date]

    # 📖 Load existing excel
    if os.path.exists(output_excel):
        df_existing = pd.read_excel(output_excel)

        # 🔧 Ensure Date column is strictly date type
        df_existing["Date"] = pd.to_datetime(df_existing["Date"]).dt.date

        # Avoid duplicates for last_date
        df_existing = df_existing[df_existing["Date"] != last_date]

        df_final = pd.concat([df_existing, df_latest], ignore_index=True)
    else:
        df_final = df_latest

    # 🔧 Ensure Date column is strictly date type before saving
    df_final["Date"] = pd.to_datetime(df_final["Date"]).dt.date

    # 💾 Save back
    df_final.to_excel(output_excel, index=False)
    print(f"✅ Updated Excel with data from {last_date}! Saved to {output_excel}")
else:
    print("⚠️ No data found in subfolders.")


# In[ ]:




