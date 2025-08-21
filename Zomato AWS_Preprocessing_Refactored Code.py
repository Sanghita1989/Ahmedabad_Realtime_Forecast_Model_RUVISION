#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd

def list_csv_files(folder_path):
    """List all CSV files in a given folder."""
    return [f for f in os.listdir(folder_path) if f.endswith('.csv')]

def read_and_process_csv(file_path):
    """Read a CSV and convert 'device_date_time' to datetime."""
    df = pd.read_csv(file_path)
    df['device_date_time'] = pd.to_datetime(df['device_date_time'], errors='coerce')
    return df

def load_all_data(folder_path):
    """Load and concatenate all CSVs in the folder."""
    csv_files = list_csv_files(folder_path)
    df_list = [read_and_process_csv(os.path.join(folder_path, f)) for f in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def create_pivot_table(df):
    """Create pivot table from combined dataframe."""
    pivot_df = df.pivot_table(
        index='device_date_time',
        columns='locality_name',
        values='rain_intensity',
        aggfunc='mean'
    )
    pivot_df.sort_index(inplace=True)
    return pivot_df

def resample_daily_sum(pivot_df, skip_days=13):
    """Resample to daily sums, skipping initial rows."""
    daily_df = pivot_df.resample('D').sum()
    daily_df = daily_df.iloc[skip_days:, :]
    daily_df.index.name = 'DateTime'
    return daily_df

def add_ahmedabad_average(daily_df):
    """Add 'Ahmedabad_City' as the mean of all columns per day."""
    daily_df['Ahmedabad_City'] = daily_df.mean(axis=1)
    return daily_df

def append_to_excel(df, file_path):
    """Append last 3 rows to existing Excel, ensuring dates only (YYYY-MM-DD)."""
    last_rows = df.tail(3)

    # Ensure DateTime is date-only string
    last_rows = last_rows.reset_index()
    last_rows['DateTime'] = pd.to_datetime(last_rows['DateTime']).dt.strftime("%Y-%m-%d")
    last_rows = last_rows.set_index('DateTime')

    if os.path.exists(file_path):
        existing_df = pd.read_excel(file_path, index_col=0)

        # Convert existing index to string YYYY-MM-DD for consistency
        existing_df.index = pd.to_datetime(existing_df.index).strftime("%Y-%m-%d")

        # Avoid duplicate dates
        last_rows = last_rows[~last_rows.index.isin(existing_df.index)]

        updated_df = pd.concat([existing_df, last_rows])
    else:
        updated_df = last_rows

    updated_df.to_excel(file_path)

def main():
    older_path = r"C:\Users\Angshudeep Majumdar\Downloads\Githubs_Code_RUVISION_Final\weatherunion_data"
    output_path = r"C:\Users\Angshudeep Majumdar\Downloads\Githubs_Code_RUVISION_Final\Plot\U1000_V1000_PREC\DailyRain_Ahmedabad.xlsx"

    combined_df = load_all_data(older_path)
    pivot_df = create_pivot_table(combined_df)
    daily_df = resample_daily_sum(pivot_df, skip_days=13)
    daily_df = add_ahmedabad_average(pivot_df)

    # Convert DateTime index to plain dates (YYYY-MM-DD)
    daily_df = daily_df.reset_index()
    daily_df['DateTime'] = pd.to_datetime(daily_df['DateTime']).dt.strftime("%Y-%m-%d")
    daily_df = daily_df.set_index('DateTime')

    append_to_excel(daily_df, output_path)

if __name__ == "__main__":
    main()


# In[ ]:




