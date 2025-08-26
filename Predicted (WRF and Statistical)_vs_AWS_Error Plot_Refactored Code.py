#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.dates as mdates

# -------------------------------
# Global plotting settings
# -------------------------------
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

# -------------------------------
# File paths
# -------------------------------
folder = 'C:/Users/Angshudeep Majumdar/Downloads/Githubs_Code_RUVISION_Final/Plot/U1000_V1000_PREC'
save_folder = 'C:/Users/Angshudeep Majumdar/Downloads/Githubs_Code_RUVISION_Final/Plot'
imd_file = "C:/Users/Angshudeep Majumdar/Downloads/Githubs_Code_RUVISION_Final/IMD_API/Rainfall_Data.xlsx"
wrf_file = r"C:/Users/Angshudeep Majumdar/Downloads/file.xlsx"  # Uploaded WRF data file

files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.xlsx')])

# -------------------------------
# Load and clean Excel columns safely
# -------------------------------
def load_and_clean_excel(file, col_idx):
    df = pd.read_excel(file, index_col=0)
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[~df.index.isna()]
    df = df.groupby(df.index).mean()
    col_series = df.iloc[:, col_idx]
    return col_series

col_1 = load_and_clean_excel(files[0], 0).iloc[1:]
col_2 = load_and_clean_excel(files[1], 0).iloc[1:]
col_3 = load_and_clean_excel(files[2], 0).iloc[1:]
col_4 = load_and_clean_excel(files[3], 15)

combined_df = pd.concat([col_1, col_2, col_3, col_4], axis=1)
combined_df.columns = ['Predicted_Lead Day_1', 'Predicted_Lead Day_2', 'Predicted_Lead Day_3', 'AWS']
combined_df.index.name = 'Datetime'
combined_df = combined_df[combined_df["AWS"].notna()]
combined_df = combined_df.sort_index().reset_index()

# -------------------------------
# Load IMD data
# -------------------------------
imd_df = pd.read_excel(imd_file, sheet_name="Sheet1")
imd_df["Date"] = pd.to_datetime(imd_df["Date"], errors="coerce")
imd_df["Past_24_hrs_Rainfall"] = pd.to_numeric(imd_df["Past_24_hrs_Rainfall"].replace("NIL", np.nan), errors="coerce")
imd_df = imd_df[["Date", "Past_24_hrs_Rainfall"]].rename(columns={"Date": "Datetime", "Past_24_hrs_Rainfall": "IMD"})
combined_df = pd.merge(combined_df, imd_df, on="Datetime", how="left")

# -------------------------------
# Load WRF data
# -------------------------------
wrf_df = pd.read_excel(wrf_file)
wrf_df.columns = ['Datetime', 'WRF_Lead Day_1', 'WRF_Lead Day_2', 'WRF_Lead Day_3']
wrf_df['Datetime'] = pd.to_datetime(wrf_df['Datetime'], errors='coerce')

# Merge WRF into main dataframe
combined_df = pd.merge(combined_df, wrf_df, on="Datetime", how="left")

# -------------------------------
# Plotting function
# -------------------------------
def plot_combined_timeseries(df, pred_cols, wrf_cols, col_aws='AWS', col_imd='IMD', save_prefix='Predicted', save_folder='.'):
    global_min, global_max = df['Datetime'].min(), df['Datetime'].max()

    for pred_col, wrf_col in zip(pred_cols, wrf_cols):
        # Restrict to rows where prediction is available
        df_sub = df.loc[df[pred_col].notna(), ['Datetime', pred_col, col_aws, col_imd, wrf_col]].copy()

        start_date = df_sub['Datetime'].min()
        df_aws = df.loc[df['Datetime'] >= start_date, ['Datetime', col_aws, col_imd]]

        fig, axs = plt.subplots(3, 1, figsize=(16, 15), sharex=True)

        # --- 1) Line plot
        axs[0].plot(df_sub['Datetime'], df_sub[pred_col], color='blue', label=f'{pred_col} (Statistical Model)(00:00–23:59 hrs)')
        axs[0].plot(df_sub['Datetime'], df_sub[col_aws], color='black', linestyle='--', label=f'{col_aws}(Zomato)(00:00–23:59 hrs)')
        axs[0].scatter(df_sub['Datetime'], df_sub[col_imd], color='deeppink', marker='*', s=120, label='IMD_AWS(08:30–08:30 hrs)')
        axs[0].plot(df_sub['Datetime'], df_sub[wrf_col], color='seagreen', marker='o', label=f'{wrf_col}(WRF Hydro)(00:00–23:59 hrs)')

        axs[0].set_ylabel("Rainfall (mm)")
        axs[0].set_title(f"{pred_col} vs Observed_Zomato_vs_IMD(AWS)_Ahmedabad City Rain Model")
        axs[0].legend(loc="upper left", fontsize=12)
        axs[0].grid(True)

        max_val = df_sub[[pred_col, wrf_col, col_aws, col_imd]].max().max()
        axs[0].set_ylim(0, max_val + 10)

        # ✅ Add vertical lines for Lead Day 2 & 3
        if "Lead Day_2" in pred_col or "Lead Day_3" in pred_col:
            axs[0].axvline(df_sub['Datetime'].iloc[0], color='gray', linestyle='--', linewidth=1.2)

        # --- 2) Error bars
        error_model = df_sub[pred_col] - df_sub[col_aws]
        error_wrf = df_sub[wrf_col] - df_sub[col_aws]

        axs[1].bar(df_sub['Datetime'] - pd.Timedelta(days=0.15), error_model, width=0.3, color='red', label=f'Error {pred_col}')
        axs[1].bar(df_sub['Datetime'] + pd.Timedelta(days=0.15), error_wrf, width=0.3, color='seagreen', label=f'Error {wrf_col}')
        axs[1].axhline(0, color='black', linewidth=1.2)
        axs[1].set_ylabel("Error (mm)")
        axs[1].set_title(f"Error: ({pred_col} - Zomato_{col_aws})")
        axs[1].legend(loc='upper left')
        axs[1].grid(True)

        # --- 3) Percentage Error
        pct_error_model = np.where(df_sub[col_aws] != 0, error_model / df_sub[col_aws], 0)
        pct_error_wrf = np.where(df_sub[col_aws] != 0, error_wrf / df_sub[col_aws], 0)

        axs[2].bar(df_sub['Datetime'] - pd.Timedelta(days=0.15), pct_error_model, width=0.3, color='orange', label=f'% Error {pred_col}')
        axs[2].bar(df_sub['Datetime'] + pd.Timedelta(days=0.15), pct_error_wrf, width=0.3, color='seagreen', label=f'% Error {wrf_col}')
        axs[2].axhline(0, color='black', linewidth=1.2)
        axs[2].set_ylabel("Error Fraction")
        axs[2].set_title(f"Percentage Error:({pred_col} - Zomato_{col_aws}) / Zomato_{col_aws}")
        axs[2].legend(loc='upper left')
        axs[2].grid(True)

        # --- X-axis formatting
        for ax in axs:
            ax.set_xlim(global_min, global_max)
            ax.tick_params(labelbottom=True)
            locator = mdates.DayLocator(interval=4)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.tick_params(axis='x', rotation=90)

        # ✅ Shared X-axis label
        fig.supxlabel("Datetime", fontsize=14)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        save_name = os.path.join(save_folder, f"{save_prefix}_{pred_col}_comparison.png")
        plt.savefig(save_name, dpi=300)
        plt.close(fig)
        print(f"Saved: {save_name}")
       
# --- Function to calculate metrics ---
def calculate_metrics(pred, obs, threshold):
    event_obs = obs >= threshold
    event_pred = pred >= threshold
    
    hits = np.sum(event_pred & event_obs)
    misses = np.sum(~event_pred & event_obs)
    false_alarms = np.sum(event_pred & ~event_obs)
    
    recall = hits / (hits + misses) if (hits + misses) > 0 else 0
    precision = hits / (hits + false_alarms) if (hits + false_alarms) > 0 else 0
    bias = (hits + false_alarms) / (hits + misses) if (hits + misses) > 0 else 0
    
    return recall, precision, bias

# --- Compute metrics table ---
def compute_metrics_table(df, stat_cols, wrf_cols, obs_col, threshold):
    results = []
    for lead, stat_col, wrf_col in zip(['Day 1', 'Day 2', 'Day 3'], stat_cols, wrf_cols):
        r1, p1, b1 = calculate_metrics(df[stat_col], df[obs_col], threshold)
        r2, p2, b2 = calculate_metrics(df[wrf_col], df[obs_col], threshold)
        
        results.append(['Statistical', lead, r1, p1, b1])
        results.append(['WRF', lead, r2, p2, b2])
    
    metrics_df = pd.DataFrame(results, columns=['Model', 'Lead', 'Recall', 'Precision', 'Bias'])
    return metrics_df

# --- Visualization (All metrics in ONE plot) ---
def plot_metrics_combined(metrics_df, save_folder):
    labels = ['Day 1', 'Day 2', 'Day 3']
    metrics = ['Recall', 'Precision', 'Bias']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Colors for bars

    x = np.arange(len(labels))  # positions for Day 1, Day 2, Day 3
    width = 0.25  # width of each bar
    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract Statistical & WRF values
    stat_vals = metrics_df[metrics_df['Model']=='Statistical'][metrics].values
    wrf_vals = metrics_df[metrics_df['Model']=='WRF'][metrics].values

    # Plot grouped bars for Statistical
    for i, metric in enumerate(metrics):
        positions = x + (i - 1) * width
        ax.bar(positions, stat_vals[:, i], width, color=colors[i], label=f'{metric} (Statistical Model)')

        # Add WRF markers for the same positions
        for j, xpos in enumerate(positions):
            wrf_value = wrf_vals[j, i]
            ax.plot(xpos, wrf_value, marker='*', color='red', markersize=7, linestyle='None',
                    label='WRF metrics' if (i == 0 and j == 0) else "")

        # Annotate Statistical bars
        for j, xpos in enumerate(positions):
            stat_value = stat_vals[j, i]
            ax.annotate(f'{stat_value:.2f}', xy=(xpos, stat_value), xytext=(0, 3),
                        textcoords="offset points", ha='center', fontsize=10)

    # Formatting
    ax.set_ylabel('Performance Score')
    ax.set_title(f"Event-based Metrics_Predicted_vs_Zomato AWS_Ahmedabad Rain (Threshold ≥ {threshold} mm)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(stat_vals.max(), wrf_vals.max()) + 0.3)
    ax.legend()
    #ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = os.path.join(save_folder, f"Performance_Metrics_{threshold}mm.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved combined metrics plot: {save_path}")

# -------------------------------
# Example usage
# -------------------------------
threshold = 25
obs_col = 'AWS'
stat_cols = ['Predicted_Lead Day_1', 'Predicted_Lead Day_2', 'Predicted_Lead Day_3']
wrf_cols = ['WRF_Lead Day_1', 'WRF_Lead Day_2', 'WRF_Lead Day_3']

metrics_df = compute_metrics_table(combined_df, stat_cols, wrf_cols, obs_col, threshold)
print(metrics_df)

plot_metrics_combined(metrics_df, save_folder)

# -------------------------------
# Run plots
# -------------------------------
prediction_columns = ['Predicted_Lead Day_1', 'Predicted_Lead Day_2', 'Predicted_Lead Day_3']
wrf_columns = ['WRF_Lead Day_1', 'WRF_Lead Day_2', 'WRF_Lead Day_3']

plot_combined_timeseries(combined_df, prediction_columns, wrf_columns, save_folder=save_folder)


# In[ ]:




