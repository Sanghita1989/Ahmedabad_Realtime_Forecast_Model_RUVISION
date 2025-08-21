#!/usr/bin/env python
# coding: utf-8

# In[53]:


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
folder = r'C:\Users\Angshudeep Majumdar\Downloads\Githubs_Code_RUVISION_Final\Plot\U1000_V1000_PREC'
save_folder = r'C:\Users\Angshudeep Majumdar\Downloads\Githubs_Code_RUVISION_Final\Plot'
imd_file = r"C:\Users\Angshudeep Majumdar\Downloads\Githubs_Code_RUVISION_Final\IMD_API\Rainfall_Data.xlsx"

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

col_1 = load_and_clean_excel(files[0], 0)
col_2 = load_and_clean_excel(files[1], 0)
col_3 = load_and_clean_excel(files[2], 0)
col_4 = load_and_clean_excel(files[3], 15)

combined_df = pd.concat([col_1, col_2, col_3, col_4], axis=1)
combined_df.columns = ['Predicted_Lead Day_1', 'Predicted_Lead Day_2', 'Predicted_Lead Day_3', 'AWS']
combined_df.index.name = 'Datetime'

# Only drop rows where AWS is missing, keep prediction start dates intact
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
# Plotting function
# -------------------------------
def plot_timeseries_separately(df, pred_cols, col_aws='AWS', col_imd='IMD', save_prefix='Predicted', save_folder='.'):
    global_min, global_max = df['Datetime'].min(), df['Datetime'].max()

    for pred_col in pred_cols:
        # Restrict to rows where prediction is available
        df_sub = df.loc[df[pred_col].notna(), ['Datetime', pred_col, col_aws, col_imd]].copy()

        
        start_date = df_sub['Datetime'].min()
        df_aws = df.loc[df['Datetime'] >= start_date, ['Datetime', col_aws, col_imd]]

        fig, axs = plt.subplots(3, 1, figsize=(16, 15), sharex=True)

        # --- Line plot
        axs[0].plot(df_sub['Datetime'], df_sub[pred_col], color='blue',
                    label=f'{pred_col}(00:00–23:59 hrs)')
        axs[0].plot(df_aws['Datetime'], df_aws[col_aws], color='black', linestyle='--',
                    label=f'{col_aws}(Observed_Zomato) (00:00–23:59 hrs)')
        axs[0].scatter(df_aws['Datetime'], df_aws[col_imd], color='deeppink', marker='*', s=120,
                       label='IMD_AWS (08:30–08:30 hrs)')
        axs[0].set_ylabel("Rainfall (mm)")
        axs[0].set_title(f"{pred_col} vs Zomato_{col_aws} vs IMD_Ahmedabad City Rain Model")
        axs[0].legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), fontsize=12, frameon=True)
        axs[0].grid(True)

        # --- Error (Prediction - AWS)
        error = df_sub[pred_col] - df_sub[col_aws]
        axs[1].bar(df_sub['Datetime'], error, color='red')
        axs[1].axhline(0, color='black', linewidth=1.2)
        axs[1].set_ylabel("Error (mm)")
        axs[1].set_title(f"Error ({pred_col} - Zomato_{col_aws})")
        axs[1].grid(True)

        # --- Percentage Error
        percentage_error = np.where(df_sub[col_aws] != 0, (error / df_sub[col_aws]), 0)
        axs[2].bar(df_sub['Datetime'], percentage_error, color='orange')
        axs[2].axhline(0, color='black', linewidth=1.2)
        axs[2].set_ylabel("Error Fraction")
        axs[2].set_title(f"Percentage Error ({pred_col} - {col_aws}) / {col_aws}")
        axs[2].grid(True)

        # --- X-axis formatting with global min/max
        for ax in axs:
            ax.set_xlim(global_min, global_max)

            # automatic locator
            locator = mdates.DayLocator(interval=7)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

            # enable bottom labels on ALL subplots
            ax.tick_params(axis='x', rotation=90, labelbottom=True)

            # force min & max to appear as ticks
            xticks = list(ax.get_xticks())
            xticks = list(set(xticks + [mdates.date2num(global_min), mdates.date2num(global_max)]))
            xticks = sorted(xticks)
            ax.set_xticks(xticks)

        # Mark start date of each lead explicitly
        axs[0].axvline(start_date, color='green', linestyle=':', linewidth=1.5,
                       label='Prediction Start')
        axs[0].legend()

        plt.tight_layout()
        save_name = os.path.join(save_folder, f"{save_prefix}_{pred_col}_comparison.png")
        plt.savefig(save_name, dpi=300)
        plt.close(fig)
        print(f"Saved: {save_name}")

# -------------------------------
# Metrics
# -------------------------------
def calculate_metrics(y_true, y_pred, threshold=25):
    obs_event = y_true >= threshold
    pred_event = y_pred >= threshold
    TP = np.sum((pred_event == 1) & (obs_event == 1))
    FP = np.sum((pred_event == 1) & (obs_event == 0))
    FN = np.sum((pred_event == 0) & (obs_event == 1))
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    bias = (TP + FP) / (TP + FN) if (TP + FN) > 0 else 0
    return recall, precision, bias

def plot_metrics(df, pred_cols, col_aws='AWS', threshold=25, save_folder='.'):
    metrics_dict = {"Lead Day": [], "Recall": [], "Precision": [], "Bias": []}
    for pred_col in pred_cols:
        recall, precision, bias = calculate_metrics(df[col_aws], df[pred_col], threshold=threshold)
        metrics_dict["Lead Day"].append(pred_col)
        metrics_dict["Recall"].append(recall)
        metrics_dict["Precision"].append(precision)
        metrics_dict["Bias"].append(bias)
        print(f"{pred_col} -> Recall: {recall:.2f}, Precision: {precision:.2f}, Bias: {bias:.2f}")

    metrics_df = pd.DataFrame(metrics_dict)
    ax = metrics_df.plot(kind="bar", x="Lead Day", figsize=(10, 5))
    plt.title(f"Event-based Metrics_Predicted_vs_Zomato AWS_Ahmedabad Rain (Threshold ≥ {threshold} mm)", fontsize=14)
    plt.ylabel("Performance Score", fontsize=12)
    ax.set_xlabel("", fontsize=12)
    plt.xticks(rotation=0, fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylim(0, max(metrics_df.max(numeric_only=True))+0.4)
    plt.grid(True, axis='y')
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", padding=3, fontsize=10)
    save_name = os.path.join(save_folder, f"Performance_Metrics_{threshold}mm.png")
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.show()
    print(f"Saved: {save_name}")

# -------------------------------
# Run all plots
# -------------------------------
prediction_columns = ['Predicted_Lead Day_1', 'Predicted_Lead Day_2', 'Predicted_Lead Day_3']
plot_timeseries_separately(combined_df, prediction_columns, save_folder=save_folder)
plot_metrics(combined_df, prediction_columns, save_folder=save_folder)


# In[ ]:




