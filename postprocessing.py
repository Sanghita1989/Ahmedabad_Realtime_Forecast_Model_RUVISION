#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import matplotlib.pyplot as plt
import pandas as pd
from config import PLOT_DIR, TODAY_STR


def plot_last_3_days_bar(pred_series_list):
    """
    Plot rainfall forecast for the last 3 days using IMD classification.

    Parameters
    ----------
    pred_series_list : list
        List of tuples [(pd.Series, lead_day), ...]
        where each series has forecast rainfall values.
    """

    # Combine into single series
    combined = pd.concat([s for s, _ in pred_series_list])
    combined.index = [
        f"{i.strftime('%d-%b')}"
        for (s, lday), i in zip(pred_series_list, combined.index)
    ]

    # --- IMD Rainfall classification colors ---
    def get_color(value):
        if value == 0:
            return "#B0B0B0"  # Grey
        elif value <= 15.5:
            return "#ADD8E6"  # Light Blue
        elif value <= 35.5:
            return "#00FF00"  # Green
        elif value <= 64.5:
            return "#FFFF00"  # Yellow
        elif value <= 115.5:
            return "#FFA500"  # Orange
        elif value <= 204.4:
            return "#FF0000"  # Red
        else:
            return "#8B0000"  # Maroon

    colors = [get_color(v) for v in combined.values]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(15, 10), dpi=600)
    bars = ax.bar(
        combined.index, combined.values,
        color=colors, edgecolor='black', linewidth=0.5
    )

    # Add values inside bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height * 0.5,
            f'{height:.1f}',
            ha='center', va='center',
            fontsize=12, color='black'
        )

    # Title and labels
    ax.set_title(
        f"3-Day Rainfall Forecast (Statistical Downscaling)\nAhmedabad - {TODAY_STR}",
        fontsize=20
    )
    ax.set_ylabel("Predicted Rainfall (mm)", fontsize=14)
    ax.set_xlabel("Forecast Date", fontsize=14)

    # Tick formatting
    plt.xticks(rotation=30, fontsize=13)
    plt.yticks(fontsize=12)

    # --- Legend ---
    legend_labels = [
        ("No Rain (0 mm)", "#B0B0B0"),
        ("Light Rain (0.1–15.5 mm)", "#ADD8E6"),
        ("Moderate (15.6–35.5 mm)", "#00FF00"),
        ("Rather Heavy (35.6–64.4 mm)", "#FFFF00"),
        ("Heavy (64.5–115.5 mm)", "#FFA500"),
        ("Very Heavy (115.5–204.4 mm)", "#FF0000"),
        ("Extremely Heavy (≥204.5 mm)", "#8B0000"),
    ]
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, linewidth=1, edgecolor="black")
        for _, color in legend_labels
    ]
    labels = [label for label, _ in legend_labels]

    ax.legend(
        handles, labels,
        title="IMD Rainfall Category",
        fontsize=14,
        title_fontsize=15,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=True,
        facecolor='white',
        framealpha=0.9,
        borderpad=1.2,
        handlelength=2.5
    )

    # --- Save high-res image ---
    save_path = os.path.join(PLOT_DIR, 'Forecast_Today.png')
    yesterday_path = os.path.join(PLOT_DIR, 'Forecast_Yesterday.png')

    if os.path.exists(save_path):
        if os.path.exists(yesterday_path):
            os.remove(yesterday_path)
        os.rename(save_path, yesterday_path)
        print(f"Renamed existing file to: {yesterday_path}")

    fig.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"Bar plot saved to: {save_path}")

