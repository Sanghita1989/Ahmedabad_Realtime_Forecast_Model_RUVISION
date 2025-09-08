import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_last_3_days_bar(config, pred_series_list):
    combined = pd.concat([s for s,_ in pred_series_list])
    combined.index = [f"{i.strftime('%d-%b')}" for (s, lday), i in zip(pred_series_list, combined.index)]

    def get_color(v):
        if v==0: return "#B0B0B0"
        elif v<=15.5: return "#ADD8E6"
        elif v<=35.5: return "#00FF00"
        elif v<=64.5: return "#FFFF00"
        elif v<=115.5: return "#FFA500"
        elif v<=204.4: return "#FF0000"
        else: return "#8B0000"

    colors = [get_color(v) for v in combined.values]
    fig, ax = plt.subplots(figsize=(15,10), dpi=600)
    bars = ax.bar(combined.index, combined.values, color=colors, edgecolor="black", linewidth=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h*0.5, f"{h:.1f}", ha="center", va="center", fontsize=12)
    ax.set_title(f"3-Day Rainfall Forecast ({config['CITY']}) - {config['TODAY_STR']}", fontsize=20)
    ax.set_ylabel("Predicted Rainfall (mm)")
    ax.set_xlabel("Forecast Date")
    plt.xticks(rotation=30)
    save_path = os.path.join(config["PLOT_DIR"], "Forecast_Today.png")
    yesterday_path = os.path.join(config["PLOT_DIR"], "Forecast_Yesterday.png")
    if os.path.exists(save_path):
        if os.path.exists(yesterday_path): os.remove(yesterday_path)
        os.rename(save_path, yesterday_path)
    fig.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"[INFO] Bar plot saved: {save_path}")
