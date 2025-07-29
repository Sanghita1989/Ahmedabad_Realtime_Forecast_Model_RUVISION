#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import glob
import re
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from datetime import datetime

# === CONFIG ===
LEAD_DAYS = [1, 2, 3]
BASE_DIR = r'C:\Users\Angshudeep Majumdar\Downloads\Githubs_Code_RUVISION_Final'
PICKLE_DIR = os.path.join(BASE_DIR, 'Pickle')
PLOT_DIR = os.path.join(BASE_DIR, 'Plot')
GFS_FILES = {
    1: ('PREC_Ahmedabad_LD_1_daily basis.xlsx', 'U1000_Ahmedabad_LD_1_daily basis.xlsx', 'V1000_Ahmedabad_LD_1_daily basis.xlsx'),
    2: ('PREC_Ahmedabad_LD_2_daily basis.xlsx', 'U1000_Ahmedabad_LD_2_daily basis.xlsx', 'V1000_Ahmedabad_LD_2_daily basis.xlsx'),
    3: ('PREC_Ahmedabad_LD_3_daily basis.xlsx', 'U1000_Ahmedabad_LD_3_daily basis.xlsx', 'V1000_Ahmedabad_LD_3_daily basis.xlsx'),
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
THRESHOLDS = {'80p': 0.2, '85p': 0.15, '90p': 0.1, '95p': 0.05, '99p': 0.01}
TODAY_STR = datetime.now().strftime('%Y-%m-%d')

# === FUNCTIONS ===

def preprocess_and_save_pca(lday):
    prec_file, u_file, v_file = GFS_FILES[lday]
    prec = pd.read_excel(os.path.join(BASE_DIR, prec_file)).set_index('DateTime')
    u = pd.read_excel(os.path.join(BASE_DIR, u_file)).set_index('DateTime')
    v = pd.read_excel(os.path.join(BASE_DIR, v_file)).set_index('DateTime')

    df = pd.concat([u, v, prec], axis=1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled, columns=df.columns)

    pca = PCA(0.99)
    transformed = pca.fit_transform(df_scaled)
    transformed_df = pd.DataFrame(transformed, index=df.index)

    filename = f'transformed_dataX_LD_{lday}_U1000_V1000_PREC.pkl'
    transformed_df.to_pickle(os.path.join(PICKLE_DIR, filename))


def load_data(pkl_path, obs_path):
    X_full = pd.read_pickle(pkl_path)
    Y = pd.read_excel(obs_path).set_index('DateTime')
    Y.index = pd.to_datetime(Y.index)
    X = X_full.iloc[:len(Y)]
    X.index = Y.index
    return X, Y['Prec_23.0_72.5'], X_full


def split_data(X, Y, split_index=3269):
    return X.iloc[:split_index], X.iloc[split_index+3:], Y.iloc[:split_index], Y.iloc[split_index+3:]


def binary_glm(X_train, Y_train):
    y_binary = (Y_train > 0).astype(int)
    glm = sm.GLM(y_binary, sm.add_constant(X_train), family=sm.families.Binomial()).fit()
    return glm.predict(sm.add_constant(X_train))


def quantile_regression(X, Y, q):
    return sm.QuantReg(Y, sm.add_constant(X)).fit(q=q).params


def reconstruct_rain(X, coefs):
    return X @ coefs[1:] + coefs[0]


def cqm_pipeline(X_train, Y_train, X_test, thresholds):
    yfit_train = binary_glm(X_train, Y_train)
    coefs_dict = {}
    rain_dict = {}

    for label, t in thresholds.items():
        q_val = float(label.replace('p', '')) / 100
        mask1 = yfit_train > t
        X1, Y1 = X_train[mask1], Y_train[mask1]
        if X1.empty or Y1.empty:
            continue
        coef1 = quantile_regression(X1, Y1, q_val)
        rain1 = reconstruct_rain(X1, coef1)
        mask2 = rain1 > 0
        X2, Y2 = X1[mask2], Y1[mask2]
        if X2.empty or Y2.empty:
            continue
        coef2 = quantile_regression(X2, Y2, q_val)
        rain_test = reconstruct_rain(X_test, coef2)
        coefs_dict[label] = coef2
        rain_dict[label] = np.maximum(rain_test, 0)

    return coefs_dict, rain_dict


def forecast_future_rain(lday):
    pattern = rf'transformed_dataX_LD_{lday}_(.+)\.pkl'
    obs_path = OBS_FILES[lday]
    jjas_start, jjas_end = pd.Timestamp(JJAS_RANGES[lday][0]), pd.Timestamp(JJAS_RANGES[lday][1])

    for pkl in glob.glob(os.path.join(PICKLE_DIR, f'transformed_dataX_LD_{lday}_*.pkl')):
        var_match = re.search(pattern, os.path.basename(pkl))
        if not var_match:
            continue

        var_comb = var_match.group(1)
        out_dir = os.path.join(PLOT_DIR, var_comb)
        os.makedirs(out_dir, exist_ok=True)

        X, Y, X_full = load_data(pkl, obs_path)
        X_train, X_test, Y_train, Y_test = split_data(X, Y)
        coefs2, _ = cqm_pipeline(X_train, Y_train, X_test, THRESHOLDS)

        # === FUTURE ===
        X_future = X_full.iloc[len(Y):]
        if X_future.empty:
            continue

        future_dates = pd.date_range(start=Y.index[-1] + pd.Timedelta(days=1), periods=len(X_future), freq='D')
        X_future.index = future_dates

        rain_future = {
            label: np.maximum(reconstruct_rain(X_future, coefs), 0)
            for label, coefs in coefs2.items()
        }

        # Extract JJAS
        mask_jjas = (X_future.index >= jjas_start) & (X_future.index <= jjas_end)
        rain_jjas = {
            label: pd.Series(data, index=X_future.index)[mask_jjas]
            for label, data in rain_future.items()
        }

        df_jjas = pd.DataFrame(rain_jjas)
        df_jjas.index = df_jjas.index.date

        excel_path = os.path.join(out_dir, f'CQM_QuantileForecast_{var_comb}_2025JJAS_LD_{lday}.xlsx')
        df_jjas.to_excel(excel_path, index_label='Date')
        print(f"Saved forecast to: {excel_path}")

        # Save last day for bar plot
        if '80p' in df_jjas.columns:
            yield df_jjas['80p'].iloc[-1:], lday


def plot_last_3_days_bar(pred_series_list):
    combined = pd.concat([s for s, _ in pred_series_list])
    combined.index = [f"{i.strftime('%d-%b')}" for (s, lday), i in zip(pred_series_list, combined.index)]

    fig, ax = plt.subplots(figsize=(15, 8))
    bars = ax.bar(combined.index, combined.values, color='blue')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.2, f'{height:.1f}', ha='center', va='bottom')

    ax.set_title(f"Daily based 3 days Rainfall Forecast using Statistical Downscaling model_{TODAY_STR} Initialization_Ahmedabad_2025", fontsize=14)
    ax.set_ylabel("Rainfall (mm)")
    ax.set_xlabel("Date")
    plt.xticks(rotation=45)

    save_path = os.path.join(PLOT_DIR, f'forecast_3_Lead_Days_{TODAY_STR}_Initialisation.png')
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Bar plot saved to: {save_path}")


# === MAIN EXECUTION ===
if __name__ == '__main__':
    for ld in LEAD_DAYS:
        preprocess_and_save_pca(ld)

    bar_series = list(forecast_future_rain(ld) for ld in LEAD_DAYS)
    flat_series = [item for sublist in bar_series for item in sublist]
    plot_last_3_days_bar(flat_series)


# In[ ]:




