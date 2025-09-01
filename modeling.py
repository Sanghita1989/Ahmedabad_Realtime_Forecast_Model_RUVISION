#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import glob
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from config import PICKLE_DIR, PLOT_DIR, GFS_FILES, OBS_FILES, JJAS_RANGES, THRESHOLDS
from datetime import timedelta

# -------------------- PCA & Save --------------------
def preprocess_and_save_pca(lday):
    """
    Load the three Excel files (PREC, U1000, V1000) for given lead day,
    concatenate, scale, run PCA(0.99), and save transformed pickle.
    """
    prec_file, u_file, v_file = GFS_FILES[lday]
    prec_path = os.path.join(os.path.dirname(PICKLE_DIR) if os.path.isabs(PICKLE_DIR) else ".", prec_file)
    u_path = os.path.join(os.path.dirname(PICKLE_DIR) if os.path.isabs(PICKLE_DIR) else ".", u_file)
    v_path = os.path.join(os.path.dirname(PICKLE_DIR) if os.path.isabs(PICKLE_DIR) else ".", v_file)

    # try local paths relative to BASE_DIR — fallback to provided names
    for p in [prec_path, u_path, v_path]:
        if not os.path.exists(p):
            p = p  # let pd.read_excel fail with informative message

    try:
        prec = pd.read_excel(os.path.join(os.path.dirname(PICKLE_DIR), prec_file)).set_index('DateTime')
        u = pd.read_excel(os.path.join(os.path.dirname(PICKLE_DIR), u_file)).set_index('DateTime')
        v = pd.read_excel(os.path.join(os.path.dirname(PICKLE_DIR), v_file)).set_index('DateTime')
    except Exception:
        # try direct filename (if these were absolute already)
        prec = pd.read_excel(prec_file).set_index('DateTime')
        u = pd.read_excel(u_file).set_index('DateTime')
        v = pd.read_excel(v_file).set_index('DateTime')

    df = pd.concat([u, v, prec], axis=1).dropna(how='all')

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.fillna(0))
    df_scaled = pd.DataFrame(scaled, columns=df.columns, index=df.index)

    pca = PCA(0.99, svd_solver='full')
    transformed = pca.fit_transform(df_scaled)
    transformed_df = pd.DataFrame(transformed, index=df.index)

    os.makedirs(PICKLE_DIR, exist_ok=True)
    filename = f'transformed_dataX_LD_{lday}_U1000_V1000_PREC.pkl'
    transformed_df.to_pickle(os.path.join(PICKLE_DIR, filename))
    print(f"[INFO] Saved transformed pickle: {filename}")


# -------------------- Load --------------------
def load_data(pkl_path, obs_path):
    X_full = pd.read_pickle(pkl_path)
    Y = pd.read_excel(obs_path).set_index('DateTime')
    Y.index = pd.to_datetime(Y.index)
    X = X_full.iloc[:len(Y)]
    X.index = Y.index
    return X, Y.iloc[:, 0] if Y.shape[1] == 1 else Y.iloc[:, 0], X_full


# -------------------- Split --------------------
def split_data(X, Y, split_index=3269):
    # Keep same splitting logic as original; allow fallback if smaller
    if split_index >= len(X):
        split_index = int(len(X) * 0.8)
    return X.iloc[:split_index], X.iloc[split_index + 3:], Y.iloc[:split_index], Y.iloc[split_index + 3:]


# -------------------- GLM Binary --------------------
def binary_glm(X_train, Y_train):
    y_binary = (Y_train > 0).astype(int)
    model = sm.GLM(y_binary, sm.add_constant(X_train), family=sm.families.Binomial())
    glm = model.fit()
    return glm.predict(sm.add_constant(X_train))


# -------------------- Quantile regression helpers --------------------
def quantile_regression(X, Y, q):
    model = sm.QuantReg(Y, sm.add_constant(X))
    res = model.fit(q=q)
    return res.params


def reconstruct_rain(X, coefs):
    """
    coefs is a pandas Series (with const first). Returns predicted values (1D array/Series).
    """
    const = coefs.iloc[0]
    betas = coefs.iloc[1:].values
    preds = X.values.dot(betas) + const
    return preds


# -------------------- CQM pipeline --------------------
def cqm_pipeline(X_train, Y_train, X_test, thresholds):
    """
    Multi-threshold censored quantile pipeline.
    Returns:
      - coefs_dict: quantile coefficients per threshold label
      - rain_dict: predicted rainfall arrays per threshold label (on X_test)
    """
    yfit_train = binary_glm(X_train, Y_train)
    coefs_dict = {}
    rain_dict = {}

    for label, t in thresholds.items():
        q_val = float(label.replace('p', '')) / 100.0

        mask1 = yfit_train > t
        X1, Y1 = X_train[mask1], Y_train[mask1]
        if X1.empty or Y1.empty:
            print(f"[WARN] No training data after mask1 for threshold {label}")
            continue

        coef1 = quantile_regression(X1, Y1, q_val)
        rain1 = reconstruct_rain(X1, coef1)

        mask2 = rain1 > 0
        # Align mask2 indexes with X1
        if mask2.sum() == 0:
            print(f"[WARN] No positive quantile-predicted rain in stage1 for {label}")
            continue

        X2 = X1.iloc[mask2]
        Y2 = Y1.iloc[mask2]

        if X2.empty or Y2.empty:
            print(f"[WARN] No training data after mask2 for threshold {label}")
            continue

        coef2 = quantile_regression(X2, Y2, q_val)
        rain_test = reconstruct_rain(X_test, coef2)

        coefs_dict[label] = coef2
        rain_dict[label] = np.maximum(rain_test, 0.0)

    return coefs_dict, rain_dict


# -------------------- Forecast / JJAS saving --------------------
def forecast_future_rain(lday):
    """
    Iterate over transformed pickle files for the lead day, load observed dataset,
    run CQM pipeline and save JJAS excel file per variable-combination.
    Yields the last 80p value (if present) and lead day for plotting upstream.
    """
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

        try:
            X, Y, X_full = load_data(pkl, obs_path)
        except Exception as e:
            print(f"[ERROR] Loading data failed for {pkl}: {e}")
            continue

        X_train, X_test, Y_train, Y_test = split_data(X, Y)
        if X_train.empty or Y_train.empty:
            print(f"[WARN] Empty train for {pkl}")
            continue

        coefs2, _ = cqm_pipeline(X_train, Y_train, X_test, THRESHOLDS)
        X_future = X_full.iloc[len(Y):]
        if X_future.empty:
            print(f"[INFO] No future X for {pkl} -> skipping future forecast")
            continue

        # create future index (daily)
        future_dates = pd.date_range(start=Y.index[-1] + pd.Timedelta(days=1), periods=len(X_future), freq='D')
        X_future.index = future_dates

        rain_future = {label: np.maximum(reconstruct_rain(X_future, coefs), 0.0) for label, coefs in coefs2.items()}

        mask_jjas = (pd.Index(future_dates) >= jjas_start) & (pd.Index(future_dates) <= jjas_end)
        rain_jjas = {label: pd.Series(data, index=future_dates)[mask_jjas] for label, data in rain_future.items()}

        df_jjas = pd.DataFrame(rain_jjas)
        df_jjas.index = df_jjas.index.date

        excel_path = os.path.join(out_dir, f'CQM_QuantileForecast_{var_comb}_2025JJAS_LD_{lday}.xlsx')
        df_jjas.to_excel(excel_path, index_label='Date')
        print(f"[INFO] Saved forecast to: {excel_path}")

        if '80p' in df_jjas.columns and not df_jjas.empty:
            # yield the last 80p value and lead day (to support bar plotting)
            yield df_jjas['80p'].iloc[-1:], lday

