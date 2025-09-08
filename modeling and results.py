import os, glob, re
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def preprocess_and_save_pca(config, lday):
    files = config["GFS_FILES"][lday]
    dfs = [pd.read_excel(files[var]).set_index("DateTime") for var in config["VARIABLES"]]
    df = pd.concat(dfs, axis=1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled, columns=df.columns)

    pca = PCA(0.99)
    transformed = pca.fit_transform(df_scaled)
    transformed_df = pd.DataFrame(transformed, index=df.index)

    filename = f"transformed_dataX_LD_{lday}_" + "_".join(config["VARIABLES"]) + ".pkl"
    transformed_df.to_pickle(os.path.join(config["PICKLE_DIR"], filename))

def load_data(pkl_path, obs_path):
    X_full = pd.read_pickle(pkl_path)
    Y = pd.read_excel(obs_path).set_index("DateTime")
    Y.index = pd.to_datetime(Y.index)
    X = X_full.iloc[:len(Y)]
    X.index = Y.index
    return X, Y.iloc[:,0], X_full

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
    coefs_dict, rain_dict = {}, {}
    for label, t in thresholds.items():
        q_val = float(label.replace("p",""))/100
        mask1 = yfit_train > t
        X1, Y1 = X_train[mask1], Y_train[mask1]
        if X1.empty: continue
        coef1 = quantile_regression(X1, Y1, q_val)
        rain1 = reconstruct_rain(X1, coef1)
        mask2 = rain1 > 0
        X2, Y2 = X1[mask2], Y1[mask2]
        if X2.empty: continue
        coef2 = quantile_regression(X2, Y2, q_val)
        rain_test = reconstruct_rain(X_test, coef2)
        coefs_dict[label] = coef2
        rain_dict[label] = np.maximum(rain_test, 0)
    return coefs_dict, rain_dict

def forecast_future_rain(config, lday):
    pattern = rf"transformed_dataX_LD_{lday}_(.+)\.pkl"
    obs_path = config["OBS_FILES"][lday]
    jjas_start, jjas_end = pd.Timestamp(config["JJAS_RANGES"][lday][0]), pd.Timestamp(config["JJAS_RANGES"][lday][1])
    for pkl in glob.glob(os.path.join(config["PICKLE_DIR"], f"transformed_dataX_LD_{lday}_*.pkl")):
        var_match = re.search(pattern, os.path.basename(pkl))
        if not var_match: continue
        out_dir = os.path.join(config["PLOT_DIR"], var_match.group(1))
        os.makedirs(out_dir, exist_ok=True)
        X, Y, X_full = load_data(pkl, obs_path)
        X_train, X_test, Y_train, Y_test = split_data(X, Y)
        coefs2, _ = cqm_pipeline(X_train, Y_train, X_test, config["THRESHOLDS"])
        _, rain_dict = cqm_pipeline(X, Y, X_full, config["THRESHOLDS"])
        df_pred = pd.DataFrame({k:v for k,v in rain_dict.items() if k in coefs2}, index=X_full.index)
        df_jjas = df_pred.loc[jjas_start:jjas_end]
        excel_path = os.path.join(out_dir, f"CQM_QuantileForecast_{var_match.group(1)}_2025JJAS_LD_{lday}.xlsx")
        df_jjas.to_excel(excel_path, index_label="Date")
        if "80p" in df_jjas.columns and not df_jjas["80p"].empty:
            yield pd.Series([float(df_jjas["80p"].iloc[-1])], index=[df_jjas.index[-1]]), lday

