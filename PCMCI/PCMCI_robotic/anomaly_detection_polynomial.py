"""
==============================================================
 Causal Discovery & Anomaly Detection (Polynomial-Tigramite)
==============================================================

Workflow:
1. Data Loading
2. Causal Model Learning (offline PCMCI)
3. Offline Polynomial Coefficient Estimation
4. Online Monitoring (moving-window polynomial coefficient updates)
5. Anomaly Detection
6. Metrics Computation

Author: (Sagar)
==============================================================
"""

# ================== IMPORTS ==================
import os
import numpy as np
import pandas as pd
import warnings
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from scipy.stats import ConstantInputWarning

import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

warnings.filterwarnings('ignore', category=ConstantInputWarning)

# ================== GLOBAL CONFIG ==================
ALPHA = 0.05
TRAINING_FRAC = 0.7
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREFIX = os.path.join(BASE_DIR, "pepper_csv")
TASK = "pepper"
MAX_FREQ_COMPONENTS = 5

# Polynomial regression config
POLY_DEGREE = 3
RIDGE_ALPHA = 0.01
RLS_LAMBDA = 0.999

# Anomaly detection tuning config
DETECTION_THRESHOLD_MULTIPLIER = 1.2
CAUSAL_STRENGTH_MULTIPLIER = 0.5
CONSECUTIVE_K = 1


def _rls_update(x_vec: np.ndarray, y_t: float, w_prev: np.ndarray, P_prev: np.ndarray, lam: float):
    """
    Standard Recursive Least Squares (RLS) Update Step.
    """
    x = x_vec.reshape(-1)
    e_t = float(y_t) - float(np.dot(x, w_prev))
    Px = P_prev @ x
    denom = float(lam + np.dot(x, Px))
    g_t = Px / denom
    w_new = w_prev + g_t * e_t
    P_new = (P_prev - np.outer(g_t, Px)) / float(lam)
    return w_new, P_new



# ================================================================
#                         DATA LOADING
# ================================================================
def read_data(path: str, task: str) -> pd.DataFrame:
    """
    Load CSV data, handle timestamp indexing.

    Visual:
        CSV -> DataFrame indexed by Timestamp
    """
    df = pd.read_csv(path, delimiter="," if task == "pepper" else ";")
    if task == "pepper":
        df["Timestamp"] = df["timestamp"]
        df.set_index("Timestamp", inplace=True)
        df.drop(columns=["timestamp"], inplace=True)
    else:
        df["Timestamp"] = pd.to_datetime(df[" Timestamp"].str.strip(),
                                         format="%d/%m/%Y %I:%M:%S %p")
        df.set_index("Timestamp", inplace=True)
        df.drop(columns=[" Timestamp"], inplace=True)
    return df


# ================================================================
#                     LEARN CAUSAL MODEL
# ================================================================
def learn_causal_model(normal_csv_path: str, save_path: str):
    """
    Learn the causal graph using PCMCI.
    Tau_max is automatically computed from the dominant frequency.
    """
    print("Learning causal model...")

    df = read_data(normal_csv_path, TASK)
    normal_data = pp.DataFrame(np.nan_to_num(df.values))
    normal_data.values[0] = normal_data.values[0][:int(TRAINING_FRAC * np.shape(normal_data.values[0])[0]), :]

    frequencies = []
    for index in range(np.shape(normal_data.values[0])[1]):
        if any([el for el in normal_data.values[0][:, index] if int(el) != el]):
            w = np.fft.fft(normal_data.values[0][:, index])
            freqs = np.fft.fftfreq(len(w))
            mods = abs(w)
            max_indices = np.argsort(mods)[::-1][:MAX_FREQ_COMPONENTS]
            main_freq = []
            for i in max_indices:
                freq = freqs[i]
                main_freq.append(freq)
            frequencies += main_freq

    sorted_freq = np.sort([el for el in frequencies if el > 0])[::-1]
    for freq in sorted_freq:
        if len([fr for fr in sorted_freq if fr < freq]) / len(sorted_freq) < 0.95:
            max_freq = freq
            sorted_freq = [s for s in sorted_freq if s <= max_freq]
            break

    subsample = max(1, int(np.floor(1 / 10 / max_freq)))
    normal_data.values[0] = normal_data.values[0][::max(1, subsample), :]
    nonconst = [idx for idx in range(np.shape(normal_data.values[0])[1]) if np.std(normal_data.values[0][:, idx]) > 0.01 * np.mean(normal_data.values[0][:, idx])]
    nonconst_data = normal_data.values[0][:, nonconst]
    for j in range(np.shape(nonconst_data)[1]):
        denom = (np.max(nonconst_data[:, j]) - np.min(nonconst_data[:, j])) + np.min(nonconst_data[:, j])
        if denom > 1e-8:
            nonconst_data[:, j] /= denom
    print(np.shape(nonconst_data))

    tau_max = int(np.floor(max_freq / np.mean(np.unique(sorted_freq))))
    print(tau_max)

    dataframe = pp.DataFrame(nonconst_data)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=ALPHA)

    np.savez(save_path,
             val_matrix=results["val_matrix"],
             p_matrix=results["p_matrix"],
             var=df.columns,
             subsample=subsample,
             nonconst=nonconst)
    print(f"Saved causal model to {save_path}")
    return results, subsample, nonconst, tau_max


# ================================================================
#           OFFLINE POLYNOMIAL COEFFICIENT FITTING
# ================================================================
def fit_normal_poly_coeffs(normal_data: np.ndarray, causal_matrix: np.ndarray,
                           degree: int = POLY_DEGREE, alpha: float = RIDGE_ALPHA):
    """
    Compute offline (baseline) polynomial coefficients for each variable.
    """
    indices = np.array(np.where(causal_matrix != 0))
    fine_coeffs = {}
    poly_maps = {}

    for var in np.unique(indices[1, :]):
        var_indices = [indices[:, k] for k in range(indices.shape[1]) if indices[1, k] == var]
        var_indices.sort(key=lambda x: x[2])
        max_delay = var_indices[-1][2]
        var_indices = var_indices[:1]

        stack = [normal_data[max_delay - el[2]: len(normal_data) - el[2], el[0]]
                 for el in var_indices]
        X = np.column_stack(stack)
        y = normal_data[max_delay:, var]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X_scaled)
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_poly, y)
        fine_coeffs[var] = np.concatenate([[float(model.intercept_)], np.ravel(model.coef_)])
        poly_maps[var] = (scaler, poly)

    return fine_coeffs, indices, poly_maps


# ================================================================
#      ONLINE POLYNOMIAL COEFFICIENTS & ERROR COMPUTATION
# ================================================================
def compute_online_poly_errors(data: np.ndarray, fine_coeffs: dict,
                               causal_matrix: np.ndarray, indices: np.ndarray,
                               poly_maps: dict, alpha: float = RIDGE_ALPHA,
                               rls_lambda: float = RLS_LAMBDA):
    """
    Recompute polynomial coefficients online sample-by-sample via RLS and compute deviations.
    """
    max_time = data.shape[0] - causal_matrix.shape[2]
    unique_vars = np.unique(indices[1, :])
    err = {}
    norm_agg = np.zeros((max_time, len(unique_vars)))

    w_state = {}
    P_state = {}
    var_prep = {}

    for var in unique_vars:
        w0 = fine_coeffs[var].copy()
        P0 = np.eye(len(w0)) * (1.0 / alpha)
        w_state[var] = w0
        P_state[var] = P0

        var_indices = [indices[:, k] for k in range(indices.shape[1]) if indices[1, k] == var]
        var_indices.sort(key=lambda x: x[2])
        var_indices = var_indices[:1]
        max_delay = var_indices[-1][2]

        stack = [data[max_delay - el[2]: max_time + max_delay - el[2], el[0]] for el in var_indices]
        X = np.column_stack(stack)
        y = data[max_delay: max_time + max_delay, var]
        scaler, poly = poly_maps[var]
        X_scaled = scaler.transform(X)
        X_poly = poly.transform(X_scaled)
        var_prep[var] = (X_poly, y)
        err[var] = np.zeros((max_time, len(w0)))

    for t in range(max_time):
        for i, var in enumerate(unique_vars):
            X_poly, y = var_prep[var]
            x_vec = np.concatenate([[1.0], X_poly[t]])
            y_t = float(y[t])

            w_prev = w_state[var]
            P_prev = P_state[var]

            w_new, P_new = _rls_update(x_vec, y_t, w_prev, P_prev, rls_lambda)
            w_state[var] = w_new
            P_state[var] = P_new

            err[var][t, :] = w_new - fine_coeffs[var]
            norm_agg[t, i] = np.linalg.norm(err[var][t, :])

    return err, norm_agg


# ================================================================
#                        ANOMALY DETECTION
# ================================================================
def detect_anomalies(err_normal, err_attack, normal_data_len, normal):
    """
    Flag anomalies if online coefficients deviate significantly from offline baseline.
    """
    indices_error = []
    for var in err_attack.keys():
        for j in range(err_attack[var].shape[1]):
            thresh = DETECTION_THRESHOLD_MULTIPLIER * np.linalg.norm(err_normal[var][:normal_data_len, j])
            if not normal:
                indices_error += list(np.where(abs(err_attack[var][:, j]) > thresh)[0])
            else:
                indices_error += list(np.where(abs(err_attack[var][normal_data_len:, j]) > thresh)[0])
    return len(np.unique(indices_error))


def score_anomaly_timeline(err_normal, err_attack, normal_data_len, normal):
    """
    Convert coefficient-drift errors into per-time predictions and scores.
    """
    if normal:
        start = normal_data_len
        max_time = min(
            (err_attack[var].shape[0] - start for var in err_attack.keys()),
            default=0,
        )
    else:
        start = 0
        max_time = min((err_attack[var].shape[0] for var in err_attack.keys()), default=0)

    scores = np.zeros(max_time, dtype=float)
    for var in err_attack.keys():
        for j in range(err_attack[var].shape[1]):
            thresh = DETECTION_THRESHOLD_MULTIPLIER * np.linalg.norm(
                err_normal[var][:normal_data_len, j]
            )
            thresh = max(float(thresh), np.finfo(float).eps)
            values = np.abs(err_attack[var][start:start + max_time, j]) / thresh
            scores = np.maximum(scores, values)

    predictions = (scores > 1.0).astype(int)
    return predictions, scores


def compute_extended_metrics(normal_predictions, normal_scores, attack_predictions, attack_scores):
    """
    Compute Accuracy, AUC-ROC, and AUC-PR metrics.
    """
    y_true = np.concatenate([
        np.zeros(len(normal_predictions), dtype=int),
        np.ones(sum(len(pred) for pred in attack_predictions), dtype=int),
    ])
    y_pred = np.concatenate([normal_predictions] + attack_predictions)
    y_score = np.concatenate([normal_scores] + attack_scores)

    accuracy = accuracy_score(y_true, y_pred)
    try:
        auc_roc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc_roc = np.nan
    try:
        auc_pr = average_precision_score(y_true, y_score)
    except ValueError:
        auc_pr = np.nan
    return accuracy, auc_roc, auc_pr


# ================================================================
#                         MAIN PIPELINE
# ================================================================
def main():
    print(f"\n========== TASK: {TASK.upper()} ==========")
    print(f"Polynomial degree: {POLY_DEGREE}")
    print(f"Ridge alpha: {RIDGE_ALPHA}")

    causal_path = os.path.join(PREFIX, f"{TASK}_normal.npz")
    print(f"Causal model path: {causal_path}")

    if not os.path.exists(causal_path):
        learn_causal_model(os.path.join(PREFIX, "normal.csv"), causal_path)
    else:
        print("Causal model found, loading...")

    f = np.load(causal_path, allow_pickle=True)
    val_matrix, p_matrix = f["val_matrix"], f["p_matrix"]
    subsample, nonconst = int(f["subsample"]), f["nonconst"]

    normal_matrix = val_matrix * (p_matrix < ALPHA) * (abs(val_matrix) > CAUSAL_STRENGTH_MULTIPLIER * np.mean(abs(val_matrix)))

    normal_df = read_data(os.path.join(PREFIX, "normal.csv"), TASK)
    normal_data = np.nan_to_num(normal_df.values[:int(TRAINING_FRAC * len(normal_df))][::subsample, nonconst])
    normal_data_full = np.nan_to_num(normal_df.values[::subsample, nonconst])

    fine_coeffs, indices, poly_maps = fit_normal_poly_coeffs(normal_data, normal_matrix)

    err_normal, norm_agg_normal = compute_online_poly_errors(normal_data_full, fine_coeffs, normal_matrix, indices, poly_maps)

    _ = norm_agg_normal

    attack_paths = [
        os.path.join(PREFIX, "WheelsControl.csv"),
        os.path.join(PREFIX, "JointControl.csv"),
        os.path.join(PREFIX, "LedsControl.csv")
    ]
    attack_dfs = [read_data(p, TASK) for p in attack_paths]

    tpos, fpos, fneg = [], [], []
    attack_predictions, attack_scores = [], []

    fpos.append(detect_anomalies(err_normal, err_normal, len(normal_data), normal=True))
    normal_predictions, normal_scores = score_anomaly_timeline(
        err_normal, err_normal, len(normal_data), normal=True
    )

    norm_agg_attacks = []
    attack_names = []

    for path, df_attack in zip(attack_paths, attack_dfs):
        attack_name = os.path.basename(path)
        attack_names.append(attack_name)
        print(f"\n--- Analyzing anomaly: {attack_name} ---")

        attack_data = np.nan_to_num(df_attack.values[::subsample, nonconst])
        err_attack, norm_agg_attack = compute_online_poly_errors(attack_data, fine_coeffs, normal_matrix, indices, poly_maps)
        norm_agg_attacks.append(norm_agg_attack)

        tp_count = detect_anomalies(err_normal, err_attack, len(normal_data), normal=False)
        tpos.append(tp_count)
        fneg.append(attack_data.shape[0] - tp_count)
        attack_prediction, attack_score = score_anomaly_timeline(
            err_normal, err_attack, len(normal_data), normal=False
        )
        if len(attack_prediction) < attack_data.shape[0]:
            pad_len = attack_data.shape[0] - len(attack_prediction)
            attack_prediction = np.concatenate([np.zeros(pad_len, dtype=int), attack_prediction])
            attack_score = np.concatenate([np.zeros(pad_len, dtype=float), attack_score])
        attack_predictions.append(attack_prediction)
        attack_scores.append(attack_score)

    import reporting_helper

    attack_scores_dict = dict(zip(attack_names, attack_scores))
    attack_labels_dict = {}
    target_wise_residuals_by_file = dict(zip(attack_names, norm_agg_attacks))

    for name, data_matrix in zip(attack_names, [read_data(p, TASK).values[::subsample, nonconst] for p in attack_paths]):
        attack_labels_dict[name] = np.ones(data_matrix.shape[0])

    output_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    reporting_helper.generate_plots_and_reports(
        output_dir=output_dir,
        model_variant="Polynomial degree 3",
        model_name="PCMCI",
        normal_scores=normal_scores,
        attack_scores_by_file=attack_scores_dict,
        threshold=1.0,
        var_names=f["var"][nonconst].tolist(),
        target_indices=np.unique(indices[1, :]).tolist(),
        directed_edges=[], 
        target_wise_residuals_by_file=target_wise_residuals_by_file,
        attack_labels_by_file=attack_labels_dict,
        dataset_name="Robotics (Pepper)"
    )


# ================================================================
#                          RUN SCRIPT
# ================================================================
if __name__ == "__main__":
    main()
