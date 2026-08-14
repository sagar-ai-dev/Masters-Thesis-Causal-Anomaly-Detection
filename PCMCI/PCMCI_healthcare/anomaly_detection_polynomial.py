"""
==============================================================
 Causal Discovery & Anomaly Detection (Polynomial-Tigramite - Healthcare / ECG)
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
import reporting_helper

warnings.filterwarnings('ignore', category=ConstantInputWarning)

# ================== GLOBAL CONFIG ==================
ALPHA = 0.05
TRAINING_FRAC = 0.7
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREFIX = os.path.join(BASE_DIR, "Healthcare dataset")
TASK = "healthcare"
MAX_FREQ_COMPONENTS = 5

NORMAL_FILE = "healthcare_normal_train.pkl"
ATTACK_FILE = "healthcare_attacks.pkl"

# Polynomial regression config
POLY_DEGREE = 3
RIDGE_ALPHA = 1.0

# Anomaly detection tuning config
DETECTION_THRESHOLD_MULTIPLIER = 1.0
ANOMALY_SCORE_THRESHOLD = 7.896730851524358e8
CAUSAL_STRENGTH_MULTIPLIER = 0.0
CONSECUTIVE_K = 1
RLS_LAMBDA = 0.995
CALIBRATION_QUANTILE = 0.99

ECG_LEADS = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


# ================================================================
#                         DATA LOADING
# ================================================================
def read_data(path: str, task: str) -> pd.DataFrame:
    """
    Load CSV or PKL data for Healthcare ECG 12-lead time series.
    """
    if path.endswith(".pkl"):
        res = pd.read_pickle(path)
        if isinstance(res, pd.DataFrame):
            available_leads = [c for c in ECG_LEADS if c in res.columns]
            return res[available_leads] if available_leads else res
        return res
    else:
        df = pd.read_csv(path, delimiter="," if task == "pepper" else ";")
        available_leads = [c for c in ECG_LEADS if c in df.columns]
        if available_leads:
            df = df[available_leads]
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
    normal_data.values[0] = normal_data.values[0][
        : int(TRAINING_FRAC * np.shape(normal_data.values[0])[0]), :
    ]

    frequencies = []
    for index in range(np.shape(normal_data.values[0])[1]):
        series = normal_data.values[0][:, index]
        if np.std(series) <= 0:
            continue
        centered = series - np.mean(series)
        if np.allclose(centered, 0):
            continue
        w = np.fft.fft(centered)
        freqs = np.fft.fftfreq(len(w))
        mods = np.abs(w)
        max_indices = np.argsort(mods)[::-1][:MAX_FREQ_COMPONENTS]
        frequencies.extend(freqs[i] for i in max_indices if freqs[i] > 0)

    sorted_freq = np.sort([el for el in frequencies if el > 0])[::-1]
    if len(sorted_freq) == 0:
        sorted_freq = np.array([0.1])

    max_freq = float(sorted_freq[0])
    for freq in sorted_freq:
        if len([fr for fr in sorted_freq if fr < freq]) / len(sorted_freq) < 0.95:
            max_freq = float(freq)
            sorted_freq = [s for s in sorted_freq if s <= max_freq]
            break

    subsample = max(1, int(np.floor(1 / 10 / max_freq)))
    normal_data.values[0] = normal_data.values[0][::subsample, :]
    nonconst = [
        idx
        for idx in range(np.shape(normal_data.values[0])[1])
        if np.std(normal_data.values[0][:, idx]) > 1e-8
    ]
    nonconst_data = normal_data.values[0][:, nonconst]
    for j in range(np.shape(nonconst_data)[1]):
        col = nonconst_data[:, j]
        col_min = np.min(col)
        col_max = np.max(col)
        scale = col_max - col_min
        if scale <= 1e-12:
            nonconst_data[:, j] = 0.0
        else:
            nonconst_data[:, j] = (col - col_min) / scale
    print(np.shape(nonconst_data))

    tau_max = int(np.floor(max_freq / np.mean(np.unique(sorted_freq))))
    print(tau_max)

    dataframe = pp.DataFrame(nonconst_data)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=ALPHA)

    np.savez(
        save_path,
        val_matrix=results["val_matrix"],
        p_matrix=results["p_matrix"],
        var=df.columns,
        subsample=subsample,
        nonconst=nonconst,
    )
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
        var_indices = var_indices[:3]

        stack = [normal_data[max_delay - el[2]: len(normal_data) - el[2], el[0]]
                 for el in var_indices]
        X = np.column_stack(stack)
        y = normal_data[max_delay:, var]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X_scaled)

        if alpha == 0.0:
            model = LinearRegression(fit_intercept=True)
        else:
            model = Ridge(alpha=alpha, fit_intercept=True)

        model.fit(X_poly, y)

        fine_coeffs[var] = model.coef_.copy()
        poly_maps[var] = (scaler, poly)

    return fine_coeffs, indices, poly_maps


# ================================================================
#      ONLINE POLYNOMIAL COEFFICIENTS & ERROR COMPUTATION
# ================================================================
def _rls_update(x, y, w_prev, P_prev, lam=0.995):
    y_pred = float(x @ w_prev)
    e_t = float(y - y_pred)
    Px = P_prev @ x
    denom = float(lam + x @ Px)
    if abs(denom) < 1e-12:
        return w_prev, P_prev, e_t
    g_t = Px / denom
    w_new = w_prev + g_t * e_t
    P_new = (P_prev - np.outer(g_t, Px)) / float(lam)
    return w_new, P_new, e_t


def compute_online_poly_errors(data: np.ndarray, fine_coeffs: dict,
                                causal_matrix: np.ndarray, indices: np.ndarray,
                                poly_maps: dict, alpha: float = RIDGE_ALPHA, rls_lambda: float = RLS_LAMBDA):
    """
    Recompute polynomial coefficients online sample-by-sample via RLS.
    """
    max_time = data.shape[0] - causal_matrix.shape[2]
    unique_vars = np.unique(indices[1, :])
    err = {}
    norm_agg = np.zeros((max_time, len(unique_vars)))

    w_state = {}
    P_state = {}
    var_prep = {}

    for var in unique_vars:
        scaler, poly = poly_maps[var]
        w0 = fine_coeffs[var].copy()
        P0 = np.eye(len(w0)) * (1.0 / max(alpha, 1e-3))
        w_state[var] = w0
        P_state[var] = P0

        var_indices = [indices[:, k] for k in range(indices.shape[1]) if indices[1, k] == var]
        var_indices.sort(key=lambda x: x[2])
        var_indices = var_indices[:3]
        max_delay = var_indices[-1][2]

        stack = [data[max_delay - el[2]: max_time + max_delay - el[2], el[0]] for el in var_indices]
        X = np.column_stack(stack)
        X_poly = poly.transform(scaler.transform(X))
        y = data[max_delay: max_time + max_delay, var]
        var_prep[var] = (X_poly, y)
        err[var] = np.zeros((max_time, len(w0)))

    for t in range(max_time):
        for i, var in enumerate(unique_vars):
            X_poly, y = var_prep[var]
            x_vec = X_poly[t]
            y_t = float(y[t])

            w_prev = w_state[var]
            P_prev = P_state[var]

            w_new, P_new, _ = _rls_update(x_vec, y_t, w_prev, P_prev, rls_lambda)
            w_state[var] = w_new
            P_state[var] = P_new

            err[var][t, :] = w_new - fine_coeffs[var]
            norm_agg[t, i] = np.linalg.norm(err[var][t, :])

    return err, norm_agg


def detect_anomalies(err_normal, err_attack, normal_data_len, normal):
    indices_error = []
    for var in err_attack.keys():
        for j in range(err_attack[var].shape[1]):
            thresh = DETECTION_THRESHOLD_MULTIPLIER * np.linalg.norm(err_normal[var][:normal_data_len, j])
            thresh = max(float(thresh), np.finfo(float).eps)
            if not normal:
                indices_error += list(np.where(abs(err_attack[var][:, j]) < thresh)[0])
            else:
                indices_error += list(np.where(abs(err_attack[var][normal_data_len:, j]) < thresh)[0])
    return len(np.unique(indices_error))


def score_anomaly_timeline(err_normal, err_attack, normal_data_len, normal):
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
            thresh = DETECTION_THRESHOLD_MULTIPLIER * np.linalg.norm(err_normal[var][:normal_data_len, j])
            thresh = max(float(thresh), np.finfo(float).eps)

            # Inverse Signal-to-Noise Ratio (ISNR) for Biological Signal Depression
            err_vals = np.maximum(np.abs(err_attack[var][start:start + max_time, j]), np.finfo(float).eps)
            values = thresh / err_vals
            scores = np.maximum(scores, values)

    predictions = (scores > ANOMALY_SCORE_THRESHOLD).astype(int)
    return predictions, scores


def compute_extended_metrics(normal_predictions, normal_scores, attack_predictions, attack_scores):
    """
    Compute Accuracy, AUC-ROC, and AUC-PR without changing Precision/Recall/F1.
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
#          FEATURE IMPORTANCE PLOTTING (SUBPLOTS)
# ================================================================
def plot_feature_importance_subplots(norm_agg_list, indices, nonconst, var_names, attack_names, output_filename="top_anomalous_variables.png"):
    import math
    import matplotlib.pyplot as plt
    import numpy as np

    n_attacks = len(norm_agg_list)
    if n_attacks == 0:
        return

    # 1. FIX THE PANORAMA: Create a nice grid (max 4 columns wide)
    cols = min(4, n_attacks)
    rows = math.ceil(n_attacks / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    plt.suptitle("Top Anomalous Variables per Attack", fontsize=16)

    for i, norm_agg_attack in enumerate(norm_agg_list):
        r = i // cols
        c = i % cols
        ax = axes[r, c]
        
        # Calculate L2 Norm (Feature Importance)
        dep_vals = {var_names[nonconst[var]]: np.linalg.norm(norm_agg_attack[:, j])
                    for j, var in enumerate(np.unique(indices[1, :]))}
        
        # Sort variables from most anomalous to least anomalous
        dep_sorted = dict(sorted(dep_vals.items(), key=lambda x: x[1], reverse=True))
        
        # 2. FIX THE GIANT SQUARES: Force it to draw up to 15 variables
        top_n_bars = 15
        top_n = min(top_n_bars, len(dep_sorted)) 
        
        if top_n == 0:
            continue
            
        top_items = list(dep_sorted.items())[:top_n]
        top_vars, top_vals = zip(*top_items)

        # Draw the bar chart
        ax.bar(top_vars, top_vals, color='salmon')
        ax.set_xticks(range(len(top_vars)))
        ax.set_xticklabels(top_vars, rotation=45, ha='right')
        ax.set_ylabel("Aggregated Error (L2 Norm)")
        ax.set_title(attack_names[i])

    # Clean up empty grey boxes in the grid
    for i in range(n_attacks, rows * cols):
        r = i // cols
        c = i % cols
        fig.delaxes(axes[r, c])

    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # Leave space for the main title
    if os.path.dirname(output_filename):
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()


# ================================================================
#                         MAIN PIPELINE
# ================================================================
def main():
    print(f"\n========== TASK: {TASK.upper()} ==========")
    print(f"Polynomial degree: {POLY_DEGREE}")
    print(f"Ridge alpha: {RIDGE_ALPHA}")

    causal_path = os.path.join(PREFIX, f"{TASK}_normal.npz")
    print(f"Causal model path: {causal_path}")

    # 1) Learn or load causal model
    if not os.path.exists(causal_path):
        learn_causal_model(os.path.join(PREFIX, NORMAL_FILE), causal_path)
    else:
        print("Causal model found, loading...")

    f = np.load(causal_path, allow_pickle=True)
    val_matrix, p_matrix = f["val_matrix"], f["p_matrix"]
    subsample, nonconst = int(f["subsample"]), f["nonconst"]

    normal_matrix = val_matrix * (p_matrix < ALPHA) * (abs(val_matrix) > CAUSAL_STRENGTH_MULTIPLIER * np.mean(abs(val_matrix)))

    # 2) Load normal data
    normal_df = read_data(os.path.join(PREFIX, NORMAL_FILE), TASK)
    normal_data = np.nan_to_num(normal_df.values[:int(TRAINING_FRAC * len(normal_df))][::subsample, nonconst])
    normal_data_full = np.nan_to_num(normal_df.values[::subsample, nonconst])

    # 3) Offline polynomial coefficients
    fine_coeffs, indices, poly_maps = fit_normal_poly_coeffs(normal_data, normal_matrix)

    # 4) Online deviations for normal reference
    err_normal, norm_agg_normal = compute_online_poly_errors(normal_data_full, fine_coeffs, normal_matrix, indices, poly_maps)
    _ = norm_agg_normal

    # 5) Load and detect anomalies
    attack_data_dict = pd.read_pickle(os.path.join(PREFIX, ATTACK_FILE))
    if isinstance(attack_data_dict, dict):
        attack_names = list(attack_data_dict.keys())
        attack_dfs = [attack_data_dict[k][[c for c in ECG_LEADS if c in attack_data_dict[k].columns]] for k in attack_names]
    else:
        attack_paths = [os.path.join(PREFIX, fname) for fname in [ATTACK_FILE]]
        attack_dfs = [read_data(p, TASK) for p in attack_paths]
        attack_names = [os.path.basename(p) for p in attack_paths]

    tpos, fpos, fneg = [], [], []
    attack_predictions, attack_scores = [], []

    # False positives
    fpos.append(detect_anomalies(err_normal, err_normal, len(normal_data), normal=True))
    normal_predictions, normal_scores = score_anomaly_timeline(
        err_normal, err_normal, len(normal_data), normal=True
    )

    norm_agg_attacks = []

    for attack_name, df_attack in zip(attack_names, attack_dfs):
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

    # 6) Metrics & Reporting via Shared Helper
    import reporting_helper

    attack_scores_dict = dict(zip(attack_names, attack_scores))
    attack_labels_dict = {}
    target_wise_residuals_by_file = dict(zip(attack_names, norm_agg_attacks))

    for name, attack_df in zip(attack_names, attack_dfs):
        data_matrix = np.nan_to_num(attack_df.values[::subsample, nonconst])
        attack_labels_dict[name] = np.ones(data_matrix.shape[0])

    output_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    reporting_helper.generate_plots_and_reports(
        output_dir=output_dir,
        model_variant="Polynomial",
        model_name="PCMCI",
        normal_scores=normal_scores,
        attack_scores_by_file=attack_scores_dict,
        threshold=ANOMALY_SCORE_THRESHOLD,
        var_names=f["var"][nonconst].tolist(),
        target_indices=np.unique(indices[1, :]).tolist(),
        directed_edges=[], 
        target_wise_residuals_by_file=target_wise_residuals_by_file,
        attack_labels_by_file=attack_labels_dict,
        dataset_name="Healthcare (ECG)"
    )


if __name__ == "__main__":
    main()
