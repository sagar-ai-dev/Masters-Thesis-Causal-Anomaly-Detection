"""
==============================================================
 Causal Discovery & Anomaly Detection (Industrial - Polynomial Authentic TS-CAM-UV)
==============================================================
"""

import os
import numpy as np
import pandas as pd
import warnings
from scipy.stats import ConstantInputWarning
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

from causallearn.search.FCMBased.lingam import CAMUV

warnings.filterwarnings('ignore', category=ConstantInputWarning)

# ================== GLOBAL CONFIG ==================
ALPHA = 0.8
TRAINING_FRAC = 0.7
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREFIX = os.path.join(BASE_DIR, "TEP")
TASK = "industrial"
MAX_FREQ_COMPONENTS = 5

NORMAL_FILE = "TEP_FaultFree_Training_run1.pkl"
ATTACK_FILE = "TEP_Faulty_Testing_fault1.pkl"

# Polynomial regression config
POLY_DEGREE = 3
RIDGE_ALPHA = 10.0

# Anomaly detection tuning config
DETECTION_THRESHOLD_MULTIPLIER = 3.0
CAUSAL_STRENGTH_MULTIPLIER = 0.0
CONSECUTIVE_K = 1




# ================================================================
#                         DATA LOADING
# ================================================================
def read_data(path: str, task: str) -> pd.DataFrame:
    if path.endswith(".pkl"):
        res = pd.read_pickle(path)
        if isinstance(res, pd.DataFrame):
            tep_vars = [f"XMEAS_{i}" for i in range(1, 42)] + [f"XMV_{i}" for i in range(1, 12)]
            available_vars = [c for c in tep_vars if c in res.columns]
            return res[available_vars] if available_vars else res
        return res
    else:
        df = pd.read_csv(path, delimiter="," if task == "pepper" else ";")
        tep_vars = [f"XMEAS_{i}" for i in range(1, 42)] + [f"XMV_{i}" for i in range(1, 12)]
        available_vars = [c for c in tep_vars if c in df.columns]
        if available_vars:
            df = df[available_vars]
        return df


# ================================================================
#                AUTHENTIC TS-CAM-UV CAUSAL MODEL
# ================================================================
def run_camuv_causal_discovery(data: np.ndarray, tau_max: int, alpha: float = 0.05):
    """
    Authentic TS-CAM-UV Time-Series Causal Discovery using causallearn.
    Calls CAMUV.execute(X, alpha, num_explanatory_vals) to discover parent sets P.
    """
    T, N = data.shape
    print(f"Starting CAMUV execution on {N} variables...")
    val_matrix = np.zeros((N, N, tau_max + 1))
    p_matrix = np.ones((N, N, tau_max + 1))

    num_explanatory = min(3, max(1, N - 1))
    P, U = CAMUV.execute(data, alpha=alpha, num_explanatory_vals=num_explanatory)
    print("CAMUV execution complete. Mapping parent sets...")

    for child_var, parents in enumerate(P):
        for parent_var in parents:
            if parent_var < N:
                for tau in range(1, tau_max + 1):
                    x_lag = data[tau_max - tau : T - tau, parent_var]
                    y_tgt = data[tau_max :, child_var]
                    if np.std(x_lag) > 1e-8 and np.std(y_tgt) > 1e-8:
                        corr = np.corrcoef(x_lag, y_tgt)[0, 1]
                        if not np.isnan(corr):
                            val_matrix[parent_var, child_var, tau] = corr
                            p_matrix[parent_var, child_var, tau] = max(1e-5, 1.0 - abs(corr))
                    else:
                        val_matrix[parent_var, child_var, tau] = 1.0
                        p_matrix[parent_var, child_var, tau] = 0.001

    return {"val_matrix": val_matrix, "p_matrix": p_matrix}


def learn_causal_model(normal_csv_path: str, save_path: str):
    print("Learning authentic TS-CAM-UV causal model...")

    df = read_data(normal_csv_path, TASK)
    raw_values = np.nan_to_num(df.values)
    raw_values = raw_values[:int(TRAINING_FRAC * raw_values.shape[0]), :]

    frequencies = []
    for index in range(raw_values.shape[1]):
        if any([el for el in raw_values[:, index] if int(el) != el]):
            w = np.fft.fft(raw_values[:, index])
            freqs = np.fft.fftfreq(len(w))
            mods = abs(w)
            max_indices = np.argsort(mods)[::-1][:MAX_FREQ_COMPONENTS]
            main_freq = [freqs[i] for i in max_indices]
            frequencies += main_freq

    sorted_freq = np.sort([el for el in frequencies if el > 0])[::-1]
    max_freq = 0.05
    for freq in sorted_freq:
        if len([fr for fr in sorted_freq if fr < freq]) / len(sorted_freq) < 0.95:
            max_freq = freq
            sorted_freq = [s for s in sorted_freq if s <= max_freq]
            break

    subsample = max(1, int(np.floor(1 / 10 / max_freq)))
    sub_data = raw_values[::subsample, :]
    nonconst = [idx for idx in range(sub_data.shape[1]) if np.std(sub_data[:, idx]) > 0.01 * (np.mean(sub_data[:, idx]) + 1e-5)]
    if not nonconst:
        nonconst = list(range(sub_data.shape[1]))

    nonconst_data = sub_data[:, nonconst]
    for j in range(nonconst_data.shape[1]):
        denom = (np.max(nonconst_data[:, j]) - np.min(nonconst_data[:, j]))
        if denom > 1e-8:
            nonconst_data[:, j] = (nonconst_data[:, j] - np.min(nonconst_data[:, j])) / denom

    mean_freq = np.mean(np.unique(sorted_freq)) if len(sorted_freq) > 0 else 0.05
    tau_max = max(1, int(np.floor(max_freq / (mean_freq + 1e-5))))

    camuv_train_data = nonconst_data[:min(400, nonconst_data.shape[0]), :]
    results = run_camuv_causal_discovery(camuv_train_data, tau_max=tau_max, alpha=ALPHA)

    np.savez(save_path,
             val_matrix=results["val_matrix"],
             p_matrix=results["p_matrix"],
             var=df.columns,
             subsample=subsample,
             nonconst=nonconst)
    print(f"Saved authentic TS-CAM-UV causal model to {save_path}")
    return results, subsample, nonconst, tau_max


# ================================================================
#           OFFLINE POLYNOMIAL COEFFICIENT FITTING
# ================================================================
def fit_normal_poly_coeffs(normal_data: np.ndarray, causal_matrix: np.ndarray, degree: int = POLY_DEGREE, alpha: float = RIDGE_ALPHA):
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.linear_model import Ridge
    indices = np.array(np.where(causal_matrix != 0))
    fine_coeffs = {}
    poly_maps = {}

    for var in np.unique(indices[1, :]):
        var_indices = [indices[:, k] for k in range(indices.shape[1]) if indices[1, k] == var]
        var_indices.sort(key=lambda x: x[2])
        max_delay = var_indices[-1][2]
        var_indices = var_indices[:3]

        stack = [normal_data[max_delay - el[2]: len(normal_data) - el[2], el[0]] for el in var_indices]
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


def compute_online_poly_errors(data: np.ndarray, fine_coeffs: dict, causal_matrix: np.ndarray, indices: np.ndarray, poly_maps: dict, alpha: float = RIDGE_ALPHA):
    from sklearn.linear_model import Ridge
    max_time = data.shape[0] - causal_matrix.shape[2]
    err = {}
    norm_agg = np.zeros((max_time, len(np.unique(indices[1, :]))))

    for t in range(max_time):
        for i, var in enumerate(np.unique(indices[1, :])):
            var_indices = [indices[:, k] for k in range(indices.shape[1]) if indices[1, k] == var]
            var_indices.sort(key=lambda x: x[2])
            max_delay = var_indices[-1][2]
            var_indices = var_indices[:3]

            end_idx = t + causal_matrix.shape[2]
            stack = [data[max_delay - el[2]: end_idx - el[2], el[0]] for el in var_indices]
            X = np.column_stack(stack)
            y = data[max_delay:end_idx, var]

            scaler, poly = poly_maps[var]
            X_scaled = scaler.transform(X)
            X_poly = poly.transform(X_scaled)

            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(X_poly, y)
            coeffs = np.concatenate([[float(model.intercept_)], np.ravel(model.coef_)])

            if var not in err:
                err[var] = np.zeros((max_time, len(coeffs)))
            err[var][t, :] = coeffs - fine_coeffs[var]
            norm_agg[t, i] = np.linalg.norm(err[var][t, :])

    return err, norm_agg


def detect_anomalies(err_normal, err_attack, normal_data_len, normal):
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


def main():
    print(f"\n========== TASK: {TASK.upper()} ==========")
    print(f"Polynomial degree: {POLY_DEGREE}")
    print(f"Ridge alpha: {RIDGE_ALPHA}")

    causal_path = os.path.join(PREFIX, f"{TASK}_normal.npz")
    print(f"Causal model path: {causal_path}")

    if not os.path.exists(causal_path):
        learn_causal_model(os.path.join(PREFIX, NORMAL_FILE), causal_path)
    else:
        print("Causal model found, loading...")

    f = np.load(causal_path, allow_pickle=True)
    val_matrix, p_matrix = f["val_matrix"], f["p_matrix"]
    subsample, nonconst = int(f["subsample"]), f["nonconst"]

    normal_matrix = val_matrix * (p_matrix < ALPHA) * (abs(val_matrix) > CAUSAL_STRENGTH_MULTIPLIER * np.mean(abs(val_matrix)))

    normal_df = read_data(os.path.join(PREFIX, NORMAL_FILE), TASK)
    normal_data = np.nan_to_num(normal_df.values[:int(TRAINING_FRAC * len(normal_df))][::subsample, nonconst])
    normal_data_full = np.nan_to_num(normal_df.values[::subsample, nonconst])

    fine_coeffs, indices, poly_maps = fit_normal_poly_coeffs(normal_data, normal_matrix)
    err_normal, norm_agg_normal = compute_online_poly_errors(normal_data_full, fine_coeffs, normal_matrix, indices, poly_maps)
    _ = norm_agg_normal

    attack_paths = [os.path.join(PREFIX, f"TEP_Faulty_Testing_fault{i}.pkl") for i in range(1, 21)]
    attack_dfs = [read_data(p, TASK) for p in attack_paths]
    attack_names = [f"Fault_{i}" for i in range(1, 21)]

    tpos, fpos, fneg = [], [], []
    attack_predictions, attack_scores = [], []

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
        model_name="TS-CAM-UV",
        normal_scores=normal_scores,
        attack_scores_by_file=attack_scores_dict,
        threshold=1.0,
        var_names=f["var"][nonconst].tolist(),
        target_indices=np.unique(indices[1, :]).tolist(),
        directed_edges=[], 
        target_wise_residuals_by_file=target_wise_residuals_by_file,
        attack_labels_by_file=attack_labels_dict,
        dataset_name="Industrial (TEP)"
    )


if __name__ == "__main__":
    main()
