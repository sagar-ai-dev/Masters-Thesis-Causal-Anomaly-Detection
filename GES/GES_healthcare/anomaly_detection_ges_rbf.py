"""
==============================================================
 Target-Wise GES Healthcare Anomaly Detection: RBF + Ridge
==============================================================

Purpose
-------
This standalone PCMCI-style script runs the target-wise GES anomaly-detection pipeline with RBF random Fourier features and Ridge regression on the PTB-XL healthcare ECG dataset. It uses the same fixed GES graph, preprocessing, cache behavior, thresholding style, and metric formulas as the Linear and Polynomial scripts.

Fixed Causal Graph
------------------
The causal-discovery result is shared by all three final GES scripts:

    score function          = local_score_BIC
    TOP_PARENT_CANDIDATES   = 5
    MAX_TARGETS             = 0  (all targets)

The graph cache behavior is unchanged. Existing cache file
Healthcare dataset/healthcare_normal_ges.npz is reused if present; otherwise
the same target-wise GES learning logic creates it.

Dataset
-------
The PTB-XL healthcare ECG dataset is used:

    healthcare_normal_full.pkl
    healthcare_normal_train.pkl
    healthcare_attacks.pkl

Regression Model
----------------
For each target variable, retained lagged GES parents are arranged into a
parent-lag design matrix X and the current target value is y.

    model   = StandardScaler plus RBFSampler plus Ridge
    formula = beta_hat = argmin_beta || y - z(X) beta ||_2^2 + alpha || beta ||_2^2
    params  = gamma = 0.001, n_components = 100, ridge_alpha = 10.0, random_state = 42, threshold_multiplier = 1.0

Coefficient-Drift Anomaly Detection
-----------------------------------
1. Fit baseline coefficients on normal training data.
2. Refit coefficients over moving online windows.
3. Compute coefficient drift as the difference between online and baseline
   coefficients.
4. Threshold the drift to count anomalies.
5. Report Precision, Recall, F1, Accuracy, AUC-ROC, and AUC-PR.
6. Save the top-anomalous-variable plot.

Computed Final Result
---------------------
Precision = 1.000
Recall    = 0.478
F1        = 0.647
Accuracy  = 0.697
AUC-ROC   = 0.840
AUC-PR    = 0.926
==============================================================
"""
# ================== IMPORTS ==================
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from causallearn.search.ScoreBased.GES import ges
from scipy.stats import ConstantInputWarning
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConstantInputWarning)


# ================== GLOBAL CONFIG ==================
ALPHA = 0.05
TRAINING_FRAC = 0.7
BASE_DIR = Path(__file__).resolve().parent
PREFIX = BASE_DIR / "Healthcare dataset"
TASK = "healthcare"
MAX_FREQ_COMPONENTS = 5
CAUSAL_STRENGTH_MULTIPLIER = 1.0
GRAPH_SCORE = "local_score_BIC"
TOP_PARENT_CANDIDATES = 5
MAX_TARGETS = 0
ATTACK_FILES = [f"Attack_{i}" for i in range(1, 21)]
ECG_COLUMNS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


@dataclass(frozen=True)
class RegressionConfig:
    """Configuration for one downstream regression experiment."""

    label: str
    regression_kind: str
    plot_filename: str
    threshold_multiplier: float
    rbf_gamma: float = 0.1
    rbf_components: int = 100
    rbf_random_state: int = 42
    ridge_alpha: float = 0.001

    def params_text(self) -> str:
        """Return a compact parameter summary for logs and reports."""
        return (
            f"gamma={self.rbf_gamma}, components={self.rbf_components}, "
            f"ridge_alpha={self.ridge_alpha}, threshold={self.threshold_multiplier}"
        )

RBF_CONFIG = RegressionConfig(
    label="RBF Ridge",
    regression_kind="rbf",
    plot_filename="rbf_ridge_top_anomalous_variables.png",
    threshold_multiplier=1.0,
    rbf_gamma=0.001,
    rbf_components=100,
    rbf_random_state=42,
    ridge_alpha=10.0,
)


# ================================================================
#                         DATA AND GRAPH LOADING
# ================================================================
def read_data(path: str) -> pd.DataFrame:
    """
    Load a healthcare ECG pickle/CSV file and enforce the 12 standard leads.
    """
    path_obj = Path(path)
    if path_obj.suffix == ".pkl":
        df = pd.read_pickle(path_obj)
    else:
        df = pd.read_csv(path_obj, delimiter=",")

    # Drop the label column if it exists
    if "label" in df.columns:
        df = df.drop(columns=["label"])

    # Force column names to be the 12 medical leads to fix numeric X-axis plots
    if len(df.columns) == 12:
        df.columns = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    return df


def select_target_parent_candidates(data: np.ndarray, target_var: int, tau_max: int, top_k: int):
    """
    Select candidate lagged parents using absolute correlation with the target.
    """
    candidates = []
    target = data[tau_max:, target_var]

    for lag in range(1, tau_max + 1):
        for parent_var in range(data.shape[1]):
            candidate = data[tau_max - lag : len(data) - lag, parent_var]
            if np.std(candidate) == 0 or np.std(target) == 0:
                score = 0.0
            else:
                corr = np.corrcoef(candidate, target)[0, 1]
                score = 0.0 if np.isnan(corr) else abs(corr)
            candidates.append((parent_var, lag, score))

    candidates.sort(key=lambda item: item[2], reverse=True)
    return candidates[:top_k]


def build_local_ges_data(data: np.ndarray, target_var: int, candidates: list, tau_max: int):
    """
    Build the local target-wise GES data matrix for one target.
    """
    rows = []
    for t in range(tau_max, len(data)):
        row = []
        for parent_var, lag, _ in candidates:
            row.append(data[t - lag, parent_var])
        row.append(data[t, target_var])
        rows.append(row)

    target_column_idx = len(candidates)
    return np.asarray(rows), target_column_idx


def _endpoint_is_tail(endpoint) -> bool:
    return "TAIL" in str(endpoint).upper()


def _endpoint_is_arrow(endpoint) -> bool:
    return "ARROW" in str(endpoint).upper()


def _node_to_index(graph, node) -> int:
    nodes = graph.get_nodes()
    for idx, candidate in enumerate(nodes):
        if candidate == node:
            return idx

    name = node.get_name() if hasattr(node, "get_name") else str(node)
    for idx, candidate in enumerate(nodes):
        candidate_name = candidate.get_name() if hasattr(candidate, "get_name") else str(candidate)
        if candidate_name == name:
            return idx

    raise ValueError(f"Could not map GES node to column index: {node}")


def ges_edge_exists(graph, parent_idx: int, child_idx: int) -> bool:
    """
    Return True only for a fully directed parent -> child GES edge.
    """
    if not hasattr(graph, "get_graph_edges"):
        return False

    for edge in graph.get_graph_edges():
        node1_idx = _node_to_index(graph, edge.get_node1())
        node2_idx = _node_to_index(graph, edge.get_node2())
        endpoint1 = edge.get_endpoint1()
        endpoint2 = edge.get_endpoint2()

        if (
            node1_idx == parent_idx
            and node2_idx == child_idx
            and _endpoint_is_tail(endpoint1)
            and _endpoint_is_arrow(endpoint2)
        ):
            return True

        if (
            node2_idx == parent_idx
            and node1_idx == child_idx
            and _endpoint_is_tail(endpoint2)
            and _endpoint_is_arrow(endpoint1)
        ):
            return True

    return False


def run_local_ges_for_target(data: np.ndarray, target_var: int, tau_max: int, top_k: int):
    """
    Run local GES for one target and return directed lagged-parent edges.
    """
    candidates = select_target_parent_candidates(data, target_var, tau_max, top_k)
    if not candidates:
        return []

    local_data, target_column_idx = build_local_ges_data(data, target_var, candidates, tau_max)
    local_data = np.nan_to_num(local_data, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        ges_results = ges(local_data)
    except np.linalg.LinAlgError as exc:
        print(f"Skipping target {target_var}: local GES failed with {exc}.")
        return []

    graph = ges_results["G"]
    edges = []
    for parent_idx, (parent_var, lag, _) in enumerate(candidates):
        if ges_edge_exists(graph, parent_idx, target_column_idx):
            edges.append((parent_var, target_var, lag))
    return edges


def select_targets_for_ges(data: np.ndarray):
    """
    Process all target variables for the final thesis configuration.
    """
    return list(range(data.shape[1]))


def learn_causal_model(normal_csv_path: str, save_path: str):
    """
    Learn and cache the canonical target-wise GES graph.

    This function mirrors the original GES causal discovery behavior and is
    kept in this standalone script to match the PCMCI-style layout.
    """
    print("Learning causal model with GES...")
    df = read_data(normal_csv_path)
    normal_data = np.nan_to_num(df.values)
    normal_data = normal_data[: int(TRAINING_FRAC * np.shape(normal_data)[0]), :]

    frequencies = []
    for index in range(np.shape(normal_data)[1]):
        if any([el for el in normal_data[:, index] if int(el) != el]):
            w = np.fft.fft(normal_data[:, index])
            freqs = np.fft.fftfreq(len(w))
            mods = abs(w)
            max_indices = np.argsort(mods)[::-1][:MAX_FREQ_COMPONENTS]
            for i in max_indices:
                frequencies.append(freqs[i])

    sorted_freq = np.sort([el for el in frequencies if el > 0])[::-1]
    if len(sorted_freq) == 0:
        sorted_freq = np.asarray([1.0])
    max_freq = float(sorted_freq[0])
    for freq in sorted_freq:
        if len([fr for fr in sorted_freq if fr < freq]) / len(sorted_freq) < 0.95:
            max_freq = float(freq)
            sorted_freq = [s for s in sorted_freq if s <= max_freq]
            break

    subsample = max(1, int(np.floor(1 / 10 / max_freq)))
    normal_data = normal_data[:: max(1, subsample), :]
    nonconst = [
        idx
        for idx in range(np.shape(normal_data)[1])
        if np.std(normal_data[:, idx]) > 0.01 * np.mean(normal_data[:, idx])
    ]
    nonconst_data = normal_data[:, nonconst]
    for j in range(np.shape(nonconst_data)[1]):
        denom = (np.max(nonconst_data[:, j]) - np.min(nonconst_data[:, j])) + np.min(nonconst_data[:, j])
        if denom != 0:
            nonconst_data[:, j] /= denom
    nonconst_data = np.nan_to_num(nonconst_data, nan=0.0, posinf=0.0, neginf=0.0)

    tau_max = max(1, int(np.floor(max_freq / np.mean(np.unique(sorted_freq)))))
    num_vars = nonconst_data.shape[1]
    val_matrix = np.zeros((num_vars, num_vars, tau_max + 1))
    p_matrix = np.ones((num_vars, num_vars, tau_max + 1))
    edge_count = 0
    targets = select_targets_for_ges(nonconst_data)
    targets_processed = 0

    print("GES mode: target_wise")
    print("Candidate selection method: correlation")
    print(f"TOP_PARENT_CANDIDATES: {TOP_PARENT_CANDIDATES}")
    print(f"Targets to process: {len(targets)}")

    for target_var in targets:
        edges = run_local_ges_for_target(nonconst_data, target_var, tau_max, TOP_PARENT_CANDIDATES)
        targets_processed += 1
        for parent_var, child_var, lag in edges:
            if val_matrix[parent_var, child_var, lag] == 0:
                edge_count += 1
            val_matrix[parent_var, child_var, lag] = 1.0
            p_matrix[parent_var, child_var, lag] = 0.0

        if targets_processed % 25 == 0 or targets_processed == len(targets):
            print(f"Processed {targets_processed}/{len(targets)} targets; edges kept so far: {edge_count}")

    np.savez(
        save_path,
        val_matrix=val_matrix,
        p_matrix=p_matrix,
        var=df.columns,
        subsample=subsample,
        nonconst=nonconst,
        edge_count=edge_count,
        tau_max=tau_max,
        ges_mode="target_wise",
        top_parent_candidates=TOP_PARENT_CANDIDATES,
        candidate_selection_method="correlation",
        targets_processed=targets_processed,
        max_targets=MAX_TARGETS,
    )
    print(f"GES causal edges kept: {edge_count}")
    print(f"Saved GES causal model to {save_path}")
    return {"val_matrix": val_matrix, "p_matrix": p_matrix}, subsample, nonconst, tau_max


def graph_cache_path() -> Path:
    """Return the canonical target-wise GES graph cache path."""
    return PREFIX / f"{TASK}_normal_ges.npz"


def ensure_canonical_ges_graph() -> Path:
    """
    Create the canonical target-wise GES graph if it is missing.

    Existing graph caches are reused. This prevents the regression comparison
    from accidentally changing the causal discovery result.
    """
    path = graph_cache_path()
    if not path.exists():
        learn_causal_model(str(PREFIX / "healthcare_normal_full.pkl"), str(path))
    return path


def load_canonical_graph():
    """
    Load the fixed target-wise GES graph and return matrices plus metadata.
    """
    path = ensure_canonical_ges_graph()
    graph = np.load(path, allow_pickle=True)
    val_matrix = graph["val_matrix"]
    p_matrix = graph["p_matrix"]
    subsample = int(graph["subsample"])
    nonconst = graph["nonconst"]
    var_names = graph["var"]

    normal_matrix = val_matrix * (p_matrix < ALPHA) * (
        np.abs(val_matrix) > CAUSAL_STRENGTH_MULTIPLIER * np.mean(np.abs(val_matrix))
    )
    if not np.any(normal_matrix):
        print("Warning: no GES edges survived strength filtering. Using raw directed GES edges.")
        normal_matrix = val_matrix * (p_matrix < ALPHA)
    if not np.any(normal_matrix):
        raise RuntimeError("The fixed target-wise GES graph contains no usable directed links.")

    return {
        "path": path,
        "normal_matrix": normal_matrix,
        "subsample": subsample,
        "nonconst": nonconst,
        "var_names": var_names,
        "edge_count": int(np.sum(normal_matrix != 0)),
    }


def load_healthcare_matrices(graph_info):
    """
    Load the PTB-XL healthcare normal and attack datasets.

    The stored pickles are used directly, while the normal training subset is
    used only for offline baseline coefficient fitting.
    """
    normal_full_df = read_data(str(PREFIX / "healthcare_normal_full.pkl"))
    attack_bundle = pd.read_pickle(PREFIX / "healthcare_attacks.pkl")

    subsample = graph_info["subsample"]
    nonconst = graph_info["nonconst"]

    normal_full = np.nan_to_num(normal_full_df.values[::subsample, nonconst])
    normal_train = normal_full[: int(TRAINING_FRAC * len(normal_full))]

    attack_data = []
    for attack_file in ATTACK_FILES:
        attack_df = attack_bundle[attack_file]
        attack_df = attack_df[[column for column in ECG_COLUMNS if column in attack_df.columns]]
        attack_data.append((attack_file, np.nan_to_num(attack_df.values[::subsample, nonconst])))

    return normal_train, normal_full, attack_data


# ================================================================
#                   PARENT-LAG DESIGN MATRICES
# ================================================================
def target_parent_groups(causal_matrix: np.ndarray):
    """
    Group retained parent-lag links by target variable.

    The original GES pipeline keeps the first/top three retained parent-lag
    links per target in the coefficient stage. The same rule is preserved for
    all three downstream regressions.
    """
    indices = np.array(np.where(causal_matrix != 0))
    groups = {}
    for var in np.unique(indices[1, :]):
        var_indices = [indices[:, k] for k in range(indices.shape[1]) if indices[1, k] == var]
        var_indices.sort(key=lambda x: x[2])
        groups[int(var)] = var_indices[:3]
    return indices, groups


def build_parent_lag_xy(data: np.ndarray, var_indices: list, end_idx=None):
    """
    Build X/y arrays for one target from retained lagged parents.
    """
    max_delay = int(var_indices[-1][2])
    if end_idx is None:
        end_idx = len(data)

    stack = [
        data[max_delay - int(edge[2]) : end_idx - int(edge[2]), int(edge[0])]
        for edge in var_indices
    ]
    x = np.column_stack(stack)
    y = data[max_delay:end_idx, int(var_indices[0][1])]
    return x, y


# ================================================================
#                    REGRESSION MODEL FITTING
# ================================================================
def fit_regression_model(x: np.ndarray, y: np.ndarray, config: RegressionConfig, fitted_transformer=None):
    """
    Fit one downstream regression model and return comparable coefficients.

    Coefficient vectors include the intercept as the first value. The anomaly
    score is based on coefficient drift, so each online refit must use the same
    feature mapping learned during the offline fit.
    """
    if config.regression_kind == "rbf":
        if fitted_transformer is None:
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)
            rbf_map = RBFSampler(
                gamma=config.rbf_gamma,
                n_components=config.rbf_components,
                random_state=config.rbf_random_state,
            )
            x_features = rbf_map.fit_transform(x_scaled)
            transformer = (scaler, rbf_map)
        else:
            scaler, rbf_map = fitted_transformer
            x_scaled = scaler.transform(x)
            x_features = rbf_map.transform(x_scaled)
            transformer = fitted_transformer

        model = Ridge(alpha=config.ridge_alpha, fit_intercept=True)
        model.fit(x_features, y)
        coeffs = np.concatenate([[float(model.intercept_)], np.ravel(model.coef_)])
        return coeffs, transformer

    raise ValueError(f"Unknown regression kind: {config.regression_kind}")


def fit_offline_coefficients(normal_train: np.ndarray, causal_matrix: np.ndarray, config: RegressionConfig):
    """
    Fit baseline coefficients for each target on normal training data.
    """
    indices, groups = target_parent_groups(causal_matrix)
    fine_coeffs = {}
    transformers = {}

    for target_var, var_indices in groups.items():
        x, y = build_parent_lag_xy(normal_train, var_indices)
        coeffs, transformer = fit_regression_model(x, y, config)
        fine_coeffs[target_var] = coeffs
        transformers[target_var] = transformer

    return fine_coeffs, indices, groups, transformers


def compute_online_errors(data, fine_coeffs, causal_matrix, indices, groups, transformers, config):
    """
    Refit coefficients online and compute L2 coefficient-drift errors.
    Uses exact Recursive Least Squares (RLS) update which provides instantaneous
    online refitting for high-dimensional feature spaces (RBF / Poly) with 0 memory overhead.
    """
    max_time = data.shape[0] - causal_matrix.shape[2]
    targets = list(groups.keys())
    err = {}
    norm_agg = np.zeros((max_time, len(targets)))
    alpha = getattr(config, "ridge_alpha", getattr(config, "poly_ridge_alpha", 10.0))

    for col, target_var in enumerate(targets):
        x_full, y_full = build_parent_lag_xy(data, groups[target_var], end_idx=data.shape[0])
        tr = transformers.get(target_var)
        if tr is not None:
            if hasattr(tr, "transform"):
                x_design = tr.transform(x_full)
            elif isinstance(tr, tuple) and len(tr) == 2:
                scaler, rbf_map = tr
                x_design = rbf_map.transform(scaler.transform(x_full))
            else:
                x_design = x_full
        else:
            x_design = x_full

        x_b = np.column_stack([np.ones(len(x_design)), x_design])
        d = x_b.shape[1]

        P = np.eye(d) / alpha
        P[0, 0] = 1e6
        beta = np.zeros(d)
        delta_matrix = np.zeros((max_time, d))

        for t in range(max_time):
            xt = x_b[t]
            u = P @ xt
            k = u / (1.0 + xt @ u)
            P -= np.outer(k, u)
            beta += k * (y_full[t] - float(xt @ beta))
            delta_matrix[t] = beta - fine_coeffs[target_var]

        err[target_var] = delta_matrix
        norm_agg[:, col] = np.linalg.norm(delta_matrix, axis=1)

    return err, norm_agg


# ================================================================
#               ANOMALY DETECTION & METRICS
# ================================================================
def detect_anomalies(
    err_normal: dict,
    err_attack: dict,
    normal_data_len: int,
    normal: bool,
    threshold_multiplier: float,
):
    """
    L2-norm aggregate detection: uses the L2 norm of the full coefficient
    drift vector per target per time step. Robust for high-dimensional RBF
    feature spaces where per-coefficient thresholds are noisy.
    """
    indices_error = []
    for var in err_attack.keys():
        # Compute L2 norm of training drift for threshold calibration
        train_norms = np.linalg.norm(err_normal[var][:normal_data_len], axis=1)
        thresh = threshold_multiplier * np.std(train_norms)
        thresh = max(float(thresh), np.finfo(float).eps)
        if not normal:
            attack_norms = np.linalg.norm(err_attack[var], axis=1)
            indices_error += list(np.where(attack_norms > thresh)[0])
        else:
            test_norms = np.linalg.norm(err_attack[var][normal_data_len:], axis=1)
            indices_error += list(np.where(test_norms > thresh)[0])
    return len(np.unique(indices_error))


def precision_recall_f1(tpos, fpos, fneg):
    """
    Compute Precision, Recall, and F1 using the same formulas as the original pipeline.
    """
    precision = np.sum(tpos) / (np.sum(tpos) + np.sum(fpos))
    recall = np.sum(tpos) / (np.sum(tpos) + np.sum(fneg))
    f1 = 2 * np.sum(tpos) / (2 * np.sum(tpos) + np.sum(fpos) + np.sum(fneg))
    return float(precision), float(recall), float(f1)


def score_and_predict(
    err_normal: dict,
    err_attack: dict,
    normal_data_len: int,
    normal: bool,
    threshold_multiplier: float,
    total_samples: int = None,
):
    """
    L2-norm aggregate scoring for high-dimensional RBF feature spaces.
    score = L2_norm(drift_vector) / (threshold_multiplier * std(L2_norm_training)).
    Anomaly when score > 1.0. Robust against per-dimension noise in 100-dim RBF space.
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

    if max_time <= 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float)

    scores = np.zeros(max_time, dtype=float)
    for var in err_attack.keys():
        # Threshold = multiplier * std of training L2 norms
        train_norms = np.linalg.norm(err_normal[var][:normal_data_len], axis=1)
        thresh = threshold_multiplier * np.std(train_norms)
        thresh = max(float(thresh), np.finfo(float).eps)

        # Score = L2 norm of full drift vector / threshold
        drift_norms = np.linalg.norm(err_attack[var][start : start + max_time], axis=1)
        values = drift_norms / thresh
        scores = np.maximum(scores, values)

    predictions = (scores > 1.0).astype(int)
    if total_samples is not None and total_samples > max_time:
        pad_len = total_samples - max_time
        predictions = np.concatenate([np.zeros(pad_len, dtype=int), predictions])
        scores = np.concatenate([np.zeros(pad_len, dtype=float), scores])

    return predictions, scores


def accuracy_auc_metrics(normal_predictions, normal_scores, attack_predictions, attack_scores):
    """
    Compute Accuracy, AUC-ROC, and AUC-PR from time-index predictions/scores.
    """
    attack_pred_all = np.concatenate(attack_predictions)
    attack_score_all = np.concatenate(attack_scores)
    y_true = np.concatenate([
        np.zeros(len(normal_predictions), dtype=int),
        np.ones(len(attack_pred_all), dtype=int),
    ])
    y_pred = np.concatenate([normal_predictions, attack_pred_all])
    y_score = np.concatenate([normal_scores, attack_score_all])

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    accuracy = float((tp + tn) / len(y_true))
    if len(np.unique(y_true)) > 1:
        auc_roc = float(roc_auc_score(y_true, y_score))
        auc_pr = float(average_precision_score(y_true, y_score))
    else:
        auc_roc = 0.5
        auc_pr = 0.5
    return {
        "accuracy": accuracy,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


# ================================================================
#                       FULL EXPERIMENT PIPELINE
# ================================================================
def run_rbf_experiment(config: RegressionConfig, verbose: bool = True):
    """
    Run one complete fixed-graph GES anomaly-detection experiment.
    """
    started = time.time()
    graph_info = load_canonical_graph()
    normal_train, normal_full, attack_data = load_healthcare_matrices(graph_info)

    if verbose:
        print(f"\n========== TASK: {TASK.upper()} ==========")
        print(f"Regression: {config.label}")
        print(f"Fixed GES graph: {graph_info['path']}")
        print(f"Score function: {GRAPH_SCORE}")
        print(f"TOP_PARENT_CANDIDATES: {TOP_PARENT_CANDIDATES}")
        print(f"MAX_TARGETS: {MAX_TARGETS}")
        print(f"Parameters: {config.params_text()}")

    fine_coeffs, indices, groups, transformers = fit_offline_coefficients(
        normal_train,
        graph_info["normal_matrix"],
        config,
    )
    err_normal, _ = compute_online_errors(
        normal_full,
        fine_coeffs,
        graph_info["normal_matrix"],
        indices,
        groups,
        transformers,
        config,
    )

    tpos, fpos, fneg = [], [], []
    fpos.append(
        detect_anomalies(
            err_normal,
            err_normal,
            len(normal_train),
            normal=True,
            threshold_multiplier=config.threshold_multiplier,
        )
    )
    normal_predictions, normal_scores = score_and_predict(
        err_normal,
        err_normal,
        len(normal_train),
        normal=True,
        threshold_multiplier=config.threshold_multiplier,
        total_samples=max(0, normal_full.shape[0] - len(normal_train)),
    )

    norm_agg_attacks = []
    attack_names = []
    attack_metrics = []
    attack_predictions = []
    attack_scores = []
    for attack_name, attack_matrix in attack_data:
        if verbose:
            print(f"\n--- Analyzing anomaly: {attack_name} ---")
        attack_names.append(attack_name)
        err_attack, norm_agg_attack = compute_online_errors(
            attack_matrix,
            fine_coeffs,
            graph_info["normal_matrix"],
            indices,
            groups,
            transformers,
            config,
        )
        norm_agg_attacks.append(norm_agg_attack)
        true_positive_count = detect_anomalies(
            err_normal,
            err_attack,
            len(normal_train),
            normal=False,
            threshold_multiplier=config.threshold_multiplier,
        )
        tpos.append(true_positive_count)
        false_negative_count = attack_matrix.shape[0] - true_positive_count
        fneg.append(false_negative_count)
        attack_metrics.append({
            "attack": attack_name,
            "samples": int(attack_matrix.shape[0]),
            "true_positive_count": int(true_positive_count),
            "false_negative_count": int(false_negative_count),
            "attack_recall": float(
                true_positive_count / (true_positive_count + false_negative_count)
            ),
        })
        attack_pred, attack_score = score_and_predict(
            err_normal,
            err_attack,
            len(normal_train),
            normal=False,
            threshold_multiplier=config.threshold_multiplier,
            total_samples=attack_matrix.shape[0],
        )
        attack_predictions.append(attack_pred)
        attack_scores.append(attack_score)

    precision, recall, f1 = precision_recall_f1(tpos, fpos, fneg)
    extended_metrics = accuracy_auc_metrics(
        normal_predictions,
        normal_scores,
        attack_predictions,
        attack_scores,
    )

    import reporting_helper

    attack_scores_dict = dict(zip(attack_names, attack_scores))
    attack_labels_dict = {}
    target_wise_residuals_by_file = dict(zip(attack_names, norm_agg_attacks))

    for name, attack_matrix in attack_data:
        attack_labels_dict[name] = np.ones(attack_matrix.shape[0])

    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    reporting_helper.generate_plots_and_reports(
        output_dir=str(output_dir),
        model_variant=config.label,
        model_name="GES",
        normal_scores=normal_scores,
        attack_scores_by_file=attack_scores_dict,
        threshold=config.threshold_multiplier,
        var_names=graph_info["var_names"][graph_info["nonconst"]].tolist(),
        target_indices=np.unique(indices[1, :]).tolist(),
        directed_edges=[],
        target_wise_residuals_by_file=target_wise_residuals_by_file,
        attack_labels_by_file=attack_labels_dict,
        dataset_name="Healthcare (PTB-XL)",
    )

    runtime = time.time() - started

    result = {
        "regression": config.label,
        "regression_kind": config.regression_kind,
        "parameters": config.params_text(),
        "score_function": GRAPH_SCORE,
        "top_parent_candidates": TOP_PARENT_CANDIDATES,
        "max_targets": MAX_TARGETS,
        "graph_cache": str(graph_info["path"]),
        "directed_edges": graph_info["edge_count"],
        "false_positive_count": int(np.sum(fpos)),
        "true_positive_count": int(np.sum(tpos)),
        "false_negative_count": int(np.sum(fneg)),
        "attack_metrics": attack_metrics,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **extended_metrics,
        "runtime_s": runtime,
    }

    if verbose:
        print("\n========== METRICS ==========")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1 Score:  {f1:.3f}")
        print(f"Accuracy:  {extended_metrics['accuracy']:.3f}")
        print(f"AUC-ROC:   {extended_metrics['auc_roc']:.3f}")
        print(f"AUC-PR:    {extended_metrics['auc_pr']:.3f}")
        print("\nAttack-wise recall:")
        for row in attack_metrics:
            print(
                f"  {row['attack']}: TP={row['true_positive_count']}, "
                f"FN={row['false_negative_count']}, recall={row['attack_recall']:.3f}"
            )
        print("=============================\n")

    return result


# ================================================================
#                              MAIN
# ================================================================
ACTIVE_CONFIG = RBF_CONFIG


def main():
    """Run this standalone fixed-graph target-wise GES experiment."""
    run_rbf_experiment(ACTIVE_CONFIG, verbose=True)


if __name__ == "__main__":
    main()
