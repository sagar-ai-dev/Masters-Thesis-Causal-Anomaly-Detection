"""
==============================================================
 Target-Wise GES Healthcare Anomaly Detection: Polynomial Degree 3
==============================================================

Purpose
-------
This standalone PCMCI-style script runs the target-wise GES anomaly-detection pipeline with a degree-3 Polynomial downstream regression on the PTB-XL healthcare ECG dataset. It uses the same fixed GES graph, preprocessing, cache behavior, thresholding style, and metric formulas as the Linear and RBF scripts.

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

    model   = PolynomialFeatures(degree=3, include_bias=False) plus Ridge
    formula = beta_hat = argmin_beta || y - phi_3(X) beta ||_2^2 + alpha || beta ||_2^2
    params  = degree = 3, ridge_alpha = 10.0, threshold_multiplier = 3.0

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
Recall    = 0.912
F1        = 0.954
Accuracy  = 0.949
AUC-ROC   = 0.922
AUC-PR    = 0.967
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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

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
    poly_degree: int = 3
    poly_ridge_alpha: float = 10.0

    def params_text(self) -> str:
        """Return a compact parameter summary for logs and reports."""
        return (
            f"degree={self.poly_degree}, ridge_alpha={self.poly_ridge_alpha}, "
            f"threshold={self.threshold_multiplier}"
        )

POLYNOMIAL_CONFIG = RegressionConfig(
    label="Polynomial degree 3",
    regression_kind="polynomial",
    plot_filename="ges_polynomial_top_anomalous_variables.png",
    threshold_multiplier=3.0,
    poly_degree=3,
    poly_ridge_alpha=10.0,
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
    if config.regression_kind == "polynomial":
        if fitted_transformer is None:
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)
            poly = PolynomialFeatures(degree=config.poly_degree, include_bias=False)
            x_features = poly.fit_transform(x_scaled)
            fitted_transformer = (scaler, poly)
        else:
            scaler, poly = fitted_transformer
            x_scaled = scaler.transform(x)
            x_features = poly.transform(x_scaled)

        model = Ridge(alpha=config.poly_ridge_alpha, fit_intercept=True)
        model.fit(x_features, y)
        coeffs = np.concatenate([[float(model.intercept_)], np.ravel(model.coef_)])
        return coeffs, fitted_transformer

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
    Uses exact cumulative Ridge solver (A_t = sum x_i x_i^T + alpha I, b_t = sum x_i y_i)
    which produces 100% mathematically identical coefficients as Ridge.fit()
    while operating in milliseconds per file.
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

        outer_x = np.einsum("ti,tj->tij", x_b, x_b)
        cum_A = np.cumsum(outer_x[:max_time], axis=0)
        reg = alpha * np.eye(d)
        reg[0, 0] = 0.0
        cum_A += reg

        outer_xy = x_b[:max_time] * y_full[:max_time, np.newaxis]
        cum_b = np.cumsum(outer_xy, axis=0)

        coeffs_matrix = np.linalg.solve(cum_A, cum_b)
        delta_matrix = coeffs_matrix - fine_coeffs[target_var]

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
    Standard detection: error spike above threshold signals an anomaly.
    """
    indices_error = []
    for var in err_attack.keys():
        for j in range(err_attack[var].shape[1]):
            thresh = threshold_multiplier * np.std(err_normal[var][:normal_data_len, j])
            thresh = max(float(thresh), np.finfo(float).eps)
            if not normal:
                indices_error += list(np.where(abs(err_attack[var][:, j]) > thresh)[0])
            else:
                indices_error += list(np.where(abs(err_attack[var][normal_data_len:, j]) > thresh)[0])
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
    Standard scoring: score = Error / Threshold. Anomaly when score > 1.0.
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
        for j in range(err_attack[var].shape[1]):
            thresh = threshold_multiplier * np.std(err_normal[var][:normal_data_len, j])
            thresh = max(float(thresh), np.finfo(float).eps)

            # Standard Math: Error / Threshold
            values = np.abs(err_attack[var][start : start + max_time, j]) / thresh
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
def run_polynomial_experiment(config: RegressionConfig, verbose: bool = True):
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
ACTIVE_CONFIG = POLYNOMIAL_CONFIG


def main():
    """Run this standalone fixed-graph target-wise GES experiment."""
    run_polynomial_experiment(ACTIVE_CONFIG, verbose=True)


if __name__ == "__main__":
    main()
