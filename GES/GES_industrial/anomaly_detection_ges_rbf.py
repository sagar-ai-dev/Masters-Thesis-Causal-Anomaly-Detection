"""
==============================================================
 Target-Wise GES Industrial Anomaly Detection: RBF
==============================================================

Purpose
-------
This standalone PCMCI-style script runs the target-wise GES anomaly-detection
pipeline with RBF kernel feature regression on the Tennessee Eastman Process (TEP)
industrial dataset.

Dataset
-------
The TEP dataset is used:
    TEP_FaultFree_Training_run1.pkl
    TEP_Faulty_Testing_run1.pkl
==============================================================
"""

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
ALPHA = 0.2
TRAINING_FRAC = 0.7
BASE_DIR = Path(__file__).resolve().parent
PREFIX = BASE_DIR / "TEP"
TASK = "industrial"
MAX_FREQ_COMPONENTS = 5
CAUSAL_STRENGTH_MULTIPLIER = 0.0
GRAPH_SCORE = "local_score_BIC"
TOP_PARENT_CANDIDATES = 5
MAX_TARGETS = 0

NORMAL_FILE = "TEP_FaultFree_Training_run1.pkl"
ATTACK_FILES = [f"TEP_Faulty_Testing_fault{i}.pkl" for i in range(1, 21)]


@dataclass(frozen=True)
class RegressionConfig:
    """Configuration for one downstream regression experiment."""

    label: str
    regression_kind: str
    plot_filename: str
    threshold_multiplier: float
    rbf_gamma: float = 0.2
    rbf_components: int = 100
    rbf_random_state: int = 42
    ridge_alpha: float = 10.0

    def params_text(self) -> str:
        """Return a compact parameter summary for logs and reports."""
        return (
            f"gamma={self.rbf_gamma}, components={self.rbf_components}, "
            f"ridge_alpha={self.ridge_alpha}, threshold={self.threshold_multiplier}"
        )


RBF_CONFIG = RegressionConfig(
    label="RBF + Ridge",
    regression_kind="rbf",
    plot_filename="ges_rbf_top_anomalous_variables.png",
    threshold_multiplier=0.5,
    rbf_gamma=0.2,
    rbf_components=100,
    rbf_random_state=42,
    ridge_alpha=10.0,
)


# ================================================================
#                         DATA AND GRAPH LOADING
# ================================================================
def read_data(path: str) -> pd.DataFrame:
    """
    Load CSV or PKL data for TEP dataset.
    Standardizes feature columns (XMEAS_1..41, XMV_1..11).
    """
    path_str = str(path)
    if path_str.endswith(".pkl"):
        df = pd.read_pickle(path_str)
    else:
        df = pd.read_csv(path_str, delimiter=";")

    df.columns = [str(c).upper() for c in df.columns]
    feature_cols = [f"XMEAS_{i}" for i in range(1, 42)] + [f"XMV_{j}" for j in range(1, 12)]
    available_cols = [c for c in feature_cols if c in df.columns]
    if available_cols:
        df = df[available_cols]
    if "SAMPLE" in df.columns:
        df.set_index("SAMPLE", inplace=True)
    return df


def select_target_parent_candidates(data: np.ndarray, target_var: int, tau_max: int, top_k: int):
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
    return list(range(data.shape[1]))


def learn_causal_model(normal_csv_path: str, save_path: str):
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
    for freq in sorted_freq:
        if len([fr for fr in sorted_freq if fr < freq]) / len(sorted_freq) < 0.95:
            max_freq = freq
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
    return PREFIX / f"{TASK}_normal_ges.npz"


def ensure_canonical_ges_graph() -> Path:
    path = graph_cache_path()
    if not path.exists():
        learn_causal_model(str(PREFIX / NORMAL_FILE), str(path))
    return path


def load_canonical_graph():
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


def load_robotics_matrices(graph_info):
    normal_df = read_data(str(PREFIX / NORMAL_FILE))
    subsample = graph_info["subsample"]
    nonconst = graph_info["nonconst"]

    normal_train = np.nan_to_num(
        normal_df.values[: int(TRAINING_FRAC * len(normal_df))][::subsample, nonconst]
    )
    normal_full = np.nan_to_num(normal_df.values[::subsample, nonconst])

    attack_data = []
    for attack_file in ATTACK_FILES:
        attack_df = read_data(str(PREFIX / attack_file))
        attack_data.append((attack_file, np.nan_to_num(attack_df.values[::subsample, nonconst])))

    return normal_train, normal_full, attack_data


# ================================================================
#                   PARENT-LAG DESIGN MATRICES
# ================================================================
def target_parent_groups(causal_matrix: np.ndarray):
    indices = np.array(np.where(causal_matrix != 0))
    groups = {}
    for var in np.unique(indices[1, :]):
        var_indices = [indices[:, k] for k in range(indices.shape[1]) if indices[1, k] == var]
        var_indices.sort(key=lambda x: x[2])
        groups[int(var)] = var_indices[:3]
    return indices, groups


def build_parent_lag_xy(data: np.ndarray, var_indices: list, end_idx=None):
    max_delay = int(var_indices[-1][2])
    if end_idx is None:
        end_idx = len(data)

    stack = [
        data[max_delay - int(edge[2]) : end_idx - int(edge[2]), int(edge[0])]
        for edge in var_indices
    ]
    X = np.column_stack(stack)
    y = data[max_delay:end_idx, int(var_indices[0][1])]
    return X, y, max_delay


# ================================================================
#             REGRESSION IMPLEMENTATION: RBF
# ================================================================
class FittedRBFPipeline:
    def __init__(self, scaler: StandardScaler, sampler: RBFSampler):
        self.scaler = scaler
        self.sampler = sampler

    def transform(self, X: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(X)
        return self.sampler.transform(scaled)


def fit_single_target_model(X: np.ndarray, y: np.ndarray, config: RegressionConfig):
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X)
    sampler = RBFSampler(
        gamma=config.rbf_gamma,
        n_components=config.rbf_components,
        random_state=config.rbf_random_state,
    )
    X_rbf = sampler.fit_transform(scaled_X)
    model = Ridge(alpha=config.ridge_alpha, fit_intercept=True)
    model.fit(X_rbf, y)
    return model.coef_, FittedRBFPipeline(scaler, sampler)


def transform_single_target_inputs(X: np.ndarray, transformer: FittedRBFPipeline):
    return transformer.transform(X)


def fit_offline_coefficients(normal_train: np.ndarray, causal_matrix: np.ndarray, config: RegressionConfig):
    indices, groups = target_parent_groups(causal_matrix)
    fine_coeffs = {}
    transformers = {}

    for var, var_indices in groups.items():
        X, y, _ = build_parent_lag_xy(normal_train, var_indices)
        coeffs, transformer = fit_single_target_model(X, y, config)
        fine_coeffs[var] = coeffs
        transformers[var] = transformer

    return fine_coeffs, indices, groups, transformers


def compute_online_errors(
    data: np.ndarray,
    fine_coeffs: dict,
    causal_matrix: np.ndarray,
    indices: np.ndarray,
    groups: dict,
    transformers: dict,
    config: RegressionConfig,
):
    max_time = data.shape[0] - causal_matrix.shape[2]
    err = {}
    norm_agg = np.zeros((max_time, len(groups)))

    for t in range(max_time):
        end_idx = t + causal_matrix.shape[2]
        for i, (var, var_indices) in enumerate(groups.items()):
            X_window, y_window, _ = build_parent_lag_xy(data, var_indices, end_idx=end_idx)
            transformer = transformers[var]
            X_design = transform_single_target_inputs(X_window, transformer)

            model = Ridge(alpha=config.ridge_alpha, fit_intercept=True)
            model.fit(X_design, y_window)
            coeffs = model.coef_

            if var not in err:
                err[var] = np.zeros((max_time, len(coeffs)))
            err[var][t, :] = coeffs - fine_coeffs[var]
            norm_agg[t, i] = np.linalg.norm(err[var][t, :])

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


def score_and_predict(
    err_normal: dict,
    err_attack: dict,
    normal_data_len: int,
    normal: bool,
    threshold_multiplier: float,
    total_samples: int = None,
):
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
            values = np.abs(err_attack[var][start : start + max_time, j]) / thresh
            scores = np.maximum(scores, values)

    predictions = (scores > 1.0).astype(int)
    if total_samples is not None and total_samples > max_time:
        pad_len = total_samples - max_time
        predictions = np.concatenate([np.zeros(pad_len, dtype=int), predictions])
        scores = np.concatenate([np.zeros(pad_len, dtype=float), scores])

    return predictions, scores


def precision_recall_f1(tpos: list, fpos: list, fneg: list):
    tp_sum = np.sum(tpos)
    fp_sum = np.sum(fpos)
    fn_sum = np.sum(fneg)
    precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def accuracy_auc_metrics(normal_predictions, normal_scores, attack_predictions, attack_scores):
    y_true = np.concatenate(
        [
            np.zeros(len(normal_predictions), dtype=int),
            np.ones(sum(len(pred) for pred in attack_predictions), dtype=int),
        ]
    )
    y_pred = np.concatenate([normal_predictions] + attack_predictions)
    y_score = np.concatenate([normal_scores] + attack_scores)

    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) > 0 else 0.0
    try:
        auc_roc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc_roc = np.nan
    try:
        auc_pr = float(average_precision_score(y_true, y_score))
    except ValueError:
        auc_pr = np.nan
    return {"accuracy": accuracy, "auc_roc": auc_roc, "auc_pr": auc_pr}


# ================================================================
#                       FULL EXPERIMENT PIPELINE
# ================================================================
def run_linear_experiment(config: RegressionConfig, verbose: bool = True):
    started = time.time()
    graph_info = load_canonical_graph()
    normal_train, normal_full, attack_data = load_robotics_matrices(graph_info)

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
        fneg.append(attack_matrix.shape[0] - true_positive_count)
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
    extended_metrics = accuracy_auc_metrics(normal_predictions, normal_scores, attack_predictions, attack_scores)

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
        threshold=1.0, # <-- CRITICAL FIX: Scores are already normalized! Do n
        var_names=graph_info["var_names"][graph_info["nonconst"]].tolist(), #
        target_indices=list(groups.keys()), # <-- CRITICAL FIX: Use actual tar
        directed_edges=[],
        target_wise_residuals_by_file=target_wise_residuals_by_file,
        attack_labels_by_file=attack_labels_dict,
        dataset_name="Industrial (TEP)"
    )

    result = {
        "regression": config.label,
        "regression_kind": config.regression_kind,
        "parameters": config.params_text(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **extended_metrics,
        "runtime_s": time.time() - started,
    }
    
    if verbose:
        print("\n========== METRICS ==========")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1 Score:  {f1:.3f}")
        print(f"Accuracy:  {extended_metrics['accuracy']:.3f}")
        print(f"AUC-ROC:   {extended_metrics['auc_roc']:.3f}")

    return result


# ================================================================
#                              MAIN
# ================================================================
ACTIVE_CONFIG = RBF_CONFIG


def main():
    run_linear_experiment(ACTIVE_CONFIG, verbose=True)


if __name__ == "__main__":
    main()
