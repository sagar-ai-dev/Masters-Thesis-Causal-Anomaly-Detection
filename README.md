# Anomaly Detection Thesis Project

This repository contains the complete codebase and experiments for my Master's Thesis: a multi-domain causal anomaly detection pipeline evaluating PCMCI, GES, and TS-CAM-UV across robotics, healthcare, and industrial systems.

## Algorithms Evaluated

- **PCMCI**: Peter and Clark Momentary Conditional Independence
- **GES**: Greedy Equivalence Search
- **TS-CAM-UV**: Time-Series Causal Additive Models with Unobserved Variables

## Datasets & Domains

- **Robotics**: PEPPER robotics anomaly-detection dataset
- **Healthcare**: PTB-XL ECG database
- **Industrial**: Tennessee Eastman Process fault-detection dataset

Large dataset files are tracked with Git LFS where needed. Source code, run scripts, result summaries, and explainability plots are included in the repository.

## Downstream Regression Variants

Completed thesis experiments are evaluated using three downstream regression variants:

- Linear / OLS / Ridge
- Polynomial degree 3
- RBF + Ridge

## Evaluation Metrics

The reported metrics are Precision, Recall, F1 Score, Accuracy, AUC-ROC, and AUC-PR. AUC values are computed from continuous anomaly scores before thresholding; Precision, Recall, F1, and Accuracy are computed from thresholded detections.

## Verified Thesis Results

The tables below only include metrics that are backed by checked-in final result files or verified execution output.

### 1. Robotics Domain - PEPPER Dataset

**GES Results**

| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|
| GES | Linear / Ridge | 1.000 | 0.681 | 0.811 | 0.815 | 0.922 | 0.967 |
| GES | Polynomial degree 3 | 1.000 | 0.912 | 0.954 | 0.949 | 0.922 | 0.967 |
| GES | RBF + Ridge | 1.000 | 0.920 | 0.959 | 0.954 | 0.922 | 0.967 |

Source: `GES/GES_robotic/FINAL_RESULTS.md`. The linear model was also smoke-tested locally and reproduced F1 = 0.8105.

**PCMCI Results**

| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|
| PCMCI | Linear / OLS | 1.000 | 0.885 | 0.939 | 0.932 | 0.891 | 0.954 |
| PCMCI | Polynomial degree 3 | 1.000 | 0.867 | 0.929 | 0.922 | 0.915 | 0.964 |
| PCMCI | RBF + Ridge | 1.000 | 0.858 | 0.924 | 0.917 | 0.884 | 0.950 |

Source: `PCMCI/PCMCI_robotic/FINAL_RESULTS.md`.

**CAM-UV Status**

CAM-UV robotics code is included in `CAM-UV/CAM_UV_robotics`. Final validated CAM-UV robotics metrics are not listed here because the checked-in CAM-UV robotics result file still marks them as pending validation.

### 2. Healthcare Domain - PTB-XL ECG Dataset

**GES Results**

| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|
| GES | Linear / Ridge | 1.000 | 0.799 | 0.889 | 0.846 | 0.998 | 1.000 |
| GES | Polynomial degree 3 | 1.000 | 0.813 | 0.897 | 0.856 | 0.998 | 1.000 |
| GES | RBF + Ridge | 1.000 | 0.998 | 0.999 | 0.999 | 0.998 | 1.000 |

Source: `GES/GES_healthcare/FINAL_RESULTS.md`.

**PCMCI Results**

| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|
| PCMCI | Linear / OLS | 0.998 | 0.997 | 0.997 | 0.997 | 0.997 | 0.998 |
| PCMCI | Polynomial degree 3 | 0.999 | 0.997 | 0.998 | 0.997 | 0.997 | 0.998 |
| PCMCI | RBF + Ridge | 0.999 | 0.997 | 0.998 | 0.997 | 0.997 | 0.998 |

Source: `PCMCI/PCMCI_healthcare/README.md`.

**CAM-UV Status**

CAM-UV healthcare code is included in `CAM-UV/CAM-UV_healthcare`. Final validated CAM-UV healthcare metrics are not listed here because the checked-in CAM-UV healthcare result file still marks them as pending validation.

### 3. Industrial Domain - TEP Dataset

**GES Results**

| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|
| GES | Linear / Ridge | 0.993 | 0.997 | 0.995 | 0.990 | 0.958 | 1.000 |
| GES | Polynomial degree 3 | 0.993 | 0.997 | 0.995 | 0.989 | 0.997 | 1.000 |
| GES | RBF + Ridge | 0.992 | 0.997 | 0.995 | 0.989 | 0.890 | 0.999 |

Source: `GES/GES_industrial/FINAL_RESULTS.md`.

**PCMCI Status**

PCMCI industrial code is included in `PCMCI/PCMCI_industrial`. Final validated PCMCI industrial metrics are not listed here because the checked-in PCMCI industrial result file still marks them as pending validation.

**CAM-UV Status**

CAM-UV industrial code is included in `CAM-UV/CAM_UV_industrial`. Final validated CAM-UV industrial metrics are not listed here because the checked-in CAM-UV industrial result file still marks them as pending validation.

## Neural Baseline and Post-Hoc XAI

For paper-oriented comparison work, a PEPPER LSTM autoencoder baseline with SHAP post-hoc feature attribution is included in `Baselines/lstm_shap_baseline.py`. This is separate from the main thesis causal-detection experiments.

Current baseline coverage:

- Implemented: LSTM autoencoder on PEPPER with SHAP DeepExplainer output.
- Not yet implemented: TCN and Transformer baselines.

## How To Run

Each domain folder is self-contained with scripts, local data, and reporting helpers.

```powershell
cd GES\GES_robotic
python .\anomaly_detection_ges.py
python .\anomaly_detection_ges_polynomial.py
python .\anomaly_detection_ges_rbf.py
```

```powershell
cd PCMCI\PCMCI_robotic
python .\anomaly_detection.py
python .\anomaly_detection_polynomial.py
python .\anomaly_detection_rbf.py
```

```powershell
python .\Baselines\lstm_shap_baseline.py
```

All thesis explainability plots are saved to the corresponding experiment folder's `outputs/` directory. Baseline plots are saved to `Baselines/outputs/`.

## Deep Learning Baseline
An LSTM Autoencoder with SHAP explainability has been implemented in the Baselines/ directory to provide a neural network performance comparison against the causal anomaly detection pipeline.
