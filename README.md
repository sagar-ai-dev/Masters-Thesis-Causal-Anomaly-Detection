# Anomaly Detection Thesis Project

This repository contains the complete codebase and experiments for my Master's Thesis: a multi-domain causal anomaly detection pipeline evaluating PCMCI, GES, and TS-CAM-UV across robotics, healthcare, and industrial systems.

## Algorithms Evaluated

1. **PCMCI** - Peter and Clark Momentary Conditional Independence
2. **GES** - Greedy Equivalence Search
3. **TS-CAM-UV** - Time-Series Causal Additive Models with Unobserved Variables

## Datasets & Domains

1. **Robotics**: PEPPER robotics anomaly-detection dataset.
2. **Healthcare**: PTB-XL ECG database / SCP statements.
3. **Industrial**: Tennessee Eastman Process (TEP) fault detection dataset.

Large dataset files are tracked with Git LFS where needed so the experiment folders can stay in the repository.

## Downstream Regression Variants

Completed experiments are evaluated using three downstream regression variants:

- Linear / OLS / Ridge
- Polynomial degree 3
- RBF + Ridge

## Evaluation Metrics

The reported metrics are Precision, Recall, F1 Score, Accuracy, AUC-ROC, and AUC-PR.

## Comprehensive Results

### 1. Robotics Domain - PEPPER Dataset

**GES Results**

| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|
| GES | Linear / Ridge | 1.000 | 0.681 | 0.811 | 0.815 | 0.922 | 0.967 |
| GES | Polynomial degree 3 | 1.000 | 0.912 | 0.954 | 0.949 | 0.922 | 0.967 |
| GES | RBF + Ridge | 1.000 | 0.920 | 0.959 | 0.954 | 0.922 | 0.967 |

**PCMCI Results**

| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|
| PCMCI | Linear / OLS | 1.000 | 0.885 | 0.939 | 0.932 | 0.891 | 0.954 |
| PCMCI | Polynomial degree 3 | 1.000 | 0.867 | 0.929 | 0.922 | 0.915 | 0.964 |
| PCMCI | RBF + Ridge | 1.000 | 0.858 | 0.924 | 0.917 | 0.884 | 0.950 |

**CAM-UV Results**

CAM-UV robotics code is included in `CAM-UV/CAM_UV_robotics`. Final validated metrics have not been published in the checked-in result files yet.

### 2. Healthcare Domain - PTB-XL ECG Dataset

**GES Results**

| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|
| GES | Linear / Ridge | 1.000 | 0.799 | 0.889 | 0.846 | 0.998 | 1.000 |
| GES | Polynomial degree 3 | 1.000 | 0.813 | 0.897 | 0.856 | 0.998 | 1.000 |
| GES | RBF + Ridge | 1.000 | 0.998 | 0.999 | 0.999 | 0.998 | 1.000 |

**PCMCI Results**

| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|
| PCMCI | Linear / OLS | 0.998 | 0.997 | 0.997 | 0.997 | 0.997 | 0.998 |
| PCMCI | Polynomial degree 3 | 0.999 | 0.997 | 0.998 | 0.997 | 0.997 | 0.998 |
| PCMCI | RBF + Ridge | 0.999 | 0.997 | 0.998 | 0.997 | 0.997 | 0.998 |

**CAM-UV Results**

CAM-UV healthcare code is included in `CAM-UV/CAM-UV_healthcare`. Final validated metrics have not been published in the checked-in result files yet.

### 3. Industrial Domain - TEP Dataset

**GES Results**

| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|
| GES | Linear / Ridge | 0.993 | 0.997 | 0.995 | 0.990 | 0.958 | 1.000 |
| GES | Polynomial degree 3 | 0.993 | 0.997 | 0.995 | 0.989 | 0.997 | 1.000 |
| GES | RBF + Ridge | 0.992 | 0.997 | 0.995 | 0.989 | 0.890 | 0.999 |

**PCMCI Results**

PCMCI industrial code is included in `PCMCI/PCMCI_industrial`. Final validated metrics have not been published in the checked-in result files yet.

**CAM-UV Results**

CAM-UV industrial code is included in `CAM-UV/CAM_UV_industrial`. Final validated metrics have not been published in the checked-in result files yet.

## How To Run

Install dependencies from the algorithm folder you want to run, then execute the relevant scripts.

```powershell
cd PCMCI\PCMCI_robotic
python .\anomaly_detection.py
python .\anomaly_detection_polynomial.py
python .\anomaly_detection_rbf.py
```

```powershell
cd GES\GES_healthcare
python .\anomaly_detection_ges.py
python .\anomaly_detection_ges_polynomial.py
python .\anomaly_detection_ges_rbf.py
```

```powershell
cd CAM-UV\CAM_UV_robotics
python .\anomaly_detection.py
python .\anomaly_detection_polynomial.py
python .\anomaly_detection_rbf.py
```
