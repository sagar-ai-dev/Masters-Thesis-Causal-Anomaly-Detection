# Anomaly Detection Thesis Project

This repository contains thesis experiments for causal anomaly detection.

## Completed Robotics Experiments

- `PCMCI/PCMCI_robotic`
- `GES/GES_robotic`
- `CAM-UV/CAM_UV_robotic`

## Dataset

Dataset: PEPPER robotics anomaly-detection dataset.

Dataset files are included because no copied file is over 100 MB.

## Downstream Regression Variants

Both algorithms are evaluated using three downstream regression variants:

- Linear / OLS
- Polynomial degree 3
- RBF + Ridge

## Metrics

The reported metrics are:

- Precision
- Recall
- F1
- Accuracy
- AUC-ROC
- AUC-PR

AUC-ROC and AUC-PR are computed from continuous anomaly scores before thresholding. Precision, Recall, F1, and Accuracy are computed from thresholded anomaly detections.

## GES Results

| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| GES | Linear / OLS | 1.000 | 0.920 | 0.959 | 0.954 | 0.922 | 0.967 | Selected |
| GES | Polynomial degree 3 | 0.990 | 0.867 | 0.925 | 0.918 | 0.888 | 0.949 | Comparison |
| GES | RBF + Ridge | 1.000 | 0.894 | 0.944 | 0.938 | 0.902 | 0.957 | Comparison |

## PCMCI Results

| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PCMCI | Linear / OLS | 1.000 | 0.920 | 0.959 | 0.953 | 0.920 | 0.967 | Selected |
| PCMCI | Polynomial degree 3 | 0.989 | 0.832 | 0.904 | 0.896 | 0.845 | 0.932 | Comparison |
| PCMCI | RBF + Ridge | 0.727 | 0.894 | 0.802 | 0.740 | 0.854 | 0.927 | Comparison |

## How To Run

Install dependencies from the algorithm folder you want to run, then execute the relevant script.

```powershell
cd PCMCI\PCMCI_robotic
python .\anomaly_detection.py
python .\anomaly_detection_polynomial.py
python .\anomaly_detection_rbf.py
```

```powershell
cd GES\GES_robotic
python .\anomaly_detection_ges.py
python .\anomaly_detection_ges_polynomial.py
python .\anomaly_detection_ges_rbf.py
```

```powershell
cd CAM-UV\CAM_UV_robotic
python .\anomaly_detection.py
python .\anomaly_detection_polynomial.py
python .\anomaly_detection_rbf.py
```
