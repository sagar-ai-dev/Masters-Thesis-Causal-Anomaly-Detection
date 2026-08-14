# PCMCI Robotic

Clean final PCMCI implementation for PEPPER robotic anomaly detection.

## Install Requirements

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python anomaly_detection.py
python anomaly_detection_polynomial.py
python anomaly_detection_rbf.py
```

## Final Results

| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PCMCI | Linear / OLS | 1.000 | 0.885 | 0.939 | 0.932 | 0.891 | 0.954 | Selected |
| PCMCI | Polynomial degree 3 | 1.000 | 0.867 | 0.929 | 0.922 | 0.915 | 0.964 | Comparison |
| PCMCI | RBF + Ridge | 1.000 | 0.858 | 0.924 | 0.917 | 0.884 | 0.950 | Comparison |

Linear / OLS is the selected PCMCI model because it has the strongest validated final F1 score in this folder.

AUC-ROC and AUC-PR are computed from continuous anomaly scores before thresholding. Precision, Recall, F1, and Accuracy are computed from thresholded anomaly detections.

Dataset files are included because no copied file is over 100 MB.