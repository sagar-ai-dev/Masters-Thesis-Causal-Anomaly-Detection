# PCMCI Healthcare

Clean final PCMCI implementation for ECG healthcare anomaly detection.

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
| PCMCI | Polynomial degree 3 | 0.999 | 0.997 | 0.998 | 0.997 | 0.997 | 0.998 | Selected |
| PCMCI | RBF + Ridge | 0.999 | 0.997 | 0.998 | 0.997 | 0.997 | 0.998 | Comparison |
| PCMCI | Linear / OLS | 0.998 | 0.997 | 0.997 | 0.997 | 0.997 | 0.998 | Comparison |

Polynomial degree 3 is the selected PCMCI model because it has the strongest validated final F1 score in this folder.

AUC-ROC and AUC-PR are computed from continuous anomaly scores before thresholding. Precision, Recall, F1, and Accuracy are computed from thresholded anomaly detections.

Dataset files are included because no copied file is over 100 MB.
