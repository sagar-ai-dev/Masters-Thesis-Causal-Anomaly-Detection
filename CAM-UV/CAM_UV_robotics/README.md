# CAM-UV Robotics

Clean final CAM-UV implementation for PEPPER robotic anomaly detection.

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
| CAM-UV | Linear / OLS | TBD | TBD | TBD | TBD | TBD | TBD | Comparison |
| CAM-UV | Polynomial degree 3 | TBD | TBD | TBD | TBD | TBD | TBD | Comparison |
| CAM-UV | RBF + Ridge | TBD | TBD | TBD | TBD | TBD | TBD | Comparison |

AUC-ROC and AUC-PR are computed from continuous anomaly scores before thresholding. Precision, Recall, F1, and Accuracy are computed from thresholded anomaly detections.

Dataset files are included because no copied file is over 100 MB.
