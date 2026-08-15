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

Final validated CAM-UV robotics metrics are pending. The implementation scripts and generated explainability plots are included, but this folder does not currently contain a locked final metrics table.

AUC-ROC and AUC-PR are computed from continuous anomaly scores before thresholding. Precision, Recall, F1, and Accuracy are computed from thresholded anomaly detections.

Large dataset files are tracked with Git LFS where needed.
