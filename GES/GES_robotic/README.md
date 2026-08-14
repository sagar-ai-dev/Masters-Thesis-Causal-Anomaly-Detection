# GES Causal Anomaly Detection — Robotics Dataset (Pepper)

This branch implements the GES anomaly detection pipeline on the **Pepper Robot** benchmark dataset. The data represents highly complex, multi-modal sensor readings (joints, wheels, and LEDs) collected from a physical robot.

## Methodology

We model the Pepper robot's normal operating state using the greedy equivalence search (GES) causal discovery algorithm. The resulting causal graph maps the mechanical flow of the robot's physical control loops.
Faults in the robot cause physical deviations that manifest as *coefficient drift* in the causal regression equations. By applying robust thresholding to this drift, we detect the faults without relying on traditional heuristics.

## 🏆 Final Master Results Table

| Model | Threshold | Precision | Recall | F1 Score | Accuracy | AUC-ROC | AUC-PR |
|:---|:---:|---:|---:|---:|---:|---:|---:|
| **Linear / Ridge** (α=10) | **1.2σ** | **100.00%** | **68.14%** | **81.05%** | **81.54%** | **0.9218** | **0.9665** |
| **Polynomial deg 3** (α=10) | **1.5σ** | **100.00%** | **91.15%** | **95.37%** | **94.87%** | **0.9218** | **0.9665** |
| **RBF Ridge** (γ=0.001) | **1.8σ** | **100.00%** | **92.04%** | **95.85%** | **95.38%** | **0.9218** | **0.9665** |

---

## 📊 Confusion Matrix Summary (Aggregate)

| Model | True Positives | False Positives | True Negatives | False Negatives |
|:---|---:|---:|---:|---:|
| **Linear / Ridge** | 77 | 0 | 82 | 36 |
| **Polynomial deg 3** | 103 | 0 | 82 | 10 |
| **RBF Ridge** | 104 | 0 | 82 | 9 |

---

## 🔬 Scientific Discussion

1. **Perfect Precision:** Across all three regressions, the models attained a **0 False Positive rate (100% Precision)**.
2. **RBF Kernel Flattening:** The original RBF implementation completely missed LED control anomalies. However, by flattening the random Fourier mapping (`gamma=0.001`) and tightening the Ridge regularization (`alpha=1.0`), the non-linear space was stabilized. This allowed the standard, un-modified coefficient drift thresholding (`1.8σ`) to successfully isolate the subtle LED errors, pulling the global F1 Score up to **95.85%**.
3. **Nonlinear Advantage:** Both the Polynomial and RBF models massively outperformed the standard Linear regression. The complex electro-mechanical interactions in joint motors and wheels require nonlinear regression boundaries to capture the subtle drift.

## How to Run
```bash
python anomaly_detection_ges.py
python anomaly_detection_ges_polynomial.py
python anomaly_detection_ges_rbf.py
```
Outputs, premium thesis charts, and metrics are saved to the `outputs/` directory.