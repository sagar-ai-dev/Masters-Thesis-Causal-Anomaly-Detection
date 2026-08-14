# Anomaly Detection Thesis Project

This repository contains the complete codebase and experiments for my Master's Thesis: **A multi-domain causal anomaly detection pipeline evaluating PCMCI, GES, and TS-CAM-UV across Robotics, Healthcare, and Industrial systems**.

## 🧠 Algorithms Evaluated
1. **PCMCI** (Peter and Clark Momentary Conditional Independence)
2. **GES** (Greedy Equivalence Search)
3. **TS-CAM-UV** (Time-Series Causal Additive Models with Unobserved Variables)

## 📊 Datasets & Domains
1. **Robotics**: PEPPER robotics anomaly-detection dataset.
2. **Healthcare**: PTB-XL ECG database / SCP statements (12-Lead, 20 Attack Patients).
3. **Industrial**: Tennessee Eastman Process (TEP) fault detection dataset (20 Fault Scenarios).

*(Note: Large dataset files like `*.RData` and models are ignored via `.gitignore` to comply with GitHub's file limits. Standard datasets are included).*

## 🔬 Downstream Regression Variants
All completed experiments are evaluated using three downstream regression variants:
- **Linear / OLS / Ridge**
- **Polynomial degree 3**
- **RBF + Ridge**

## 📈 Evaluation Metrics
The reported metrics are:
- Precision
- Recall
- F1-Score
- Accuracy
- AUC-ROC
- AUC-PR

---

## 🏆 Comprehensive Results 

### 1. Robotics Domain (PEPPER Dataset)

**GES Results**
| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | 
|---|---|---:|---:|---:|---:|---:|---:|
| GES | Linear / OLS | 1.000 | 0.920 | 0.959 | 0.954 | 0.922 | 0.967 | 
| GES | Polynomial degree 3 | 0.990 | 0.867 | 0.925 | 0.918 | 0.888 | 0.949 | 
| GES | RBF + Ridge | 1.000 | 0.894 | 0.944 | 0.938 | 0.902 | 0.957 | 

**PCMCI Results**
| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|
| PCMCI | Linear / OLS | 1.000 | 0.920 | 0.959 | 0.953 | 0.920 | 0.967 |
| PCMCI | Polynomial degree 3 | 0.989 | 0.832 | 0.904 | 0.896 | 0.845 | 0.932 | 
| PCMCI | RBF + Ridge | 0.727 | 0.894 | 0.802 | 0.740 | 0.854 | 0.927 | 

**CAM-UV Results**
| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|
| CAM-UV | Linear / OLS | TBD | TBD | TBD | TBD | TBD | TBD |
| CAM-UV | Polynomial degree 3 | TBD | TBD | TBD | TBD | TBD | TBD |
| CAM-UV | RBF + Ridge | TBD | TBD | TBD | TBD | TBD | TBD |

### 2. Healthcare Domain (PTB-XL ECG Dataset)

**GES Results**
| Model | Threshold | Precision | Recall | F1 Score | Accuracy | AUC-ROC | AUC-PR |
|:---|:---:|---:|---:|---:|---:|---:|---:|
| **Linear / Ridge** (α=10) | 3.0σ | 100.00% | 79.94% | 88.85% | 84.57% | 0.9980 | 0.9995 |
| **Polynomial deg 3** (α=10) | 3.0σ | 100.00% | 81.25% | 89.66% | 85.58% | 0.9980 | 0.9995 |
| **RBF Ridge** | 1.0σ | 100.00% | 99.80% | 99.90% | 99.85% | 0.9980 | 0.9995 |

**PCMCI Results**
| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|
| PCMCI | Linear / OLS | TBD | TBD | TBD | TBD | TBD | TBD |
| PCMCI | Polynomial degree 3 | TBD | TBD | TBD | TBD | TBD | TBD |
| PCMCI | RBF + Ridge | TBD | TBD | TBD | TBD | TBD | TBD |

**CAM-UV Results**
| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|
| CAM-UV | Linear / OLS | TBD | TBD | TBD | TBD | TBD | TBD |
| CAM-UV | Polynomial degree 3 | TBD | TBD | TBD | TBD | TBD | TBD |
| CAM-UV | RBF + Ridge | TBD | TBD | TBD | TBD | TBD | TBD |

### 3. Industrial Domain (TEP Dataset)

**GES Results**
| Model | Threshold | Precision | Recall | F1 Score | Accuracy | AUC-ROC | AUC-PR |
|:---|:---:|---:|---:|---:|---:|---:|---:|
| **Linear / Ridge** (α=10) | 0.5σ | 99.30% | 99.69% | 99.50% | 99.00% | 0.9576 | 0.9997 |
| **Polynomial deg 3** (α=10) | 0.5σ | 99.25% | 99.69% | 99.47% | 98.94% | 0.9965 | 1.0000 |
| **RBF Ridge** (γ=0.2) | 0.5σ | 99.24% | 99.69% | 99.46% | 98.93% | 0.8901 | 0.9991 |

**PCMCI Results**
| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|
| PCMCI | Linear / OLS | TBD | TBD | TBD | TBD | TBD | TBD |
| PCMCI | Polynomial degree 3 | TBD | TBD | TBD | TBD | TBD | TBD |
| PCMCI | RBF + Ridge | TBD | TBD | TBD | TBD | TBD | TBD |

**CAM-UV Results**
| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|
| CAM-UV | Linear / OLS | TBD | TBD | TBD | TBD | TBD | TBD |
| CAM-UV | Polynomial degree 3 | TBD | TBD | TBD | TBD | TBD | TBD |
| CAM-UV | RBF + Ridge | TBD | TBD | TBD | TBD | TBD | TBD |

---

## 🚀 How To Run

Install dependencies from the algorithm folder you want to run, then execute the relevant script.

```powershell
cd PCMCI\PCMCI_robotic
python .\anomaly_detection.py
python .\anomaly_detection_polynomial.py
python .\anomaly_detection_rbf.py
```

```powershell
cd GES\GES_healthcare
python .\anomaly_detection.py
python .\anomaly_detection_polynomial.py
python .\anomaly_detection_rbf.py
```
*(You can run CAM-UV and Industrial experiments similarly by changing the directory).*
