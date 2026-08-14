# Anomaly Detection Thesis Project

This repository contains the complete codebase and experiments for my Master's Thesis: **A multi-domain causal anomaly detection pipeline evaluating PCMCI, GES, and TS-CAM-UV across Robotics, Healthcare, and Industrial systems.**

## 🧠 Algorithms Evaluated
- **PCMCI** (Peter and Clark Momentary Conditional Independence)
- **GES** (Greedy Equivalence Search)
- **TS-CAM-UV** (Time-Series Causal Additive Models with Unobserved Variables)

## 📊 Datasets & Domains
- **Robotics**: PEPPER robotics anomaly-detection dataset.
- **Healthcare**: PTB-XL ECG database (12-Lead, 20 Attack Patients).
- **Industrial**: Tennessee Eastman Process (TEP) fault detection dataset (20 Fault Scenarios).

*(Note: Large dataset files like `*.RData` and models are safely tracked via Git LFS or ignored via `.gitignore` to comply with GitHub's file limits. Standard datasets and scripts are fully included).*

## 🔬 Downstream Regression Variants
All completed experiments are evaluated using three downstream regression variants to test the structural inductive bias of the models:
- Linear / OLS / Ridge
- Polynomial degree 3
- RBF + Ridge

## 📈 Evaluation Metrics
The reported metrics are:
- Precision & Recall
- F1-Score (Primary target metric)
- Accuracy
- AUC-ROC & AUC-PR

---

## 🏆 Comprehensive Results

### 1. Robotics Domain (PEPPER Dataset)

**GES Results**
| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|
| GES | Linear / Ridge | 1.000 | 0.920 | 0.959 | 0.954 | 0.922 | 0.967 |
| GES | Polynomial degree 3 | 0.990 | 0.867 | 0.925 | 0.918 | 0.888 | 0.949 |
| GES | RBF + Ridge | 1.000 | 0.894 | 0.944 | 0.938 | 0.902 | 0.957 |

**PCMCI Results**
| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|
| PCMCI | Linear / Ridge | 1.000 | 0.920 | 0.959 | 0.953 | 0.920 | 0.967 |
| PCMCI | Polynomial degree 3 | 0.989 | 0.832 | 0.904 | 0.896 | 0.845 | 0.932 |
| PCMCI | RBF + Ridge | 0.727 | 0.894 | 0.802 | 0.740 | 0.854 | 0.927 |

**CAM-UV Results**
| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|
| CAM-UV | Linear / Ridge | 1.000 | 0.920 | 0.958 | 0.953 | 0.920 | 0.967 |
| CAM-UV | Polynomial degree 3 | 1.000 | 0.745 | 0.854 | 0.842 | 0.850 | 0.900 |
| CAM-UV | RBF + Ridge | 1.000 | 0.598 | 0.748 | 0.730 | 0.800 | 0.880 |

### 2. Healthcare Domain (PTB-XL ECG Dataset)
*Domain Adaptation: Models applied Inverse Signal-to-Noise Ratio (ISNR) to detect biological signal depression (myocardial infarction) as opposed to mechanical energy spikes.*

**GES Results**
| Model | Threshold | Precision | Recall | F1 Score | Accuracy | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|
| Linear / Ridge | 3.0σ | 1.000 | 0.799 | 0.888 | 0.845 | 0.998 | 0.999 |
| Polynomial deg 3 | 3.0σ | 1.000 | 0.812 | 0.896 | 0.855 | 0.998 | 0.999 |
| RBF + Ridge | 1.0σ | 1.000 | 0.998 | 0.999 | 0.998 | 0.998 | 0.999 |

**PCMCI Results**
| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|
| PCMCI | Linear / Ridge | 0.998 | 0.997 | 0.997 | 0.997 | 0.997 | 0.998 |
| PCMCI | Polynomial degree 3 | 0.999 | 0.997 | 0.998 | 0.997 | 0.997 | 0.998 |
| PCMCI | RBF + Ridge | 0.999 | 0.997 | 0.998 | 0.997 | 0.997 | 0.998 |

**CAM-UV Results**
| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|
| CAM-UV | Linear / Ridge | 1.000 | 0.996 | 0.998 | 0.998 | 0.996 | 0.999 |
| CAM-UV | Polynomial degree 3 | 1.000 | 0.526 | 0.690 | 0.690 | 0.996 | 0.850 |
| CAM-UV | RBF + Ridge | 1.000 | 0.526 | 0.690 | 0.690 | 0.996 | 0.850 |

### 3. Industrial Domain (TEP Dataset)

**GES Results**
| Model | Threshold | Precision | Recall | F1 Score | Accuracy | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|
| Linear / Ridge | 0.5σ | 0.993 | 0.996 | 0.995 | 0.990 | 0.957 | 0.999 |
| Polynomial deg 3 | 0.5σ | 0.992 | 0.996 | 0.994 | 0.989 | 0.996 | 1.000 |
| RBF + Ridge | 0.5σ | 0.992 | 0.996 | 0.994 | 0.989 | 0.890 | 0.999 |

**PCMCI Results**
| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|
| PCMCI | Linear / Ridge | 0.994 | 0.732 | 0.843 | 0.729 | 0.950 | 0.990 |
| PCMCI | Polynomial degree 3 | 0.995 | 0.817 | 0.897 | 0.810 | 0.996 | 0.999 |
| PCMCI | RBF + Ridge | 0.993 | 0.862 | 0.923 | 0.860 | 0.990 | 0.998 |

**CAM-UV Results**
| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|
| CAM-UV | Linear / Ridge | 0.993 | 0.996 | 0.994 | 0.989 | 0.957 | 0.999 |
| CAM-UV | Polynomial degree 3 | 0.992 | 0.996 | 0.994 | 0.989 | 0.996 | 1.000 |
| CAM-UV | RBF + Ridge | 0.992 | 0.996 | 0.994 | 0.989 | 0.890 | 0.999 |

---

## 🚀 How To Run
Each domain folder is completely self-contained with its own datasets, global configurations, and reporting helpers.
Navigate to the algorithm and domain you wish to evaluate, and run the downstream regression variants sequentially:

```powershell
# Example: Running the GES Industrial Pipeline
cd GES\GES_industrial

python anomaly_detection_ges.py
python anomaly_detection_ges_polynomial.py
python anomaly_detection_ges_rbf.py
```
All exhaustive global metrics, per-class confusion matrices, and Explainable AI plots will be saved to the local `outputs/` directory inside that folder.
