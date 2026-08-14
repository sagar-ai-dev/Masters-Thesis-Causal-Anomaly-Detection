# PCMCI Robotics (Pepper Robot) Benchmark Results

## 🏆 EXHAUSTIVE GLOBAL METRICS SUMMARY

| Metric | Linear PCMCI (OLS) | Polynomial ($d=3, \text{Expanding Window}, N<15$) | RBF Kernel ($\gamma=0.001, \text{Expanding Window}, N<15$) |
| :--- | :---: | :---: | :---: |
| **Total True Positives (TP)** | **100** | **98** | **97** |
| **Total False Positives (FP)** | **0** | **0** | **0** |
| **Total True Negatives (TN)** | **79** | **79** | **79** |
| **Total False Negatives (FN)** | **13** | **15** | **16** |
| **Overall Accuracy** | **0.9323** | **0.9219** | **0.9167** |
| **Overall Precision** | **1.0000** | **1.0000** | **1.0000** |
| **Overall Recall** | **0.8850** | **0.8673** | **0.8584** |
| **Overall F1 Score** | **0.9390** | **0.9289** | **0.9238** |
| **Overall AUC-ROC** | **0.8907** | **0.9154** | **0.8842** |
| **Overall AUC-PR** | **0.9537** | **0.9641** | **0.9495** |

## 📝 Expanding Window & Startup Guard Thesis Defense Justification

"To balance long-term baseline memory retention with startup sample safety, an expanding window architecture with an initial sample guard ($N < 15$) was evaluated. While the expanding memory stabilizes long-term baseline parameter variance, the startup guard prevents early transient mathematical explosions on raw sensor streams, ensuring robust, reproducible coefficient tracking across physical robot fault evaluations."