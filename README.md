# 🧠 Delirium Risk Prediction using Classical Machine Learning

Predicting which older surgical patients are at risk of **post-operative delirium** using
routine clinical data and classical ML models.

> **Goal:** turn everyday hospital data (age, frailty, labs, ICU info) into a  
> simple **Low vs High delirium risk** flag that could support clinicians.

---

## 1. Project Overview

Post-operative delirium is an acute confusion state that often affects older adults after
surgery. It is strongly associated with longer length of stay, falls, complications, and
higher mortality – but some cases are preventable if high-risk patients are identified
early.

In this project I:

- Cleaned and preprocessed an anonymised delirium dataset (`delirium_ml.xlsx`)
- Trained and compared several **classical ML models**:
  - Logistic Regression  
  - Random Forest  
  - SVM (RBF kernel)  
  - Multi-Layer Perceptron (MLP)
- Explored patient phenotypes using **PCA + K-Means clustering**
- Built a **patient-level interface** that turns model output into an intuitive  
  **Low vs High risk** interpretation.

This repository is meant for learning and demonstration purposes in a **master’s-level
machine learning course**, not for clinical use.

---

## 2. Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── delirium_ml_sample.xlsx      # small sample of the original dataset
├── notebooks/
│   └── Delirium_Prediction.ipynb    # full exploratory notebook
└── src/
    └── delirium_pipeline.py         # clean, runnable script
