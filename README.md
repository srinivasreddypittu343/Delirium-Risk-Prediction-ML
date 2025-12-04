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
```

📊 3. Dataset

The project uses an anonymised delirium dataset of elderly surgical / hip-fracture patients.

Each row = one patient
Each column = a clinical feature

Main columns:

🧍‍♂️ Demographics

Age, Height, BMI

💪 Frailty

Frailty Index

🧪 Laboratory values

Hct (haematocrit)

Alb (albumin)

Cre (creatinine)

(plus other routine labs)

🏥 Peri-operative data

DuraAnes – duration of anaesthesia

DuraSurg – duration of surgery

ICU – ICU admission (0/1)

Infusion, FFP, Vasopressor, Postop LOS, etc.

🎯 Target label

Postop Delirium → 0 = no, 1 = yes

Delirium is the minority class → imbalanced classification

For sharing in this repo, data/delirium_ml_sample.xlsx contains a small, de-identified subset with the same schema.
If you have permission, you can swap in the full dataset locally.

🛠 4. Methods
4.1 Pre-processing pipeline

Key steps before modelling:

🔢 Convert text-encoded numeric columns (e.g. LOS, Hct, Alb, Cre) → proper numeric

🕳 Handle missing values with median imputation

🆔 Drop any ID columns (SN, ID, etc.) to avoid leakage

📦 Define:

X – all feature columns

y – Postop Delirium

✂️ Split data with stratified 80/20 train–test split

📏 Apply StandardScaler for models that require scaling (LR, SVM, MLP)

4.2 Models (all in scikit-learn)

Logistic Regression

Linear baseline, interpretable coefficients

Uses class_weight="balanced" to handle class imbalance

Random Forest

Ensemble of decision trees

Captures non-linear interactions

Provides feature importances (used for key risk factors and interface)

SVM (RBF kernel)

Margin-based classifier with a non-linear RBF kernel

Trained on scaled features

MLP Neural Network

Small feed-forward network (2 hidden layers, ReLU, early stopping)

Learns more complex non-linear relations

4.3 Evaluation metrics

⭐ Primary metric: ROC AUC on the test set

📉 Also report: precision, recall, F1-score, confusion matrix

🩺 Clinical focus: recall for delirium = 1
→ missing high-risk patients is more serious than a few extra false positives

Approximate test ROC AUC:

Model	ROC AUC (test)
Logistic Regression	~0.78
Random Forest	~0.60
SVM (RBF)	~0.69
MLP Neural Network	~0.80 (best)
🔍 5. Unsupervised Analysis – PCA & Clustering

To understand the structure of the patient cohort, the project also uses unsupervised methods:

5.1 PCA (Principal Component Analysis)

Apply PCA to the scaled features

PC1 explains ≈ 14.2% of variance

PC2 explains ≈ 8.6% of variance

Together ≈ 23% of total variance

Plot patients in PC1–PC2 space, coloured by delirium outcome
→ certain regions contain more delirium cases → potential high-risk zones

5.2 K-Means clustering (k = 3)

Run K-Means (k = 3) in the PCA space

Clusters (approximate values):

Cluster sizes: 109, 30, 57 patients

Delirium rates: 6.4%, 33.3%, 12.3%

Interpretation:

One cluster clearly has a much higher delirium rate

Acts as a high-risk phenotype – a subgroup of patients who are significantly more likely to develop delirium

This unsupervised view supports the supervised models by showing that high-risk groups emerge naturally from the data.

🧑‍⚕️ 6. Patient-Level Risk Interface

The repository also contains a small patient-level interface built around the trained Random Forest.

6.1 How it works

Build a template patient using the median value of each feature from the training data.

Accept a Python dict of overrides (e.g. Age, Frailty Index, ICU, Hct, Alb, Cre).

Construct a one-row DataFrame in the same column order as training.

Use the Random Forest to predict:

class label (0 / 1)

probability of delirium (0–1)

Convert the probability into a simple interpretation:

p < 0.5 → “LOW chance of post-operative delirium”

p ≥ 0.5 → “HIGH risk – consider closer monitoring”

6.2 Why it’s useful

Hides model complexity behind a single, intuitive output

Uses variables that are already present in hospital systems (age, frailty, labs, ICU)

Easy to extend into:

a Streamlit app

or an EHR-integrated risk widget in future work

This interface shows how a research model can be turned into something that is clinician-friendly and actionable, while still being fully transparent in the code.
---

