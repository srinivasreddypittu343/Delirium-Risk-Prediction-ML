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
---

3. Dataset

The original dataset (delirium_ml.xlsx) contains >200 elderly surgical / hip-fracture
patients, with:

Demographics – Age, Height, BMI

Frailty – Frailty Index

Labs – haematocrit (Hct), albumin (Alb), creatinine (Cre), etc.

Peri-operative variables – duration of anaesthesia / surgery, ICU admission,
infusions, blood products, post-operative length of stay

Target label – Postop Delirium (0 = no, 1 = yes)

For this repository, data/delirium_ml_sample.xlsx contains a small, de-identified
sample with the same schema.
To reproduce all results, replace it with the full dataset if you are allowed to use it.

4. Methods
4.1 Pre-processing

Convert text-encoded numeric columns (e.g. LOS, Hct, Alb, Cre) to numeric

Median imputation for missing numeric values

Drop patient identifier column

Define X (features) and y (Postop Delirium)

Stratified 80/20 train–test split

StandardScaler for models that need scaling (LR, SVM, MLP)

4.2 Models

All models are implemented with scikit-learn:

Logistic Regression

Linear baseline, class_weight="balanced"

Random Forest

Ensemble of decision trees, used for feature importance and the interface

SVM (RBF)

Non-linear margin-based classifier on scaled features

Multi-Layer Perceptron (MLP)

Small feed-forward neural network with two hidden layers and early stopping

4.3 Evaluation

Main metric: ROC AUC (test set)

Additional metrics: precision, recall, F1-score, confusion matrix

Focus on recall for the delirium class (1) – missing high-risk patients is costly

Approximate test ROC AUC:

Model	ROC AUC (test)
Logistic Regression	~0.78
Random Forest	~0.60
SVM (RBF)	~0.69
MLP Neural Network	~0.80 (best)
5. Unsupervised Analysis: PCA & Clustering

To explore patient phenotypes:

PCA on scaled features

PC1 ≈ 14.2% variance

PC2 ≈ 8.6% variance

Plot patients in PC1–PC2 space, coloured by delirium outcome

Apply K-Means (k = 3) in PCA space

Resulting clusters (approx.):

Cluster sizes: 109, 30, 57 patients

Delirium rates: 6.4%, 33.3%, 12.3%

One cluster clearly shows a much higher delirium rate, suggesting a high-risk
patient phenotype.

6. Patient-Level Risk Interface

The delirium_pipeline.py script and notebook implement a simple interface:

Build a template patient from median feature values.

Override key fields via a Python dict (Age, Frailty, ICU, Hct, Alb, Cre, …).

Use the Random Forest to predict:

class (0/1)

probability of delirium

Map the probability to:

< 0.5 → LOW chance of post-operative delirium

≥ 0.5 → HIGH risk – consider closer monitoring

This turns a complex model into a single, intuitive output that clinicians can
understand at a glance.
