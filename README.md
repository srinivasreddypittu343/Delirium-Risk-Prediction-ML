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

## 📊 2. Dataset and Pre-Processing

The dataset (stored in delirium_ml.xlsx) contains data for a little over 200 elderly surgical or hip-fracture patients. Each row represents a single patient encounter, and each column corresponds to a clinical variable. Examples of features include demographics such as age, height, and body mass index; frailty information such as a frailty index; pre- and post-operative laboratory results such as haematocrit, albumin, and creatinine; and peri-operative variables such as duration of anaesthesia, duration of surgery, ICU admission, use of vasopressors, infusions, and blood products, as well as post-operative length of stay. The target variable is Postop Delirium, encoded as zero if the patient did not develop delirium and one if the patient did. Delirium cases are less frequent than non-delirium cases, so this is an imbalanced binary classification problem.

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

## 🛠 3. Methods
3.1 Pre-processing pipeline

Key steps before modelling:

🔢 Convert text-encoded numeric columns (e.g. LOS, Hct, Alb, Cre) → proper numeric

🕳 Handle missing values with median imputation

🆔 Drop any ID columns (SN, ID, etc.) to avoid leakage

📦 Define:

X – all feature columns

y – Postop Delirium

✂️ Split data with stratified 80/20 train–test split

📏 Apply StandardScaler for models that require scaling (LR, SVM, MLP)

3.2 Models (all in scikit-learn)

📌 Logistic Regression

**→** We used Logistic Regression as an interpretable baseline model to predict the probability of post-operative delirium.

**→** It was trained on the cleaned and standardized clinical features with class_weight="balanced" to handle the imbalanced target.

**→** This model achieved a strong ROC AUC and helped us understand which features increased or decreased delirium risk through its coefficients. 
<img width="453" height="742" alt="image" src="https://github.com/user-attachments/assets/8dcfeedd-8ed4-44b3-8e46-b3ebe6b66ecb" />

🌲 Random Forest

**→** Random Forest was used to capture non-linear relationships and interactions between clinical variables.

**→** It was trained on the same preprocessed dataset and evaluated on the 20% test set alongside the other models.

**→** Although its ROC AUC was lower, it provided useful feature importance scores that highlighted key risk factors like age, ICU admission, and abnormal lab values.
<img width="453" height="741" alt="image" src="https://github.com/user-attachments/assets/807b30ad-b09e-48fc-89d1-37767d9c2592" />

📉 SVM with RBF Kernel

**→** The SVM with an RBF kernel was used to learn non-linear decision boundaries between delirium and non-delirium patients.

**→** We trained it on standardized features to ensure all variables were on a similar scale.

**→** Its ROC AUC was moderate, and the confusion matrix showed that at the default threshold it struggled with the minority delirium class, which reflects the class imbalance.
<img width="450" height="737" alt="image" src="https://github.com/user-attachments/assets/c103146d-4c9a-4dc1-b041-462d418e389d" />

🧠 Multi-Layer Perceptron (MLP)

**→** The MLP was a small neural network with two hidden layers designed to model complex non-linear patterns in the data.

**→** It was trained on standardized features with early stopping and regularization to reduce overfitting.

**→** This model achieved the highest ROC AUC on the test set, showing the best overall ability to separate delirium and non-delirium patients, even though it is less interpretable than Logistic Regression.
<img width="456" height="743" alt="image" src="https://github.com/user-attachments/assets/6e364ba4-d3d5-4e8b-9143-f6801a29518c" />


3.3 Evaluation metrics

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
## 🔍 4. Unsupervised Analysis – PCA & Clustering

To understand the structure of the patient cohort, the project also uses unsupervised methods:

4.1 PCA (Principal Component Analysis)

Apply PCA to the scaled features

PC1 explains ≈ 14.2% of variance

PC2 explains ≈ 8.6% of variance

Together ≈ 23% of total variance

Plot patients in PC1–PC2 space, coloured by delirium outcome
→ certain regions contain more delirium cases → potential high-risk zones
<img width="578" height="480" alt="image" src="https://github.com/user-attachments/assets/035be79c-3536-43b1-b83b-25b5210776c9" />

4.2 K-Means clustering (k = 3)

Run K-Means (k = 3) in the PCA space

Clusters (approximate values):

Cluster sizes: 109, 30, 57 patients

Delirium rates: 6.4%, 33.3%, 12.3%

<img width="572" height="475" alt="image" src="https://github.com/user-attachments/assets/8af82463-38dc-4ba0-8d9d-1a8655c6f14d" />


Interpretation:

One cluster clearly has a much higher delirium rate

Acts as a high-risk phenotype – a subgroup of patients who are significantly more likely to develop delirium

This unsupervised view supports the supervised models by showing that high-risk groups emerge naturally from the data.

## 🧑‍⚕️ 5. Patient-Level Risk Interface

The repository also contains a small patient-level interface built around the trained Random Forest.

5.1 How it works

**→** Build a template patient using the median value of each feature from the training data.

**→** Accept a Python dict of overrides (e.g. Age, Frailty Index, ICU, Hct, Alb, Cre).

**→** Construct a one-row DataFrame in the same column order as training.

Use the Random Forest to predict:

class label (0 / 1), probability of delirium (0–1)

Convert the probability into a simple interpretation:

p < 0.5 → “LOW chance of post-operative delirium” and p ≥ 0.5 → “HIGH risk – consider closer monitoring”

5.2 Why it’s useful

Hides model complexity behind a single, intuitive output. Uses variables that are already present in hospital systems (age, frailty, labs, ICU)

Easy to extend into:

a Streamlit app or an EHR-integrated risk widget in future work

This interface shows how a research model can be turned into something that is clinician-friendly and actionable, while still being fully transparent in the code.
---

