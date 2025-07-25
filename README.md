# 📊 Bank Marketing Dataset - Machine Learning Analysis

This project uses the **Bank Marketing Dataset** to predict whether a client will subscribe to a term deposit using various machine learning models and data preprocessing techniques.

---

## 📁 Dataset Overview

- **Source:** UCI Machine Learning Repository  
- **File:** `bank-full.csv`  
- **Total Records:** 45,211  
- **Features:** 17 (both numerical and categorical)

---

## 🎯 Project Goal

Build, compare, and visualize classification models to predict the target variable:  
`y` → Whether the client subscribed to a term deposit (`yes` or `no`).

---

## 🔧 Preprocessing Steps

- Replaced `'unknown'` with `NaN`
- Dropped missing rows for simplicity
- Label encoding for categorical features
- Standardized numeric features using `StandardScaler`
- Split data into training and testing sets (70/30)

---

## 🧠 Machine Learning Models Used

| Model                   | Description                                |
|-------------------------|--------------------------------------------|
| Logistic Regression     | Baseline linear classifier                 |
| Decision Tree           | Simple tree-based decision model           |
| Random Forest           | Ensemble of decision trees (high accuracy) |
| K-Nearest Neighbors     | Distance-based classifier                  |
| Support Vector Machine  | Margin-based classifier                    |

---

## 📊 Evaluation Metrics

- **Accuracy**
- **Precision / Recall / F1-Score**
- **Confusion Matrix**
- **ROC Curve (AUC)**
- **Cross-Validation Score**

---

## 📉 Visualizations

### 🔲 Confusion Matrices
Each model has its own confusion matrix visualized using heatmaps.

![Confusion Matrix Example](images/confusion_matrix_rf.png)

---

### 📊 Model Accuracy Comparison

Bar plot to visualize and compare model accuracies.

![Accuracy Comparison](images/accuracy_comparison.png)

---

### 🌲 Feature Importance (Random Forest)

Random Forest feature importances highlight key contributors.

![Feature Importance](images/feature_importance_rf.png)

---

### 📈 ROC Curve (All Models)

AUC-ROC curves for visual comparison of all models.

![ROC Curve](images/roc_curve.png)

---

## 🧪 Running in Google Colab

1. Upload `bank-full.csv` using the Colab file uploader.
2. Copy-paste the provided Colab code into cells.
3. View the in-notebook outputs (accuracy, plots, confusion matrices, ROC curves).

---

## 📦 Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
