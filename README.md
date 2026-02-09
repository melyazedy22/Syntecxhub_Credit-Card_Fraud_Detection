# Syntecxhub_Credit-Card_Fraud_Detection
project 2 - Week 3
# 💳 Credit Card Fraud Detection Project

This project focuses on detecting fraudulent credit card transactions using machine learning. The dataset is highly imbalanced, requiring specialized techniques to train effective models.

## 📌 Project Overview
- **Goal**: Classify transactions as **Fraud (1)** or **Legitimate (0)**.
- **Challenge**: Extreme class imbalance (very few fraud cases).
- **Solution**: Exploratory Data Analysis (EDA), Undersampling, and Tree-based models.

## 📂 Files
- `Credit-Card_Fraud_Detection.ipynb`: Main Jupyter Notebook containing EDA, preprocessing, modeling, and evaluation.
- `AIML Dataset.csv`: The dataset file (ensure it's in the same directory).

## ⚙️ Methodology

### 1. Data Preprocessing
- **Cleaning**: Checked for missing values and duplicates.
- **Feature Engineering**: One-hot encoding for categorical variables (`type`).
- **Sampling**: Applied **Random Undersampling** to balance the dataset.
    - *Note: SMOTE caused MemoryError on the full dataset, so Undersampling was chosen for efficiency.*

### 2. Models Implemented
- **Random Forest Classifier**: Robust to overfitting, handles non-linear data well.
- **XGBoost Classifier**: High-performance gradient boosting model, optimized for speed and accuracy.

### 3. Evaluation Metrics
Standard accuracy is misleading for imbalanced data. We focused on:
- **Precision & Recall**: To minimize false positives and false negatives.
- **ROC-AUC**: To measure the model's ability to distinguish classes.
- **AUPRC (Average Precision)**: Critical for rare event detection.
- **Confusion Matrix**: Visual breakdown of correct/incorrect predictions.

## 🚀 How to Run
1. Ensure `AIML Dataset.csv` is in the project folder.
2. Install dependencies:
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn xgboost
   ```
3. Run the notebook `Credit-Card_Fraud_Detection.ipynb` cell by cell.

## 📊 Results Summary
- Both Random Forest and XGBoost provide competitive results.
- Undersampling allowed for rapid training without memory issues.
- Detailed performance plots (ROC, PR Curves) are generated within the notebook.
