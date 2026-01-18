# 🛡️ Fraud Detection using Machine Learning (Pipeline + SMOTE)

This project builds a **Fraud Detection Model** using the **Synthetic Financial Dataset**, featuring a robust preprocessing workflow and class imbalance handling using **SMOTE**. The trained pipeline is exported as a `.pkl` file for seamless integration into production environments like FastAPI, Flask, or Streamlit.

---

## 🚀 Features

- ✅ **Data Cleaning**: Automatically drops unused columns, removes duplicates, and handles missing values.
- ✅ **Custom Transformers**: Scikit-learn compatible classes for specialized feature engineering.
- ✅ **Preprocessing Pipeline**:
  - Label Encoding for categorical transaction `type`.
  - Power Transformation (Yeo-Johnson) to stabilize variance and minimize skewness.
  - Standard Scaling for consistent feature magnitude.
- ✅ **Imbalance Handling**: Integrated **SMOTE** (Synthetic Minority Over-sampling Technique) to address the rare occurrence of fraud.
- ✅ **Model**: High-performance **RandomForestClassifier**.
- ✅ **Portability**: Saves the entire end-to-end pipeline as `fraud_pipeline.pkl`.

---

## 📂 Dataset

**File:** `Synthetic_Financial_datasets_log.csv`  
**Target Column:** `isFraud` (0: Legitimate, 1: Fraud)

**Dropped Columns:**
- `nameOrig`: Unique ID of the sender.
- `nameDest`: Unique ID of the receiver.
- `isFlaggedFraud`: Inherited business logic flag.

---

## 🧠 Workflow Overview

1. **Data Ingestion**: Load dataset from CSV.
2. **Feature Selection**: Drop non-predictive ID columns.
3. **Data Quality**: Scrub duplicates and null values.
4. **Data Splitting**: Partition into Train and Test sets.
5. **Transformation**: Execute the preprocessing pipeline.
6. **Balancing**: Apply SMOTE **strictly on training data** to prevent data leakage.
7. **Training**: Fit the Random Forest model.
8. **Serialization**: Export the final model with `joblib`.

---

## 🛠️ Tech Stack

- **Python** 🐍
- **Pandas/NumPy**: Data manipulation.
- **Scikit-learn**: Machine learning framework.
- **Imbalanced-learn**: SMOTE implementation.
- **Joblib**: Model serialization.

---

## 📁 Project Structure

```bash
fraud-detection-ml/
│
├── fraud_pipeline_train.py    # Main training script
├── fraud_pipeline.pkl         # Saved model (generated after training)
├── README.md                  # Documentation
├── requirements.txt           # Dependency list
└── data/
    └── Synthetic_Financial_datasets_log.csv  # Dataset file
