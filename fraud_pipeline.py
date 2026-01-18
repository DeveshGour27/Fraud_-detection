# train_pipeline.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from transformers import (
    ColumnDropper,
    SafeTypeEncoder,
    PowerTransformColumns,
    ScaleColumns
)

# =========================
# Load data
# =========================
df = pd.read_csv(
    r"D:\Machine Learning\new_fraud\Synthetic_Financial_datasets_log.csv"
)


df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
df = df.drop_duplicates().dropna()

# =========================
# Columns
# =========================
num_pt_cols = [
    'amount',
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest'
]

num_scale_cols = [
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest'
]

# =========================
# Split
# =========================
X = df.drop('isFraud', axis=1)
y = df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# Pipeline
# =========================
pipeline = ImbPipeline(steps=[
    ('drop_unused', ColumnDropper()),
    ('encode_type', SafeTypeEncoder()),
    ('power_transform', PowerTransformColumns(num_pt_cols)),
    ('scale', ScaleColumns(num_scale_cols)),
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    ))
])

# =========================
# Train & Save
# =========================
pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "fraud_pipeline.pkl")
print("Saved fraud_pipeline.pkl")
