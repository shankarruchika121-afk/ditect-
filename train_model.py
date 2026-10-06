"""
Train 4 models on the Pima Indians Diabetes dataset and save everything
the Streamlit app needs into ./artifacts

Run once:   python train_model.py
"""
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "diabetes.csv"
ART = BASE / "artifacts"
ART.mkdir(exist_ok=True)

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
# In this dataset a 0 in these columns really means "not measured"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

df = pd.read_csv(DATA_PATH)
df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)

X = df[FEATURES]
y = df["Outcome"]

# stratify keeps the diabetic / non-diabetic ratio the same in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


def make_pipe(clf):
    # Imputer + scaler live INSIDE the pipeline, so they are learned from the
    # training data only (no data leakage) and are applied automatically at
    # prediction time in the app.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


models = {
    "Logistic Regression": make_pipe(LogisticRegression(max_iter=1000)),
    "Decision Tree": make_pipe(DecisionTreeClassifier(max_depth=5, random_state=42)),
    "Random Forest": make_pipe(RandomForestClassifier(n_estimators=300, random_state=42)),
    "SVM": make_pipe(SVC(probability=True, random_state=42)),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics = {}

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    cv_auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()

    metrics[name] = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "cv_roc_auc": cv_auc,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "roc_curve": {"fpr": fpr.round(4).tolist(), "tpr": tpr.round(4).tolist()},
    }
    print(f"{name:20s} acc={metrics[name]['accuracy']:.3f}  "
          f"auc={metrics[name]['roc_auc']:.3f}  cv_auc={cv_auc:.3f}")

best = max(metrics, key=lambda n: metrics[n]["cv_roc_auc"])
print(f"\nBest model (by 5-fold CV ROC-AUC): {best}")

rf = models["Random Forest"].named_steps["clf"]
importance = dict(zip(FEATURES, rf.feature_importances_.round(4).tolist()))

joblib.dump(models, ART / "models.joblib")

# ---- .pkl files (plain pickle) ----
with open(ART / "models.pkl", "wb") as f:          # all 4 models in a dict
    pickle.dump(models, f)
with open(ART / "best_model.pkl", "wb") as f:      # only the best model
    pickle.dump(models[best], f)
with open(ART / "metrics.json", "w") as f:
    json.dump({"best_model": best, "models": metrics, "rf_importance": importance,
               "n_train": len(X_train), "n_test": len(X_test)}, f)

print("Saved artifacts/models.joblib, models.pkl, best_model.pkl and metrics.json")
