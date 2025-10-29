# src/model.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
from pathlib import Path

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def train_default_model(X_train, y_train):
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    return model

def save_model(model, scaler, name="logistic_v1"):
    joblib.dump({"model": model, "scaler": scaler}, MODEL_DIR / f"{name}.joblib")
    return MODEL_DIR / f"{name}.joblib"

def load_model(path):
    obj = joblib.load(path)
    return obj["model"], obj["scaler"]

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    return acc, cm
