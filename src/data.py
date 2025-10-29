import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from pathlib import Path

DATA_FILE = Path("data.csv")

def load_csv_dataset(path=DATA_FILE):
    df = pd.read_csv(path)
    # cleanup columns from common Kaggle file
    df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
    # ensure diagnosis is present
    if 'diagnosis' not in df.columns:
        raise ValueError("Expected 'diagnosis' column in dataset.")
    return df

def prepare_data(df, test_size=0.2, random_state=42):
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis'].map({'M': 0, 'B': 1})  # Malignant=0, Benign=1
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns