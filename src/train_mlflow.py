import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

import mlflow
import mlflow.sklearn

# Tracking URI MLflow
MLFLOW_TRACKING_URI = 'http://127.0.0.1:8080'
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Nama experiment
mlflow.set_experiment("Tugas ML")

# Path ke data bersih
base_dir = Path(__file__).parent.parent
clean_path = base_dir / "clean_data" / "clean_car_evaluation.csv"

if not clean_path.exists():
    raise FileNotFoundError("File clean_car_evaluation.csv tidak ditemukan di folder clean_data")

# baca dataset
df = pd.read_csv(clean_path)

# split fitur dan target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# split data 70:30
test_size = 0.3
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

def save_text_report(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def train_and_log(model, model_name: str):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model", model_name)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cls_report = classification_report(y_test, y_pred, zero_division=0)
        mlflow.log_metric("accuracy", float(acc))

        reports_dir = base_dir / "report_site"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_txt_path = reports_dir / f"{model_name}_classification_report.txt"
        save_text_report(f"Akurasi: {acc}\n\n{cls_report}", report_txt_path)
        mlflow.log_artifact(str(report_txt_path), artifact_path="reports")

        # Simpan model lokal
        model_dir = base_dir / "model" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        local_model_path = model_dir / f"{model_name}_model.pkl"
        joblib.dump(model, local_model_path)

        # Log model ke MLflow
        mlflow.sklearn.log_model(model, artifact_path=model_name)

        print(f"\n=== {model_name.upper()} ===")
        print("Akurasi:", acc)
        print(cls_report)

# Training SVM
svm = SVC(kernel="rbf", random_state=random_state)
train_and_log(svm, "svm")

# Training Naive Bayes
nb = CategoricalNB()
train_and_log(nb, "nb_categorical")

print("\nKedua model dan report berhasil diperbarui di MLflow")
