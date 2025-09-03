import pandas as pd
from pathlib import Path
import re
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# path ke data mentah
base_dir = Path(__file__).parent.parent
clean_path = base_dir / "clean_data" / "clean_car_evaluation.csv"

# ===== Cari file data terbaru =====
if not clean_path.exists():
    raise FileNotFoundError("File clean_car_evaluation.csv tidak ditemukan di folder clean_data")


# baca dataset
df = pd.read_csv(clean_path)

## TRAIN
# pisahkan fitur dan target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# split data 70:30
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ========== Training SVM ==========
svm_model = SVC(kernel="rbf", random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

svm_acc = accuracy_score(y_test, y_pred_svm)
svm_report = classification_report(y_test, y_pred_svm, zero_division=0)

print("\n=== Support Vector Machine (SVM) ===")
print("Akurasi:", svm_acc)
print(svm_report)

# Simpan model dan report svm
svm_dir = base_dir / "model" / "svm"
joblib.dump(svm_model, svm_dir / "svm_model.pkl")

with open(svm_dir / "svm_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Akurasi: {svm_acc}\n\n")
    f.write(svm_report)

# ========== Training Naive Bayes ==========
nb_model = CategoricalNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

nb_acc = accuracy_score(y_test, y_pred_nb)
nb_report = classification_report(y_test, y_pred_nb, zero_division=0)

print("\n=== Naive Bayes (CategoricalNB) ===")
print("Akurasi:", nb_acc)
print(nb_report)

# Simpan model dan report svm
nb_dir = base_dir / "model" / "nb"
joblib.dump(nb_model, nb_dir / "nb_model.pkl")

with open(nb_dir / "nb_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Akurasi: {nb_acc}\n\n")
    f.write(nb_report)


print("Kedua model dan report berhasil diperbarui")


# Simpan report ke folder site agar bisa tampil di GitHub Pages
site_dir = base_dir / "report_site"
site_dir.mkdir(exist_ok=True)

with open(site_dir / "svm_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Akurasi: {svm_acc}\n\n")
    f.write(svm_report)

with open(site_dir / "nb_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Akurasi: {nb_acc}\n\n")
    f.write(nb_report)

print("Report tersimpan di report_site/ untuk web.")