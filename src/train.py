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
clean_dir = base_dir / "clean_data" 

# ===== Cari file data terbaru =====
files = list(clean_dir.glob("clean_car_evaluation-ver-*.csv"))
if not files:
    raise FileNotFoundError("Tidak ada file clean_car_evaluation-ver-*.csv di folder clean_data")

# ambil versi terbesar
def get_version(f):
    match = re.search(r"ver-(\d+)", f.name)
    return int(match.group(1)) if match else 0

latest_file = max(files, key=get_version)
print(f"Menggunakan data terbaru: {latest_file.name}")

# baca dataset
df = pd.read_csv(latest_file)

## TRAIN
# pisahkan fitur dan target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# split data 70:30
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ===== Helper untuk cek versi =====
def get_next_version(save_dir: Path, prefix: str):
    save_dir.mkdir(parents=True, exist_ok=True)
    existing = list(save_dir.glob(f"{prefix}-ver-*.pkl")) + list(save_dir.glob(f"{prefix}-ver-*.txt"))
    if not existing:
        return 1
    versions = [int(re.search(r"ver-(\d+)", f.name).group(1)) for f in existing if re.search(r"ver-(\d+)", f.name)]
    return max(versions) + 1

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
svm_ver = get_next_version(svm_dir, "svm_model")
joblib.dump(svm_model, svm_dir / f"svm_model-ver-{svm_ver}.pkl")

with open(svm_dir / f"svm_report-ver-{svm_ver}.txt", "w", encoding="utf-8") as f:
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
nb_ver = get_next_version(nb_dir, "nb_model")
joblib.dump(nb_model, nb_dir / f"nb_model-ver-{nb_ver}.pkl")

with open(nb_dir / f"nb_report-ver-{nb_ver}.txt", "w", encoding="utf-8") as f:
    f.write(f"Akurasi: {nb_acc}\n\n")
    f.write(nb_report)


print("Kedua model dan report tersimpan")
