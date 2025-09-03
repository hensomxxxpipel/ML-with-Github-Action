import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# path ke data mentah
base_dir = Path(__file__).parent.parent
data_dir = base_dir / "raw_data" / "car_evaluation.csv"

df = pd.read_csv(data_dir)

# Encoding semua kolom
encoder = LabelEncoder()
for col in df.columns:
    df[col] = encoder.fit_transform(df[col])
    

# path folder untuk data bersih
clean_dir = base_dir / "clean_data"

# tentukan nama file versi
version = 1
while True:
    clean_path = clean_dir / f"clean_car_evaluation-ver-{version}.csv"
    if not clean_path.exists():
        break
    version += 1

df.to_csv(clean_path, index=False, encoding="utf-8")

print("Preprocessing Berhasil Loh Ya")