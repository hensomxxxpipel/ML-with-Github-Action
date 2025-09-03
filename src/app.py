import streamlit as st
from pathlib import Path

# path ke folder model
base_dir = Path(__file__).parent.parent
svm_report_path = base_dir / "model" / "svm" / "svm_report.txt"
nb_report_path = base_dir / "model" / "nb" / "nb_report.txt"

st.title("Car Evaluation Model Report")
st.markdown("Tampilan report akurasi dan classification report dari model SVM dan Naive Bayes")

# Pilih model untuk ditampilkan
model_choice = st.selectbox("Pilih model:", ["SVM", "Naive Bayes"])

if model_choice == "SVM":
    if svm_report_path.exists():
        with open(svm_report_path, "r", encoding="utf-8") as f:
            report = f.read()
        st.subheader("Support Vector Machine (SVM)")
        st.text(report)
    else:
        st.warning("Report SVM belum tersedia.")
else:
    if nb_report_path.exists():
        with open(nb_report_path, "r", encoding="utf-8") as f:
            report = f.read()
        st.subheader("Naive Bayes (CategoricalNB)")
        st.text(report)
    else:
        st.warning("Report Naive Bayes belum tersedia.")
