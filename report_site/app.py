import streamlit as st
from pathlib import Path

base_dir = Path(__file__).parent.parent
svm_report_path = base_dir / "report_site" / "svm_report.txt"
nb_report_path = base_dir / "report_site" / "nb_report.txt"

st.set_page_config(page_title="Car Evaluation Model Report", layout="wide")

st.title("🚗 Car Evaluation Model Report")
st.markdown(
    """
    <style>
    .big-font {
        font-size:20px !important;
        font-weight:bold;
    }
    .report-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        font-family: monospace;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<p class="big-font">Laporan akurasi dan classification report dari model SVM dan Naive Bayes</p>',
    unsafe_allow_html=True,
)

# Pilihan model
model_choice = st.radio("Pilih model yang ingin ditampilkan:", ["SVM", "Naive Bayes"], horizontal=True)

# Load report
if model_choice == "SVM":
    report_path = svm_report_path
    header = "Support Vector Machine (SVM)"
else:
    report_path = nb_report_path
    header = "Naive Bayes (CategoricalNB)"

if report_path.exists():
    with open(report_path, "r", encoding="utf-8") as f:
        report = f.read()

    st.subheader(header)
    # tampilkan akurasi di atas
    accuracy_line = [line for line in report.splitlines() if "Akurasi" in line]
    if accuracy_line:
        acc = accuracy_line[0].split(":")[1].strip()
        st.metric(label="Akurasi", value=acc)

    st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)
else:
    st.warning(f"Report {header} belum tersedia.")
