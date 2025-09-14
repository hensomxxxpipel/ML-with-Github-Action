import streamlit as st
from pathlib import Path
import pandas as pd

base_dir = Path(__file__).parent.parent
svm_report_path = base_dir / "report_site" / "svm_report.txt"
nb_report_path = base_dir / "report_site" / "nb_report.txt"

st.set_page_config(
    page_title="Car Evaluation Model Report", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {font-family: 'Inter', sans-serif;}
.main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}
.block-container {
    padding: 2rem 1rem;
    max-width: 900px;
    margin: auto;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
    margin-top: 2rem;
    margin-bottom: 2rem;
}

/* Header */
.main-header {text-align: center;margin-bottom: 2rem;}
.main-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
.subtitle {
    font-size: 1.2rem;
    color: #64748b;
    font-weight: 400;
    margin-bottom: 2rem;
}

/* Model selector */
.model-selector {
    background: linear-gradient(135deg, #f8fafc, #e2e8f0);
    padding: 1.5rem;
    border-radius: 16px;
    margin: 2rem 0;
    border: 1px solid #e2e8f0;
    text-align: center;
}
.selector-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 1rem;
}

/* Selectbox styling */
div[data-baseweb="select"] > div {
    background: white !important;
    border-radius: 12px !important;
    border: 2px solid #e2e8f0 !important;
    font-weight: 500;
    transition: all 0.3s ease;
}
div[data-baseweb="select"] > div:hover {
    border-color: #667eea !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2) !important;
}

/* Warna teks selectbox */
div[data-baseweb="select"] span {
    color: #1e293b !important;
}
div[data-baseweb="select"] div {
    color: #1e293b !important;
}

/* Animations */
@keyframes fadeIn {
    from {opacity:0;}
    to {opacity:1;}
}
.fade-in {animation: fadeIn 0.8s ease forwards;}

.model-header {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 1.5rem;
    border-radius: 16px 16px 0 0;
    text-align: center;
    font-size: 1.5rem;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}
.accuracy-card {
    background: linear-gradient(135deg, #10b981, #059669);
    color: white;
    padding: 2rem;
    border-radius: 16px;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
    transform: scale(1);
    transition: transform 0.3s ease;
}
.accuracy-card:hover {transform: scale(1.02);}
.accuracy-value {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}
.accuracy-label {font-size: 1.2rem;font-weight: 500;opacity: 0.9;}

.metrics-table {
    margin-top: 1rem;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}
.dataframe th {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 1rem !important;
}
.dataframe td {
    padding: 0.75rem !important;
    border-bottom: 1px solid #f1f5f9 !important;
}
.dataframe tr:nth-child(even) {background-color: #f8fafc !important;}
.dataframe tr:hover {background-color: #e2e8f0 !important;}

.report-box {
    background: linear-gradient(135deg, #1e293b, #334155);
    color: #f1f5f9;
    padding: 1.5rem;
    border-radius: 12px;
    font-family: 'Fira Code', monospace;
    white-space: pre-wrap;
    overflow-x: auto;
    margin-top: 1rem;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    border: 1px solid #475569;
}
.footer {
    text-align: center;
    margin-top: 3rem;
    padding: 1rem;
    color: #64748b;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🚗 Car Evaluation Report</h1>
    <p class="subtitle">Analisis performa model Machine Learning untuk evaluasi mobil</p>
</div>
""", unsafe_allow_html=True)

# Model selector UI
st.markdown("""
<div class="model-selector">
    <div class="selector-title">🤖 Pilih Model Machine Learning</div>
</div>
""", unsafe_allow_html=True)

# Center selectbox
col1, col2, col3 = st.columns([1,2,1])
with col2:
    model_choice = st.selectbox(
        "",
        ["Pilih Model", "SVM", "Naive Bayes"],
        index=0,
        label_visibility="collapsed",
        key="select_model"
    )

# Show report
if model_choice != "Pilih Model":
    if model_choice == "SVM":
        report_path = svm_report_path
        header = "Support Vector Machine (SVM)"
        icon = "⚡"
    else:
        report_path = nb_report_path
        header = "Naive Bayes (CategoricalNB)"
        icon = "🧠"

    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            report = f.read()

        st.markdown(f"""
            <div class="report-container fade-in">
                <div class="model-header">
                    {icon} {header}
                </div>
            </div>
        """, unsafe_allow_html=True)

        accuracy_line = [line for line in report.splitlines() if "Akurasi" in line]
        acc = accuracy_line[0].split(":")[1].strip() if accuracy_line else "N/A"

        st.markdown(f"""
            <div class="accuracy-card fade-in">
                <div class="accuracy-value">{acc}</div>
                <div class="accuracy-label">Akurasi Model</div>
            </div>
        """, unsafe_allow_html=True)

        # Table metrics
        lines = report.splitlines()
        data_lines = [l for l in lines if l.strip() and ":" not in l and "Akurasi" not in l]
        parsed = []
        for l in data_lines[1:]:
            parts = l.split()
            if len(parts) >= 5:
                parsed.append([parts[0], parts[1], parts[2], parts[3], parts[4]])

        if parsed:
            df = pd.DataFrame(parsed, columns=["Class", "Precision", "Recall", "F1-score", "Support"])
            st.markdown("""
                <div class="fade-in">
                    <h3 style="text-align: center; color: #1e293b; margin: 2rem 0 1rem 0;">
                        📊 Metrics per Class
                    </h3>
                </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="metrics-table fade-in">', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="fade-in">
                    <h3 style="text-align: center; color: #1e293b; margin: 2rem 0 1rem 0;">
                        📄 Classification Report
                    </h3>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(f'<div class="report-box fade-in">{report}</div>', unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ Report {header} belum tersedia. Pastikan file report sudah di-generate.")

st.markdown("""
<div class="footer">
    <p>🔬 Machine Learning Model Evaluation Dashboard</p>
    <p>Dibuat dengan ❤️ menggunakan Streamlit</p>
</div>
""", unsafe_allow_html=True)
