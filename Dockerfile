FROM python:3.12-slim

WORKDIR /app

COPY requirements_train.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install --no-cache-dir streamlit

EXPOSE 8501

# Command default saat container jalan (jalankan streamlit)
CMD ["streamlit", "run", "report_site/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
