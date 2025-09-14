FROM python:3.12-slim

WORKDIR /app

COPY ../requirements_streamlit.txt .
RUN pip install --no-cache-dir -r  requirements_streamlit.txt

# COPY . /app/report_site
COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "report_site/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
