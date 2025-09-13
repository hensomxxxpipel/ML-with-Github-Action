# FROM python:3.12-slim

# WORKDIR /app

# COPY ../requirements_streamlit.txt .
# RUN pip install --no-cache-dir -r  requirements_streamlit.txt

# COPY . /app/report_site

# EXPOSE 8501

# CMD ["streamlit", "run", "report_site/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

FROM python:3.12-slim

WORKDIR /app

# copy requirements dari root project
COPY ../requirements_streamlit.txt /app/requirements_streamlit.txt
RUN pip install --no-cache-dir -r /app/requirements_streamlit.txt

# copy semua isi report_site ke /app
COPY . /app/

EXPOSE 8501

# jalankan app.py yang ada di /app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
