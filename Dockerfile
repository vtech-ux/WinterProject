# Simple Dockerfile for Streamlit app
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
ENV STREAMLIT_SERVER_HEADLESS=true
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8501"]
