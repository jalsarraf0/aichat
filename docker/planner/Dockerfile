FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /data
COPY app.py .
EXPOSE 8102
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8102"]
