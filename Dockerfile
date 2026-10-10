FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY assets ./assets
COPY backend ./backend
COPY dashboard ./dashboard
COPY data/processed ./data/processed
# Cloud Run sets $PORT; default 8080 for local `docker run`
ENV PORT=8080
CMD exec gunicorn app:server --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120 --preload
