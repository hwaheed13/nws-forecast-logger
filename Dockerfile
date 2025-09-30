# Dockerfile
FROM python:3.12-slim

# System deps: libgomp = OpenMP runtime needed by scikit-learn wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# App setup
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code + model files
COPY . .

# Start the Flask app via gunicorn (Railway injects $PORT)
CMD ["sh", "-c", "gunicorn -w 2 -b 0.0.0.0:$PORT api:app"]

