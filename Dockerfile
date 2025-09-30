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

# Railway exposes $PORT; default to 8080 locally
ENV PORT=8080

# Start the Flask app via gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "api:app"]
