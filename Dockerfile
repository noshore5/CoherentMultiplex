# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first and install early (good for Docker cache)
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    g++ \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    libatlas-base-dev \
    libfftw3-dev \
    pkg-config \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code (config.py will be excluded by .dockerignore)
COPY . .

# Remove any config files that might have been copied
RUN rm -f config.py config_safe.py

# Create a runtime config that uses environment variables only
RUN echo "import os" > config.py && \
    echo "CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY', '')" >> config.py && \
    echo "DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'" >> config.py

# Expose the port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]

