FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopencv-dev \
    libsndfile1 \
    ffmpeg \
    pkg-config \
    libx11-dev \
    libopenblas-dev \
    liblapack-dev \
    libgtk-3-dev \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install packages
RUN python -m venv /venv && \
    . /venv/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy==1.24.3 && \
    pip install --no-cache-dir "dlib" && \
    pip install --no-cache-dir "opencv-python-headless" && \
    pip install --no-cache-dir tensorflow && \
    pip install --no-cache-dir fer && \
    pip install --no-cache-dir -r requirements.txt

ENV PATH="/venv/bin:$PATH"

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for storing captured frames (optional)
RUN mkdir -p /app/captured_frames

# Expose port
EXPOSE 8000

# Health check (optional but recommended)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]