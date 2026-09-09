# Use official lightweight Python image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    PORT=7860

# Set working directory
WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source, data, models, and frontend assets
COPY src/ src/
COPY data/ data/
COPY results/ results/
COPY scripts/ scripts/
COPY static/ static/
COPY main.py .
COPY .env.example .env

# Expose default Hugging Face Spaces port (7860)
EXPOSE 7860

# Run FastAPI backend with mounted static dashboard
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
