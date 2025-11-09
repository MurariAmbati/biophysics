# Computational Cell - Reproducible Container
# Pinned versions for deterministic builds

FROM python:3.11-slim as base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies (pinned versions)
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e ".[all]"

# Copy source code
COPY cc/ ./cc/
COPY examples/ ./examples/
COPY tests/ ./tests/

# Set environment variables for reproducibility
ENV PYTHONHASHSEED=42
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

# Create data directory for mounted volumes
RUN mkdir /data

# Default command
CMD ["python", "-m", "cc.core.run", "--help"]

# Build command:
# docker build -t computational-cell:0.1.0 .
#
# Run command:
# docker run -v $(pwd)/data:/data computational-cell:0.1.0 \
#   python -m cc.core.run --config /data/config.yaml --output /data/output
