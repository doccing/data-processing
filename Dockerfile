# Dockerfile for Ad Performance Aggregator
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files (app, tests, benchmarks)
COPY aggregator.py benchmark.py test_aggregator.py .
COPY test_data/ test_data/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Create output directory with proper permissions
# Make it world-writable to work with bind mounts from any host user
RUN mkdir -p /out && \
    chmod 777 /out && \
    chown appuser:appuser /out

# Switch to non-root user
USER appuser

# Set default entrypoint and command
ENTRYPOINT ["python", "aggregator.py"]
CMD ["--help"]
