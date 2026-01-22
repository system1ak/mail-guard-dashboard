# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .

# Copy models directory
COPY models/ ./models/

# Verify models directory contents
RUN ls -la /app/models/ || echo "Warning: models directory issue"

# Create .streamlit directory for configuration
RUN mkdir -p .streamlit

# Create Streamlit config file with Cloud Run port compatibility
RUN echo "[theme]" > .streamlit/config.toml && \
    echo "primaryColor = \"#667eea\"" >> .streamlit/config.toml && \
    echo "backgroundColor = \"#ffffff\"" >> .streamlit/config.toml && \
    echo "secondaryBackgroundColor = \"#f0f2f6\"" >> .streamlit/config.toml && \
    echo "textColor = \"#262730\"" >> .streamlit/config.toml && \
    echo "" >> .streamlit/config.toml && \
    echo "[server]" >> .streamlit/config.toml && \
    echo "port = 8080" >> .streamlit/config.toml && \
    echo "headless = true" >> .streamlit/config.toml && \
    echo "runOnSave = true" >> .streamlit/config.toml && \
    echo "enableCORS = false" >> .streamlit/config.toml && \
    echo "maxUploadSize = 50" >> .streamlit/config.toml

# Expose port 8080 (Cloud Run standard)
EXPOSE 8080

# Set environment variable for Cloud Run
ENV PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8080/_stcore/health || exit 1

# Run Streamlit application with dynamic port from environment
CMD exec streamlit run app.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --logger.level=debug \
    --client.showErrorDetails=true
