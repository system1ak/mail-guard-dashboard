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

# Create Streamlit config file
RUN echo "[theme]" > .streamlit/config.toml && \
    echo "primaryColor = \"#667eea\"" >> .streamlit/config.toml && \
    echo "backgroundColor = \"#ffffff\"" >> .streamlit/config.toml && \
    echo "secondaryBackgroundColor = \"#f0f2f6\"" >> .streamlit/config.toml && \
    echo "textColor = \"#262730\"" >> .streamlit/config.toml && \
    echo "" >> .streamlit/config.toml && \
    echo "[server]" >> .streamlit/config.toml && \
    echo "port = 8501" >> .streamlit/config.toml && \
    echo "headless = true" >> .streamlit/config.toml && \
    echo "runOnSave = true" >> .streamlit/config.toml

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
