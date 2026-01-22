# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Copy the models directory with explicit glob pattern
COPY models/*.pkl models/

# Copy .streamlit config
COPY .streamlit/ .streamlit/

# Expose port
EXPOSE 8080

# Run Streamlit on Cloud Run
CMD ["streamlit", "run", "app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false", \
     "--client.showErrorDetails=true"]
