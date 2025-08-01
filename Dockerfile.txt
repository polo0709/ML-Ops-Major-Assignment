# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model
COPY src/ src/
COPY model.joblib model.joblib

# Default command to run when container starts
# This runs the prediction step to verify everything works
CMD ["python", "src/predict.py"]
