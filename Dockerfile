# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install only essential system dependencies for OpenCV and Flask
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies (FAST now without dlib/tensorflow)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create necessary directories
RUN mkdir -p dataset attendance

# Expose port 7860 (required by Hugging Face Spaces)
EXPOSE 7860

# Start the application using Gunicorn
CMD ["gunicorn", "login:app", "--bind", "0.0.0.0:7860", "--timeout", "120"]
