# Use an official lightweight Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and force console output to be unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory inside the container
WORKDIR /app

# Copy dependencies first to leverage Docker cache
COPY requirements.txt .

# Install libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Command to start the server
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]