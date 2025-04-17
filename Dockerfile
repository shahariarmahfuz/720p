# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
# Set the working directory in the container
WORKDIR /app

# Install ffmpeg - crucial for video processing
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Make port 8000 available to the world outside this container
# Render.com typically uses port 10000, but Gunicorn binds to 8000 here.
# Render will map its internal port to the external one automatically.
EXPOSE 5001

# Define environment variable for the Gunicorn worker count (optional, Render might set this)
ENV WORKERS=${WORKERS:-4}

# Run app.py when the container launches using Gunicorn
# Gunicorn is a production-ready WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "4", "app:app"]
