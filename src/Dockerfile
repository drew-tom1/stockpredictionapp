# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /stockpredictionapp

# Copy the requirements file into the container
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8080

# Run app.py when the container launches
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
