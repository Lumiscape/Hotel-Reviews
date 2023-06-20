# Use the official Python image as the base image
FROM python:3.10.6

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install -r requirements.txt
RUN pip install uvicorn

# Copy the entire project directory to the container
COPY . .

# Set the environment variable for the service account key file
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/service_account_key.json

# Run the FastAPI application using uvicorn
CMD ["uvicorn", "hotelapp.api.fast:app", "--host", "0.0.0.0", "--port", "8080"]
