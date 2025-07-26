# Use official Python 3.10 image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy files to container
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port (optional, useful if you're running a web app)
EXPOSE 8000

# Command to run your app
CMD ["python", "main.py"]
