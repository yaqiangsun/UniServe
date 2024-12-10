# Base image with Python3.12
FROM python:3.12

# Create a work directory
WORKDIR /code

# Configure requirements file
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install requirements
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy code
COPY . /code/app

# Run the application on port 8085
CMD ["python", "run.py"]