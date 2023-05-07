# Use the official Python 3.9 image as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements to the container
COPY ./requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code to the container
COPY . .

# Install pauliopt
RUN python setup.py install

# Execute all the unit tests in the ./tests folder
CMD ["python", "-m", "unittest", "discover", "-s", "./tests/", "-p", "test_*.py"]