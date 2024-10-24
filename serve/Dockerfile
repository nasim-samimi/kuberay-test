FROM rayproject/ray:2.9.0

# Install any additional dependencies
RUN pip install tensorflow numpy python-multipart

# Copy your Python script into the container
COPY mobilenet.py /app/mobilenet.py

# Set the working directory
WORKDIR /app

# Start Ray Serve
CMD ["python", "-m", "ray", "serve", "start"]