FROM rayproject/ray:2.9.0

# Install any additional dependencies
RUN pip install numpy torch torchvision pandas

# RUN mkdir -p /tmp/ray && chmod -R 777 /tmp/ray
