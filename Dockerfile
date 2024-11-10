FROM rayproject/ray:2.9.0

# Install any additional dependencies
RUN pip install numpy torch torchvision pandas

RUN sudo apt-get update && sudo apt-get install -y util-linux

# Copy a custom entrypoint script (explained below)
# COPY entrypoint.sh /entrypoint.sh
# RUN chmod +x /entrypoint.sh

# Override the default entrypoint to use the custom script
ENTRYPOINT ["/entrypoint.sh"]
