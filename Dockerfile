FROM rayproject/ray:2.9.0

RUN pip install numpy torch torchvision pandas

RUN sudo apt-get update && sudo apt-get install -y util-linux

COPY entrypoint.sh /entrypoint.sh
# RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
