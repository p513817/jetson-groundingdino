FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
ARG DEBIAN_FRONTEND=noninteractive

# Remove opencv-python
RUN pip uninstall opencv-python && \
    rm -rf /usr/local/lib/python3.8/dist-packages/cv2 /usr/lib/python3.8/dist-packages/cv2

# Install GroundingDINO and related python package
WORKDIR /opt/program
RUN echo $CUDA_HOME
ENV PATH=/usr/local/cuda/bin:$PATH
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git
RUN cd GroundingDINO/ && pip install .
RUN pip install fastapi uvicorn[standard] pydantic numpy==1.21

# For model and tests
RUN mkdir weights ; cd weights ; wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth ; cd ..
COPY ./docker/gd_test.py gd_test.py

# Check and Warmup
RUN python3 gd_test.py

# Entry
COPY ./docker/entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh
ENTRYPOINT ["/opt/program/entrypoint.sh"]
WORKDIR /workspace
