ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /workspace

COPY flamingo /workspace/flamingo/
COPY train_face.py /workspace/train_face.py

RUN find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf; \