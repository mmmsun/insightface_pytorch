FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
ARG PYTHON_VERSION=3.6
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libjpeg-dev \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*


RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include cython typing && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda90 && \
     /opt/conda/bin/conda install -c eumetsat expect && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH


RUN /opt/conda/bin/conda install pytorch torchvision ignite -c pytorch
RUN pip install --upgrade git+https://github.com/tensorpack/tensorpack.git


RUN pip install \
        "mxnet-cu90==1.2.1" \
        "tensorboardX==1.4" \
        "scikit-learn==0.20.0" \
        "PyYAML==3.13" \
        "lmdb==0.94"

WORKDIR /workspace

