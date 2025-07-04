FROM nvcr.io/nvidia/cuda:12.1.1-devel-ubuntu22.04
RUN dpkg --add-architecture i386 && \
    apt-get update && apt-get install -y --no-install-recommends \
        libxau6 libxau6:i386 \
        libxdmcp6 libxdmcp6:i386 \
        libxcb1 libxcb1:i386 \
        libxext6 libxext6:i386 \
        libx11-6 libx11-6:i386 && \
    rm -rf /var/lib/apt/lists/*

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
        ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
        ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compat32,utility

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

COPY .devcontainer/NGC-DL-CONTAINER-LICENSE /

# Required for non-glvnd setups.
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglvnd0 libglvnd0:i386 \
        libgl1 libgl1:i386 \
        libglx0 libglx0:i386 \
        libegl1 libegl1:i386 \
        libgles2 libgles2:i386 && \
    rm -rf /var/lib/apt/lists/*

COPY .devcontainer/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libglvnd-dev libglvnd-dev:i386 \
        libgl1-mesa-dev libgl1-mesa-dev:i386 \
        libegl1-mesa-dev libegl1-mesa-dev:i386 \
        libgles2-mesa-dev libgles2-mesa-dev:i386 && \
    rm -rf /var/lib/apt/lists/*

### Installing basic development tools
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections \
  && apt update \
  && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
    git libsparsehash-dev build-essential wget curl ninja-build tmux \
    neovim nano software-properties-common python3-pip python3-dev python3-tk \
    freeglut3-dev \
  && apt-get clean && rm -rf /var/lib/apt

### Installing CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.24.1/cmake-3.24.1-Linux-x86_64.sh \
  -q -O /tmp/cmake-install.sh \
  && chmod u+x /tmp/cmake-install.sh \
  && mkdir /opt/cmake-3.24.1 \
  && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.24.1 \
  && rm /tmp/cmake-install.sh \
  && ln -s /opt/cmake-3.24.1/bin/* /usr/local/bin


### Install specific version of OMPL, with python bindings
ENV OMPL_VERSION=1.6.0
RUN apt-get update \
    && apt-get install -y g++ pkg-config libboost-serialization-dev libboost-filesystem-dev libboost-system-dev libboost-program-options-dev libboost-test-dev libeigen3-dev libode-dev wget libyaml-cpp-dev \ 
    && apt-get install -y castxml libboost-python-dev libboost-numpy-dev python3-numpy pypy3 \ 
    && pip install pygccxml \
    && pip install pyplusplus
RUN wget https://github.com/ompl/ompl/archive/refs/tags/$OMPL_VERSION.tar.gz \
    && tar zxf $OMPL_VERSION.tar.gz \ 
    && cd ompl-$OMPL_VERSION \ 
    && mkdir -p build/Release && cd build/Release \
    && export CXX=g++ && export MAKEFLAGS="-j `nproc`" \ 
    && cmake ../.. -DOMPL_BUILD_PYBINDINGS=ON -DPYTHON_EXEC=/usr/bin/python3 \
    && make update_bindings \
    && make install

RUN pip install --upgrade pip \
  && pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121 \
  && pip install \
    numpy==1.24.4 ipython lightning==2.0.0 h5py wandb pybullet GitPython ikfast-pybind geometrout==0.1.3 torchmetrics matplotlib \
    tqdm urchin termcolor meshcat moviepy imageio splendor-render pyopengl slack_sdk yourdfpy ipdb torch_geometric \
  && yes | splendor_asset_installer \
  # Necessary because of version incompatibilities
  && pip install --upgrade fastapi \
  && pip install numba \
  && pip install git+https://github.com/fishbotics/atob.git \
  && pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.2+cu121.html

# Use submodule robofin instead of pip version
ENV PYTHONPATH="/workspace/robofin:${PYTHONPATH}"
