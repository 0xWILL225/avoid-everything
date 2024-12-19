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

COPY NGC-DL-CONTAINER-LICENSE /

# Required for non-glvnd setups.
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglvnd0 libglvnd0:i386 \
        libgl1 libgl1:i386 \
        libglx0 libglx0:i386 \
        libegl1 libegl1:i386 \
        libgles2 libgles2:i386 && \
    rm -rf /var/lib/apt/lists/*

COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libglvnd-dev libglvnd-dev:i386 \
        libgl1-mesa-dev libgl1-mesa-dev:i386 \
        libegl1-mesa-dev libegl1-mesa-dev:i386 \
        libgles2-mesa-dev libgles2-mesa-dev:i386 && \
    rm -rf /var/lib/apt/lists/*

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections \
  && apt update \
  && apt install tzdata \
  && ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime \
  && echo "America/Los_Angeles" > /etc/timezone \
  && dpkg-reconfigure -f noninteractive tzdata \
  && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
    git libsparsehash-dev build-essential wget curl ninja-build tmux \
    neovim software-properties-common python3-pip python3-dev python3-tk \
    freeglut3-dev \
  && apt-get clean && rm -rf /var/lib/apt

RUN wget https://ompl.kavrakilab.org/install-ompl-ubuntu.sh \
  && chmod u+x install-ompl-ubuntu.sh \
  && ./install-ompl-ubuntu.sh --github --python

RUN pip install --upgrade pip \
  && pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121 \
  && pip install \
    numpy ipython lightning==2.0.0 h5py wandb pybullet GitPython ikfast-pybind geometrout==0.1.3 torchmetrics matplotlib \
    tqdm urchin termcolor meshcat moviepy imageio splendor-render pyopengl robofin==0.0.4.4 slack_sdk yourdfpy ipdb torch_geometric \
  && yes | splendor_asset_installer \
  # Necessary because of version incompatibilities
  && pip install --upgrade fastapi \
  && pip install numba \
  && pip install git+https://github.com/fishbotics/atob.git \
  && pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
