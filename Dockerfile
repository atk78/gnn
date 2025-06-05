FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y\
    build-essential \
    git \
    python3 \
    python3-dev \
    python3-pip  &&\
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip

RUN pip install uv

WORKDIR /home/app
COPY pyproject.toml /home/app
RUN uv sync
RUN . .venv/bin/activate

CMD ["/bin/bash"]
