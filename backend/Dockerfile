# --- Stage 1: Builder ---
# Base image with CUDA support for building
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 AS builder

WORKDIR /app

# Set environment variables
ENV TZ=Asia/Tokyo \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    build-essential \
    cmake \
    libsndfile1 \
    ffmpeg \
    git \
    git-lfs \
    wget \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --set python3 /usr/bin/python3.10

# Install pip
RUN wget https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py \
    && python3 /tmp/get-pip.py \
    && rm /tmp/get-pip.py \
    && python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies from requirements.txt
COPY requirements.txt /app/
RUN python3 -m pip install --no-cache-dir --timeout 600 -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu118 \
        --extra-index-url https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/wheels-cu118

# --- Stage 2: Final Image ---
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

WORKDIR /app

ENV TZ=Asia/Tokyo \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Install runtime system dependencies (including git, git-lfs, wget needed for initialize.py step)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    libsndfile1 \
    ffmpeg \
    git \
    git-lfs \
    wget \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --set python3 /usr/bin/python3.10

# Copy Python environment from builder stage
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Make sure pip and huggingface_hub are runnable (needed for transformers to download models)
RUN python3 -m pip install --no-cache-dir --upgrade pip huggingface_hub

# --- Download Style-Bert-VITS2 Assets in Final Stage using initialize.py ---
# Clone the repository temporarily, run initialize.py, then remove the repo
# Assets will be created in the WORKDIR (/app), e.g., /app/bert, /app/model_assets
RUN git lfs install \
    && echo "Cloning Style-Bert-VITS2 repository..." \
    && git clone --depth 1 https://github.com/litagin02/Style-Bert-VITS2.git /tmp/Style-Bert-VITS2 \
    && cd /tmp/Style-Bert-VITS2 \
    && echo "Running initialize.py (This might take a while)..." \
    && python3 initialize.py \
    && echo "initialize.py finished. Moving assets to /app..." \
    # Move downloaded assets to the correct location (/app)
    # Use || true to prevent build failure if move fails (e.g., target exists)
    && (mv bert /app/bert || true) \
    && (mv model_assets /app/model_assets || true) \
    && (mv dict_data /app/dict_data || true) \
    # Debug: Check if the specific tokenizer folders exist AFTER initialize.py and move
    && echo "Checking for tokenizer folders in /app/bert..." \
    && (ls -l /app/bert/bert-base-japanese-v3-owakata || echo "JP-Extra tokenizer folder NOT found in /app/bert") \
    && (ls -l /app/bert/bert-base-japanese-v3 || echo "Standard JP tokenizer folder NOT found in /app/bert") \
    && echo "Cleaning up /tmp directory..." \
    && cd /app \
    && rm -rf /tmp/Style-Bert-VITS2

# Copy application code
COPY ./app /app/app

# Create output directory
RUN mkdir -p /app/output

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 