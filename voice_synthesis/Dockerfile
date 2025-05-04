# ステージ 1: ソースコード取得 (Git LFS 対応)
FROM ubuntu:latest AS builder

# Git LFS 実行に必要なパッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Git LFS の初期化
RUN git lfs install --system

# Style-Bert-VITS2 リポジトリのクローン (LFSファイルはまだポインタ)
WORKDIR /app
RUN git clone https://github.com/litagin02/Style-Bert-VITS2.git src

# LFSオブジェクト情報の取得
RUN cd src && git lfs fetch --all

# ポインタファイルを実際のLFSファイルに置き換え (ダウンロード実行)
RUN cd src && git lfs checkout

# --- ステージ 2: 実行環境 ---
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# 環境変数設定 (Python関連、文字コード)
ENV PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

# 必要なシステムパッケージのインストール (Python 3.10 from PPA)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3.10-distutils \
    ffmpeg \
    libsndfile1 \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# python のシンボリックリンク作成 (python3.10 をデフォルトの python に)
# これにより python コマンドで python3.10 が実行される
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# python3.10 用の pip を確実にインストールし、アップグレード
RUN python -m ensurepip --upgrade

# 作業ディレクトリ設定
WORKDIR /app

# ステージ 1 からソースコードをコピー
COPY --from=builder /app/src /app/src

# requirements.txt をコピーして依存関係をインストール
# src 内にあるはずの requirements.txt を使う
COPY --from=builder /app/src/requirements.txt /app/requirements.txt
# CUDA 11.8 に対応した PyTorch を先にインストール (python -m pip を使用)
RUN python -m pip install --no-cache-dir torch==2.1.0+cu118 torchaudio==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118
# その他の依存関係をインストール (python -m pip を使用)
RUN python -m pip install --no-cache-dir -r requirements.txt

# 必要なモデルファイル (BERTなど) やアセットをダウンロード
WORKDIR /app/src
RUN python initialize.py
WORKDIR /app

# 推論用スクリプトをコピー (自分で作成したもの)
COPY ./src/infer_cli.py /app/src/infer_cli.py

# コンテナを起動したままにするためのコマンド (デバッグ用にbashを起動するのもあり)
CMD ["tail", "-f", "/dev/null"]
# CMD ["bash"] # デバッグしやすいようにbashを起動 