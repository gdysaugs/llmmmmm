# Style-Bert-VITS2 音声合成機能 設計書

## 1. 目的

Style-Bert-VITS2 と指定されたモデル (litagin/sbv2_chupa) を使用し、WSL2 上の Docker コンテナ内で GPU (CUDA 11.8) を利用して CLI から音声合成のテストを実行可能な環境を構築する。既存の環境とは分離し、新しいフォルダ構成で行う。

## 2. フォルダ構成

プロジェクトルート (`/home/adama/projects/Stylebert`) 直下に以下の構成で `voice_synthesis` フォルダを新規作成する。

```
Stylebert/
├── (既存のファイルやフォルダ...)
└── voice_synthesis/
    ├── Dockerfile
    ├── docker-compose.yml
    ├── requirements.txt  # Style-Bert-VITS2 からコピー、または生成
    ├── .env
    ├── .env.example
    ├── models/           # 学習済みモデル配置用
    │   └── chupa_1/      # モデルごとのフォルダ (手動配置)
    │       ├── model.safetensors  # (例)
    │       └── config.json      # (例)
    ├── output/           # 生成された音声ファイルの出力先
    └── src/              # Style-Bert-VITS2 のソースコード (Dockerfile内でClone)
        ├── (Style-Bert-VITS2のファイル群...)
        └── infer_cli.py  # CLI実行用スクリプト (元リポジトリに存在する場合)
```

## 3. Docker 環境構築 (`Dockerfile` & `docker-compose.yml`)

### 3.1. Dockerfile

マルチステージビルドを採用し、Git LFS を確実に処理する。

*   **ステージ 1 (ソースコード取得):**
    *   ベースイメージ: `ubuntu:latest` (または適切な軽量イメージ)
    *   `apt-get update && apt-get install -y git git-lfs ca-certificates --no-install-recommends`
    *   `git lfs install --system`
    *   `git clone https://github.com/litagin02/Style-Bert-VITS2.git /app/src` (リポジトリURLは確認)
    *   `cd /app/src && git lfs fetch --all`
    *   `cd /app/src && git lfs checkout`

*   **ステージ 2 (実行環境):**
    *   ベースイメージ: `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04` (CUDA 11.8 と cuDNN8 に対応)
    *   必要なシステムパッケージ (Python3, pip など) をインストール。
    *   Python のバージョンを指定 (例: 3.10)。
    *   作業ディレクトリ設定 (`WORKDIR /app`)。
    *   ステージ 1 から `/app/src` をコピー (`COPY --from=0 /app/src /app/src`)。
    *   `src` フォルダ内の `requirements.txt` をコピーし、`pip install -r requirements.txt` を実行。PyTorch は CUDA 11.8 対応バージョンを明示的に指定する可能性あり。
    *   必要なポートは公開しない (CLI実行のみのため)。
    *   `CMD ["tail", "-f", "/dev/null"]` などでコンテナを起動したままにする。

### 3.2. docker-compose.yml

*   `version: '3.8'` などバージョン指定。
*   `services:`
    *   `voice-synthesis:`
        *   `build: ./voice_synthesis`
        *   `container_name: voice-synthesis-container`
        *   `volumes:`
            *   `./voice_synthesis/models:/app/models` (モデルファイルをコンテナにマウント)
            *   `./voice_synthesis/output:/app/output` (出力ファイルをホストに永続化)
            *   `(オプション)` `./voice_synthesis/src:/app/src` (開発用にソースコードをマウントする場合。Dockerfileでのコピーと排他)
        *   `deploy:` (GPU 利用設定)
            *   `resources:`
                *   `reservations:`
                    *   `devices:`
                        *   `- driver: nvidia`
                        *   `  count: 1`
                        *   `  capabilities: [gpu]`
        *   `environment:` (必要に応じて .env ファイルから読み込み)
            *   `NVIDIA_VISIBLE_DEVICES=all`
            *   `NVIDIA_DRIVER_CAPABILITIES=compute,utility`
        *   `working_dir: /app`
        *   `tty: true` (コンテナ内でインタラクティブシェルを使うため)
        *   `stdin_open: true` (同上)

### 3.3. 依存関係 (`requirements.txt`)

*   `Style-Bert-VITS2` リポジトリ内の `requirements.txt` をベースにする。
*   PyTorch のバージョンは CUDA 11.8 に対応するもの (`torch==<version>+cu118` など) を確認・指定する。torchvision, torchaudio も同様。
*   その他、推論に必要なライブラリが含まれていることを確認する。

## 4. モデルの準備

1.  `voice_synthesis/models/chupa_1` フォルダを作成する。
2.  指定された Hugging Face リポジトリ ([https://huggingface.co/litagin/sbv2_chupa/tree/main/chupa_1](https://huggingface.co/litagin/sbv2_chupa/tree/main/chupa_1)) から、必要なモデルファイル (`*.safetensors`, `config.json` 等) をダウンロードし、作成したフォルダ内に配置する。
    *   注意: Hugging Face のアクセスには認証が必要な場合がある。

## 5. 実行手順

1.  WSL2 の Ubuntu ターミナルで `voice_synthesis` ディレクトリに移動する。
2.  `(初回)` Docker イメージをビルドし、コンテナをバックグラウンドで起動する:
    ```bash
    sudo docker-compose up -d --build
    ```
3.  コンテナに入る:
    ```bash
    sudo docker-compose exec voice-synthesis bash
    ```
4.  コンテナ内で、Style-Bert-VITS2 の CLI スクリプト (`infer_cli.py` または相当するもの) を実行する。引数はスクリプトの仕様に合わせて指定する。
    例:
    ```bash
    python src/infer_cli.py \
        --model_dir /app/models/chupa_1 \
        --config_path /app/models/chupa_1/config.json \
        --output_path /app/output/test.wav \
        --text "これはテスト用の音声合成です。" \
        # 他に必要な引数 (話者ID、スタイル等) があれば追加
    ```
5.  ホストの `voice_synthesis/output` フォルダに `test.wav` が生成されていることを確認する。
6.  コンテナから出る: `exit`
7.  コンテナを停止する: `sudo docker-compose down`

## 6. 環境変数 (`.env`, `.env.example`)

現時点では必須の環境変数は想定されていないが、将来的な拡張のために空のファイルを用意しておく。

`.env.example`:
```
# 例: 将来的にAPIキーなどが必要になった場合
# HUGGINGFACE_TOKEN=
```

`.env`:
```
# .env.example をコピーして実際の値を設定 (現状は空)
```

## 7. その他

*   既存の LLM 機能用コンテナとは `docker-compose.yml` 内で別サービスとして定義するため、ポート競合などの心配はない。
*   エラー発生時は、コンテナログ (`sudo docker-compose logs voice-synthesis`)、Dockerfile のビルドログ、`infer_cli.py` の実行時出力などを確認する。
*   WSL2 での NVIDIA Container Toolkit のセットアップが前提となる。未セットアップの場合は別途手順が必要。 