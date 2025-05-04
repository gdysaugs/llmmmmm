# LLM 連携 音声合成 Web アプリケーション 設計書

## 1. 目的

ユーザーからのテキスト入力に対して、LLM が応答を生成し、その応答テキストを Style-Bert-VITS2 を用いて音声合成し、Web インターフェース上で再生可能にするアプリケーションを構築する。

## 2. 全体構成

*   **Frontend:** React + Vite + Tailwind CSS (ユーザーインターフェース、テキスト入力、音声再生)
*   **Backend:** FastAPI (Python) (API エンドポイント、LLM 連携、StyleBERT 音声合成)
*   **Database:** PostgreSQL (今回はオプション、チャット履歴保存などに将来利用可能)
*   **Docker:** 各サービスをコンテナ化し、`docker-compose.yml` で管理する。

```mermaid
graph LR
    User[ユーザー] --> Browser[ブラウザ/Frontend]
    Browser --> BackendAPI[Backend (FastAPI)]
    BackendAPI --> LLM[LLM Service]
    BackendAPI --> StyleBERT[StyleBERT Service (同一コンテナ内)]
    StyleBERT -- GPU --> GPU[NVIDIA GPU]
    BackendAPI --> DB[(Database/PostgreSQL)]
```

## 3. フォルダ構成 (プロジェクトルート `/home/adama/projects/Stylebert`)

```
Stylebert/
├── frontend/           # React フロントエンド
│   ├── public/
│   ├── src/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── Dockerfile     # 本番用イメージビルド
│   └── Dockerfile.dev # 開発用 (ホットリロード対応)
├── backend/            # FastAPI バックエンド
│   ├── app/            # アプリケーションコード
│   │   ├── __init__.py
│   │   ├── main.py      # FastAPI アプリ定義、ルーター設定
│   │   ├── schemas.py   # Pydantic スキーマ (APIリクエスト/レスポンス用)
│   │   ├── services/    # ビジネスロジック
│   │   │   ├── __init__.py
│   │   │   ├── llm_service.py  # LLM 連携処理
│   │   │   └── voice_service.py # 音声合成処理 (StyleBERTラッパー)
│   │   └── core/        # 設定、共通モジュール
│   │       └── config.py  # 環境変数読み込みなど
│   ├── models/          # 音声合成モデル配置用 (例: Anneli-nsfw/)
│   │   └── Anneli-nsfw/
│   │       ├── Anneli-nsfw_e300_s5100.safetensors
│   │       ├── config.json
│   │       └── style_vectors.npy
│   ├── output/          # 生成された音声ファイルの一時保存/配信元
│   ├── requirements.txt # Python 依存関係
│   ├── Dockerfile       # Python, FastAPI, StyleBERT, GPU 対応イメージビルド
│   ├── .env
│   └── .env.example
├── (削除予定) voice_synthesis/ # CLIテスト用ディレクトリ (最終的に削除)
├── docker-compose.yml  # 全体コンテナ管理 (Frontend, Backend, DB)
└── llm_voice_app_design.md # この設計書
```

## 4. Frontend コンテナ (`frontend`)

*   **役割:** ユーザーインターフェースを提供。テキスト入力欄、送信ボタン、LLM 応答表示エリア、音声再生ボタン/コントロール。
*   **技術スタック:** Node.js 18, React, Vite, Tailwind CSS, Axios (API 通信用)
*   **Dockerfile:**
    *   `Dockerfile.dev`: 開発用。Node.js 18 ベース、ソースコードマウント、`npm run dev` 実行 (ホットリロード)。
    *   `Dockerfile`: 本番用。マルチステージビルド。Node.js 18 でビルドし、静的ファイルを Nginx などの軽量サーバーで配信。
*   **docker-compose.yml 設定:**
    *   開発時: ビルドコンテキスト、`Dockerfile.dev` 指定、ポートフォワーディング (例: `5173:5173`)、ボリュームマウント (`./frontend:/app`)。
    *   本番時: ビルドコンテキスト、`Dockerfile` 指定、ポートフォワーディング (例: `80:80`)。

## 5. Backend コンテナ (`backend`)

*   **役割:** API エンドポイントを提供。LLM サービスと連携し、応答を取得。StyleBERT を使用して音声合成を実行し、音声ファイルを提供する。
*   **技術スタック:** Python 3.10+, FastAPI, Uvicorn, `style-bert-vits2`, Pydantic, python-dotenv, (LLM クライアントライブラリ)
*   **Dockerfile:**
    *   ベースイメージ: `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04` (GPU 利用のため)
    *   Python 3.10, pip, ffmpeg, libsndfile1, wget 等のシステムパッケージインストール (CLI テスト時の Dockerfile を参考に PPA 使用)。
    *   `requirements.txt` に基づく Python ライブラリインストール (PyTorch CUDA 11.8 対応版を含む)。
    *   `Style-Bert-VITS2` 関連アセットのダウンロード (`initialize.py` 実行など、CLI テスト時の知見を活用)。
    *   FastAPI アプリケーションコード (`./app`) をコンテナにコピー。
    *   `CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]` などで FastAPI サーバーを起動。
*   **docker-compose.yml 設定:**
    *   ビルドコンテキスト、`Dockerfile` 指定。
    *   ポートフォワーディング (例: `8000:8000`)。
    *   ボリュームマウント:
        *   `./backend/models:/app/models`: 音声モデルをマウント。
        *   `./backend/output:/app/output`: 生成された音声ファイルをホストと共有。
        *   `(開発時)` `./backend/app:/app/app`: ホットリロードのためにソースコードをマウント (Uvicorn の `--reload` オプションと併用)。
    *   GPU 設定 (`deploy` セクション または `environment` 変数)。
    *   `.env` ファイル読み込み (`env_file`)。

## 6. API 仕様 (`/chat` エンドポイント)

*   **Method:** POST
*   **Path:** `/chat`
*   **Request Body (JSON):**
    ```json
    {
      "text": "ユーザーが入力したテキスト"
    }
    ```
*   **Response Body (JSON):**
    ```json
    {
      "llm_response": "LLM が生成した応答テキスト",
      "audio_url": "/audio/response_timestamp.wav" // Backend が配信する音声ファイルへの相対 URL
    }
    ```
    または
    ```json
    {
      "error": "エラーメッセージ"
    }
    ```

## 7. LLM 連携 (`llm_service.py`)

*   具体的な使用 LLM (例: GPT-4, Claude, ローカル LLM) とその API クライアントライブラリを使用する。
*   API キー等の認証情報は `.env` ファイルで管理する。
*   ユーザー入力テキストを受け取り、LLM に送信し、応答テキストを返す関数を実装する。
*   エラーハンドリング (API エラー、タイムアウト等) を行う。

## 8. StyleBERT 連携 (`voice_service.py`)

*   CLI テストで作成した `infer_cli.py` のロジックをベースに、FastAPI から呼び出せる関数として再実装する。
*   `TTSModel` のインスタンスを適切に管理する (リクエストごとに生成するか、起動時にシングルトンとして保持するか検討)。
*   使用するモデル (`Anneli-nsfw` をデフォルトにするか、リクエストで指定可能にするか検討) を選択するロジック。
*   テキストを受け取り、指定されたモデルで音声合成を実行する。
*   生成された音声ファイルを `output/` ディレクトリに一意な名前 (例: タイムスタンプベース) で保存する。
*   保存した音声ファイルにアクセスするための URL パス (例: `/audio/filename.wav`) を返す。
*   FastAPI で静的ファイル配信の設定を行い (`/audio` パスで `output/` ディレクトリを配信)、Frontend からアクセスできるようにする。
*   エラーハンドリング (モデルロードエラー、合成エラー等) を行う。

## 9. 音声ファイルの扱い

*   Backend は生成した音声ファイルを `/app/output` (ホストの `backend/output` にマウント) に保存する。
*   Backend は `/audio` などのパスで `/app/output` ディレクトリを静的ファイルとして配信する設定を行う (FastAPI の `StaticFiles`)。
*   Frontend は Backend から受け取った `audio_url` (例: `/audio/response_12345.wav`) を `<audio>` タグの `src` に設定して再生する。
*   古い音声ファイルを定期的に削除する仕組みを検討する (オプション)。

## 10. 環境変数 (`backend/.env`, `backend/.env.example`)

*   LLM API キー (`OPENAI_API_KEY` など)
*   (必要であれば) StyleBERT モデルパスやデフォルト設定
*   (DB を使う場合) PostgreSQL 接続情報

```dotenv
# .env.example
# LLM Settings
# OPENAI_API_KEY=

# Voice Synthesis Settings
# DEFAULT_VOICE_MODEL=Anneli-nsfw

# Database Settings (Optional)
# POSTGRES_USER=user
# POSTGRES_PASSWORD=password
# POSTGRES_DB=mydb
# POSTGRES_HOST=database
# POSTGRES_PORT=5432
```

## 11. その他

*   Frontend と Backend 間の CORS (Cross-Origin Resource Sharing) 設定が必要になる。FastAPI 側で許可するオリジン (Frontend の URL) を設定する。
*   エラーハンドリングを各所で適切に行う。 