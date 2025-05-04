import logging
import time
from llama_cpp import Llama
from fastapi import HTTPException

from ..core.config import settings

logger = logging.getLogger(__name__)

# グローバル変数としてモデルインスタンスを保持
llm_instance: Llama | None = None

def load_llm_model():
    """アプリケーション起動時にLlamaモデルをロードする"""
    global llm_instance
    if llm_instance is not None:
        logger.info("LLM model is already loaded.")
        return

    logger.info(f"Loading LLM model from: {settings.llm_model_path}")
    logger.info(f"GPU Layers: {settings.n_gpu_layers}, Context: {settings.n_ctx}, Threads: {settings.n_threads}")
    start_time = time.time()
    try:
        llm_instance = Llama(
            model_path=settings.llm_model_path,
            n_gpu_layers=settings.n_gpu_layers,
            n_ctx=settings.n_ctx,
            n_threads=settings.n_threads,
            verbose=settings.llm_verbose,
        )
        load_time = time.time() - start_time
        logger.info(f"LLM model loaded successfully in {load_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Failed to load LLM model: {e}", exc_info=True)
        # アプリケーション起動時にモデルロード失敗した場合、
        # FastAPIが起動しないようにエラーを発生させるか、
        # llm_instance が None のままであることを確認する必要がある
        llm_instance = None # Ensure instance is None on failure
        raise RuntimeError(f"Failed to load LLM model: {e}")

async def get_llm_response(prompt: str, max_tokens: int = 512) -> str:
    """LLMにプロンプトを送信し、応答を取得する"""
    global llm_instance
    if llm_instance is None:
        logger.error("LLM model is not loaded.")
        # 本来は起動時にロードされているはずだが、念のため
        # ここでロードを試みるか、エラーを返すか検討
        # 今回はエラーを返す
        raise HTTPException(status_code=503, detail="LLM model is not available")

    logger.info(f"Generating LLM response for prompt (first 50 chars): {prompt[:50]}...")
    start_time = time.time()

    # TODO: 会話履歴を考慮したプロンプト構築 (src/cli.py を参考にする)
    # system_prompt = "システム: あなたはツンデレな性格の女の子..."
    # full_prompt = f"{system_prompt}ユーザー: {prompt}\nアシスタント: "
    full_prompt = f"ユーザー: {prompt}\nアシスタント: " # シンプルなプロンプト

    try:
        # ストリーミングではなく、一度にレスポンスを取得
        response = llm_instance(
            full_prompt,
            max_tokens=max_tokens,
            stop=["ユーザー:"], # 応答の停止条件
            echo=False # プロンプト自体は出力しない
        )

        inference_time = time.time() - start_time
        logger.info(f"LLM inference completed in {inference_time:.2f} seconds.")

        # response オブジェクトからテキストを抽出
        # llama-cpp-python のバージョンによって形式が違う可能性あり
        if isinstance(response, dict) and "choices" in response and len(response["choices"]) > 0:
            llm_response_text = response["choices"][0]["text"].strip()
        else:
            logger.warning(f"Unexpected LLM response format: {response}")
            llm_response_text = "" # or handle error appropriately

        logger.info(f"LLM response (first 50 chars): {llm_response_text[:50]}...")
        return llm_response_text

    except Exception as e:
        logger.error(f"Error during LLM inference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM inference failed: {e}") 