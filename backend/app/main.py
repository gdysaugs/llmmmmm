from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import logging
from dotenv import load_dotenv

from .schemas import ChatRequest, ChatResponse
from .services.llm_service import get_llm_response, load_llm_model
from .services.voice_service import synthesize_voice, load_voice_model
from .core.config import settings # Import settings

# 環境変数をロード (主に開発用)
load_dotenv()

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORSミドルウェアの設定
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静的ファイル (生成された音声) の配信設定
# Use output_dir from settings
if not os.path.exists(settings.output_dir):
    os.makedirs(settings.output_dir)
app.mount("/audio", StaticFiles(directory=settings.output_dir), name="audio")

# --- アプリケーション起動時の処理 --- 
@app.on_event("startup")
async def startup_event():
    logger.info("Loading models on startup...")
    try:
        load_llm_model() # Load LLM model
    except RuntimeError as e:
        logger.error(f"Fatal error during LLM model loading: {e}. Application might not function correctly.")
        # ここでアプリを停止させるか、エラー状態を示すか

    try:
        load_voice_model(settings.default_voice_model_dir) # Load default voice model
    except RuntimeError as e:
        logger.error(f"Fatal error during Voice model loading: {e}. Application might not function correctly.")
        # ここでアプリを停止させるか、エラー状態を示すか
    logger.info("Model loading process finished.")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    logger.info(f"Received chat request: {request.text}")
    try:
        # 1. LLMで応答を生成
        llm_response_text = await get_llm_response(request.text)
        if not llm_response_text:
             logger.error("LLM failed to generate a response.")
             raise HTTPException(status_code=500, detail="LLM failed to generate response")
        logger.info(f"Generated LLM response: {llm_response_text[:100]}...") # Log more chars

        # 2. 音声合成を実行
        output_filename = f"response_{uuid.uuid4()}.wav"
        # Use output_dir from settings
        output_path = os.path.join(settings.output_dir, output_filename)

        # Use default voice model for now
        success = await synthesize_voice(llm_response_text, output_path, settings.default_voice_model_dir)

        if not success:
            logger.error("Voice synthesis failed.")
            # Return response without audio URL or raise error
            return ChatResponse(llm_response=llm_response_text, error="Voice synthesis failed")
            # raise HTTPException(status_code=500, detail="Voice synthesis failed")

        logger.info(f"Synthesized audio saved to: {output_path}")
        audio_url = f"/audio/{output_filename}"

        return ChatResponse(llm_response=llm_response_text, audio_url=audio_url)

    except HTTPException as http_exc: # Re-raise HTTPExceptions from services
        raise http_exc
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        # Return a generic error response
        return ChatResponse(llm_response="An error occurred.", error=str(e))
        # Or raise a generic 500 error
        # raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/")
async def root():
    return {"message": "Backend is running!"} 