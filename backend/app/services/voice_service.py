import logging
import os
import time
import glob
import torch
import soundfile as sf
# from style_bert_vits2.tts import TTS # 正しいインポート?
# from style_bert_vits2 import TTS # もしかして直下にある？
# from StyleTTS2.tts import TTS # これが正解のはず！
from style_bert_vits2.tts_model import TTSModel
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from fastapi import HTTPException

from ..core.config import settings

logger = logging.getLogger(__name__)

# グローバル変数としてモデルインスタンスとロード済みディレクトリを保持
# voice_model: TTS | None = None
voice_model: TTSModel | None = None # 型アノテーション修正
current_model_dir: str | None = None

# Helper function to find files, similar to infer_cli.py logic
def find_model_files(model_dir):
    safetensors_files = glob.glob(os.path.join(model_dir, "*.safetensors"))
    config_path = os.path.join(model_dir, "config.json")
    style_vec_path = os.path.join(model_dir, "style_vectors.npy")

    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors model found in {model_dir}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    # style_vec_path is optional for some operations but usually needed for inference
    if not os.path.exists(style_vec_path):
         logger.warning(f"style_vectors.npy not found in {model_dir}. Using default style if applicable.")
         style_vec_path = None # Allow proceeding without it, model might handle default

    return safetensors_files[0], config_path, style_vec_path

def load_voice_model(model_dir: str = settings.default_voice_model_dir):
    """アプリケーション起動時または必要時にStyleTTS2モデルをロードする"""
    global voice_model, current_model_dir

    if voice_model is not None and current_model_dir == model_dir:
        logger.info(f"Voice model for {model_dir} is already loaded.")
        return

    logger.info(f"Loading voice model from: {model_dir}")
    start_time = time.time()

    try:
        # --- Explicitly load JP-Extra BERT/Tokenizer before TTSModel --- 
        # Use the model identified from web search for JP-Extra
        bert_model_name = "ku-nlp/deberta-v2-large-japanese-char-wwm"
        logger.info(f"Loading JP-Extra BERT model and tokenizer: {bert_model_name}")
        # This will download the model from Hugging Face Hub if not cached
        bert_models.load_model(Languages.JP, bert_model_name)
        bert_models.load_tokenizer(Languages.JP, bert_model_name)
        logger.info("JP-Extra BERT model and tokenizer loaded.")
        # --------------------------------------------------------------

        model_path, config_path, style_vec_path = find_model_files(model_dir)

        # デバイス選択 (GPU優先)
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        logger.info(f"Using device: {device}")

        # TTS モデルの初期化
        # Note: lazy_init=False might be needed depending on how TTSModel handles loading
        # voice_model = TTS(model_path, config_path, device=device)
        voice_model = TTSModel(model_path, config_path, style_vec_path, device=device) # style_vec_path を追加！
        # Optionally load style vectors if provided and needed by the model explicitly
        # voice_model.load_style_vectors(style_vec_path) # Check if TTS class needs this

        current_model_dir = model_dir # Update loaded model directory
        load_time = time.time() - start_time
        logger.info(f"Voice model loaded successfully in {load_time:.2f} seconds.")

    except FileNotFoundError as e:
        logger.error(f"Failed to find model files: {e}", exc_info=True)
        voice_model = None
        current_model_dir = None
        raise RuntimeError(f"Failed to find voice model files: {e}")
    except Exception as e:
        logger.error(f"Failed to load voice model: {e}", exc_info=True)
        voice_model = None
        current_model_dir = None
        raise RuntimeError(f"Failed to load voice model: {e}")

async def synthesize_voice(text: str, output_path: str, model_dir: str = settings.default_voice_model_dir) -> bool:
    """テキストから音声を合成し、指定されたパスに保存する"""
    global voice_model, current_model_dir

    # Ensure the correct model is loaded
    if voice_model is None or current_model_dir != model_dir:
        try:
            logger.info(f"Switching/loading voice model to: {model_dir}")
            # This load function is synchronous, consider making it async if it becomes a bottleneck
            load_voice_model(model_dir)
        except RuntimeError as e:
             logger.error(f"Failed to load required voice model {model_dir}: {e}")
             raise HTTPException(status_code=503, detail=f"Voice model {model_dir} is not available")

    if voice_model is None: # Double check after loading attempt
         logger.error("Voice model is not loaded after attempt.")
         raise HTTPException(status_code=503, detail="Voice model is not available")


    logger.info(f"Synthesizing voice for text (first 50 chars): {text[:50]}...")
    start_time = time.time()

    try:
        # 推論の実行 (infer_cli.py で見つかった問題を考慮)
        # compute_type はデバイスに基づいて決定 (GPUならfp16、CPUならfp32)
        compute_type = "fp16" if voice_model.device == "cuda" else "fp32"
        logger.info(f"Using compute type: {compute_type}")

        # --- Handle potential return tuple order issue --- 
        # It seems model.infer returns (sample_rate, audio_data)
        sr, wav = voice_model.infer(text=text) # Corrected order!
        # If it returns (sr, wav), swap them:
        # wav, sr = voice_model.infer(text=text) # Original incorrect assumption
        # -------------------------------------------------

        synthesis_time = time.time() - start_time
        logger.info(f"Voice synthesis completed in {synthesis_time:.2f} seconds.")

        # --- Ensure wav data is in the correct shape for soundfile --- 
        # soundfile expects [samples, channels], but model might return 1D array
        if wav.ndim == 1:
            logger.debug("Reshaping 1D wav data to 2D for soundfile.")
            wav = wav.reshape(-1, 1) # Reshape to (n_samples, 1)
        # ------------------------------------------------------------

        # 音声データをファイルに保存
        sf.write(output_path, wav, sr)
        logger.info(f"Synthesized audio saved to: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error during voice synthesis: {e}", exc_info=True)
        # Don't raise HTTPException here directly, let the main endpoint handle it
        # Consider returning False or raising a custom exception
        return False 