from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    # LLM Settings
    # Use the path provided by the user as default, matching docker-compose volume mount
    llm_model_path: str = os.getenv("LLM_MODEL_PATH", "/app/models/Berghof-NSFW-7B.i1-IQ4_XS.gguf")
    n_gpu_layers: int = os.getenv("N_GPU_LAYERS", -1) # -1 means use all possible layers
    n_ctx: int = os.getenv("N_CTX", 2048)
    n_threads: int | None = os.getenv("N_THREADS", None) # None lets llama-cpp decide
    llm_verbose: bool = os.getenv("LLM_VERBOSE", False)

    # Voice Synthesis Settings (add model path for voice later if needed)
    default_voice_model_dir: str = os.getenv("DEFAULT_VOICE_MODEL_DIR", "/app/models/Anneli-nsfw")
    output_dir: str = "/app/output"

    # OpenAI API Key (Keep for potential future use or comparison)
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY", None)

    class Config:
        # If you create a .env file in the backend directory, its variables will override these defaults.
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings() 