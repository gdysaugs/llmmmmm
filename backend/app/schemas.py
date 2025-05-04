from pydantic import BaseModel

class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    llm_response: str
    audio_url: str | None = None
    error: str | None = None 