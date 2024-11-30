from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    OPENAI_API_KEY: str
    MAX_HISTORY: int = 10
    CRISIS_THRESHOLD: float = 0.7
    
    class Config:
        env_file = ".env"