import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Loads .env into os.environ so LiteLLM can also find provider keys (GEMINI_API_KEY, etc.).
load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_model: str = os.getenv("LLM_MODEL", "gemini/gemini-2.5-flash")
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql://copilot:copilot@localhost:5432/copilot"
    )


settings = Settings()
