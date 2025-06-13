import os
from functools import lru_cache
from dotenv import load_dotenv

@lru_cache(maxsize=1)
def load_env(dotenv_path: str = 'rag_chatbot/.env') -> None:
    """Load environment variables from a .env file once."""
    load_dotenv(dotenv_path=dotenv_path)


def get_env_var(key: str, default: str | None = None) -> str:
    """Get an environment variable value ensuring .env is loaded."""
    load_env()
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"Vari\u00e1vel de ambiente {key} n\u00e3o encontrada.")
    return value


def get_google_api_key() -> str:
    """Return the Google API key from the environment."""
    return get_env_var('GOOGLE_API_KEY')
