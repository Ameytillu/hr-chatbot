# src/core/config.py
import os
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()  # read .env for local runs

def _get_secret_or_none(key: str) -> Optional[str]:
    # Only try st.secrets if Streamlit is present; never crash if secrets.toml is missing
    try:
        import streamlit as st
        try:
            return st.secrets[key]  # may raise if secrets.toml not present
        except Exception:
            return None
    except Exception:
        return None

def getenv(key: str, default: Optional[str] = None) -> Optional[str]:
    # Prefer OS env (.env); fall back to Streamlit secrets; else default
    v = os.getenv(key)
    if v not in (None, ""):
        return v
    s = _get_secret_or_none(key)
    return s if s not in (None, "") else default

class Settings(BaseModel):
    EMBEDDINGS_PROVIDER: str = getenv("EMBEDDINGS_PROVIDER", "st")
    EMBEDDINGS_MODEL: str = getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
    ST_MODEL: str = getenv("ST_MODEL", "all-MiniLM-L6-v2")
    USE_DENSE: bool = str(getenv("USE_DENSE", "1")).strip() == "1"
    TOP_K: int = int(getenv("TOP_K", "6"))
    DATABASE_URL: Optional[str] = getenv("DATABASE_URL")

settings = Settings()
