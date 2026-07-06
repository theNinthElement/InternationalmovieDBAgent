"""
Central configuration for the movie intelligence system.

Every agent gets its LLM from get_llm() so the whole system can be switched
between providers (Anthropic / OpenAI) with one environment variable instead
of hunting through multiple files.
"""
import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

DB_PATH = os.getenv("MOVIE_DB_PATH", "data/movies.db")
PDF_PATH = os.getenv("RAG_PDF_PATH", "data/RAG_movies.pdf")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_movie_db")


def get_llm(temperature: float = 0):
    """Factory so every agent (supervisor + 3 specialists) shares one
    consistent, swappable way of getting an LLM."""
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=OPENAI_MODEL, temperature=temperature)
    else:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=ANTHROPIC_MODEL, temperature=temperature)
