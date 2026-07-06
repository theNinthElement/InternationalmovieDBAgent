
import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "haiku-4.5")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DB_PATH = os.getenv("MOVIE_DB_PATH", "data/SQL_Movies.db")
PDF_PATH = os.getenv("RAG_PDF_PATH", "data/RAG_movies.pdf")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_movie_db")


def get_llm(temperature: float = 0):
    """Factory so every agent (supervisor + 3 specialists) shares one
    consistent, swappable way of getting an LLM."""
    if LLM_PROVIDER == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=OPENROUTER_MODEL,
            temperature=temperature,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=OPENROUTER_BASE_URL,
        )
    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=OPENAI_MODEL, temperature=temperature)
    else:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=ANTHROPIC_MODEL, temperature=temperature)
 