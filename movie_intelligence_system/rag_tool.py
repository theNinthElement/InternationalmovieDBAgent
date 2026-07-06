"""
RAG specialist agent: semantic search over the 100 detailed movie
descriptions extracted from RAG_movies.pdf (see ingestion.py), embedded into
a local Chroma vector store.

Design choice: wrapped as a small ReAct agent (rather than a fixed
retrieve-then-answer chain) so it can re-query with a refined phrase if the
first search misses, e.g. broadening "a man who fights a rules-based
underground club" -> "underground fight club Project Mayhem".
"""
import os

from langchain_core.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.prebuilt import create_react_agent

from config import get_llm, CHROMA_DIR

RAG_SYSTEM_PROMPT = """You are a movie-plot research assistant. You have a
semantic search tool over 100 detailed movie descriptions (each with a movie
name, genre, and plot description).

Rules:
1. Call the search tool with a natural-language description of the plot,
   theme, or genre you're looking for - not just a bare title if you don't
   already have a confirmed one.
2. If the top result is a clear match, answer using its movie name, genre,
   and a brief (1-3 sentence) description grounded in the retrieved text.
3. If nothing is a good match, say so plainly instead of guessing.
4. Never invent plot details that are not present in the retrieved text.
"""

_embeddings = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embeddings


def build_rag_agent(persist_dir: str = CHROMA_DIR):
    if not os.path.isdir(persist_dir):
        raise FileNotFoundError(
            f"No vector store found at '{persist_dir}'. Run `python ingestion.py` "
            "first to build it from RAG_movies.pdf."
        )
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=_get_embeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retriever_tool = create_retriever_tool(
        retriever,
        name="search_movie_descriptions",
        description=(
            "Search a database of 100 detailed movie plot descriptions and genres. "
            "Input should be a natural-language description of a plot, theme, "
            "character, or genre you're trying to identify or summarize."
        ),
    )
    llm = get_llm()
    return create_react_agent(llm, [retriever_tool], prompt=RAG_SYSTEM_PROMPT)
