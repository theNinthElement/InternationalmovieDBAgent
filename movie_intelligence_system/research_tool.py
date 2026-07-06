from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent

from config import get_llm

RESEARCH_SYSTEM_PROMPT = """You are a film research assistant with live web
search.

Rules:
1. Use web search for facts not available from a movie database or plot
   descriptions - e.g. director, producer, studio, cast trivia, awards, box
   office records, release news.
2. Prefer 2-3 targeted searches over one vague one if the first result is thin.
3. Answer concisely and only state facts you actually found in the search
   results.
4. If sources disagree or nothing reliable turns up, say so honestly rather
   than guessing.
"""


def build_research_agent():
    llm = get_llm()
    search_tool = DuckDuckGoSearchRun(name="web_search")
    return create_react_agent(llm, [search_tool], prompt=RESEARCH_SYSTEM_PROMPT)


# --- Optional upgrade: Tavily gives cleaner, source-attributed results ---
# from langchain_community.tools.tavily_search import TavilySearchResults
#
# def build_research_agent():
#     llm = get_llm()
#     search_tool = TavilySearchResults(max_results=4)
#     return create_react_agent(llm, [search_tool], prompt=RESEARCH_SYSTEM_PROMPT)
