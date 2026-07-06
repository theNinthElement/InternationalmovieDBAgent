"""
SQL specialist agent: answers analytical questions over the structured movie
database (movies, financials, actors, movie_actor, languages).

Design choice: rather than hand-writing SQL myself, I hand the agent the
standard LangChain SQLDatabaseToolkit (list_tables, get_schema, query,
query_checker) and let it introspect the schema at run time. That means it
keeps working even if column names differ slightly from what's described in
the assignment brief, and it self-corrects malformed SQL via the checker tool.
"""
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent

from config import get_llm, DB_PATH

SQL_SYSTEM_PROMPT = """You are a meticulous SQL analyst for a movie database.
Tables: movies (movie_id, movie_title, industry, release_year, imdb_rating,
studio, language_id), financials (movie_id, budget, revenue, unit, currency),
actors (actor_id, actor_name, birth_year), movieactor (join table:
movie_id, actor_id), languages (language_id, language_name).
Note the join table is named `movieactor` - no underscore. Financial figures
carry `unit` (e.g. millions/billions) and `currency` columns - report those
alongside any budget/revenue number, and never sum across mixed units.

Rules:
1. If a query fails or a column seems missing, inspect the schema with the
   list-tables/get-schema tools rather than guessing variations.
2. Only ever run SELECT queries. Never INSERT, UPDATE, DELETE, DROP, or ALTER.
3. Use the query-checker tool before executing a query you are unsure about.
4. Prefer precise, minimal queries (WHERE / JOIN / GROUP BY as needed) over
   pulling entire tables into memory.
5. When you have the answer, respond with a short, direct, natural-language
   sentence - do not dump raw SQL result tables at the user.
"""


def build_sql_agent(db_path: str = DB_PATH):
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    llm = get_llm()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return create_react_agent(llm, toolkit.get_tools(), prompt=SQL_SYSTEM_PROMPT)
