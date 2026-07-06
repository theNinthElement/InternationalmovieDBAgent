# Movie Intelligence System

A multi-agent movie assistant built with **LangChain + LangGraph** that answers
questions by combining structured SQL analytics, RAG over 100 movie
descriptions, and live web research — routed automatically by an LLM
supervisor, no hand-coded "if multi-part question" logic.

## Architecture

```
                        ┌─────────────┐
             ┌─────────▶│ supervisor  │◀─────────┐
             │          └──────┬──────┘          │
             │                 │ routes to one     │
             │        ┌────────┼────────┐         │
             │        ▼        ▼        ▼         │
             │   sql_agent  rag_agent  research_agent
             │        │        │        │         │
             └────────┴────────┴────────┴─────────┘
                                │
                       supervisor says FINISH
                                ▼
                          final_answer ──▶ END
```

- **Supervisor** (`graph.py`): an LLM with structured output (`RouteDecision`)
  that reads the conversation and picks exactly one specialist to act next —
  `sql_agent`, `rag_agent`, `research_agent`, or `FINISH`.
- **Three specialists**, each a small LangGraph `create_react_agent` bound to
  its own tool(s):
  - `sql_tool.py` — the standard LangChain `SQLDatabaseToolkit` (list tables,
    get schema, run query, check query) over the `.db` file. The agent
    inspects the schema itself rather than me hard-coding column names, so it
    keeps working even if `movies`/`financials`/`actors` columns differ
    slightly from the brief.
  - `rag_tool.py` — a Chroma retriever tool over the 100 parsed movie
    descriptions (`ingestion.py` builds the store from the PDF).
  - `research_tool.py` — DuckDuckGo web search (no API key needed) for
    anything not in the DB or description corpus: directors, producers,
    studios, awards.
- **Final answer node**: after the supervisor calls `FINISH`, one more LLM
  call synthesizes everything the specialists found into a single clean
  answer, instead of just returning the last specialist's raw message.
- Control **always returns to the supervisor** after a specialist runs. That
  loop is what makes multi-hop questions work: for *"which film talks about
  Project Mayhem, when was it released, and who produced it?"*, the
  supervisor chains `rag_agent` (identifies Fight Club from the plot) →
  `sql_agent` (release year from the DB) → `research_agent` (producer, since
  that's not in the SQL schema) → `FINISH`, with no branch of that logic
  written by hand.

### Why this design

- **Supervisor pattern over a single ReAct agent with 3 tools.** A single
  agent with all three tools would also technically satisfy the brief (the
  assignment explicitly allows it), but a supervisor gives each specialist
  its own focused system prompt and a narrow toolset, which in practice makes
  routing more reliable and the transcript easier to explain/debug live — you
  can point at exactly which agent produced which fact.
- **All three specialists built the same way** (`create_react_agent` + a
  toolset), so the framework is uniform and easy to extend with a fourth
  agent later.
- **LLM-driven routing, not keyword rules.** Whether a question needs one
  agent or three is decided by the supervisor at run time, so it generalizes
  to phrasings I didn't anticipate.
- **MemorySaver checkpointer** gives the CLI/Streamlit apps free multi-turn
  memory (e.g. a follow-up "what about its sequel?"), keyed by `thread_id`.
- **DuckDuckGo over Tavily by default** — zero signup friction for a timed
  take-home. Swap-in for Tavily is left as a commented block in
  `research_tool.py`.
- **Local HuggingFace embeddings (`all-MiniLM-L6-v2`) over an embeddings
  API** — avoids requiring a second API key just to embed 100 short
  documents; the accuracy tradeoff is negligible at this corpus size.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in your API key(s)

# Put your assignment files here:
#   data/movies.db
#   data/RAG_movies.pdf

python ingestion.py         # one-time: parse the PDF, build the Chroma store
python app_cli.py           # chat in the terminal
# or:
streamlit run app_streamlit.py
```

## Verified against the actual assignment files

- **PDF parser**: all 100 records (Movie IDs 101–200) parse with all four
  fields present. The PDF extracts with a newline between nearly every word,
  so the parser normalizes whitespace first, then extracts labeled fields.
  Page-number artifacts at page breaks (e.g. "1seeking") are stripped, with
  ordinals like "85th" protected.
- **SQL database**: tables are `movies`, `financials`, `actors`,
  `movieactor` (note: no underscore, unlike the brief), `languages` —
  100 movies, IDs 101–200, matching the PDF, so RAG hits can be joined
  directly against SQL by ID or title. Financials carry `unit`/`currency`
  columns; the SQL agent's prompt instructs it to report those and never sum
  across mixed units.
- **Wiring**: Chroma store build, retriever tool, all three agents, and the
  full supervisor graph compile and run against these files (verified with a
  deterministic test embedding; the real `all-MiniLM-L6-v2` model, ~90 MB,
  downloads automatically from HuggingFace on your first `python
  ingestion.py`).
- After ingestion, run `python smoke_test.py` to execute the three example
  questions from the brief and print the routing for each.

## If you're short on the 3-4 hour budget

Priority order to keep, in case you need to cut scope:
1. `sql_tool.py` + `rag_tool.py` + `graph.py` + `app_cli.py` — the core
   multi-agent loop and two of the three example queries.
2. `research_tool.py` — needed for the full three-hop example query.
3. `app_streamlit.py` — explicitly optional per the brief; the CLI alone is
   enough for a live demo.

## Known limitations / next steps with more time

- No evaluation harness — I'd add a small set of golden Q&A pairs and check
  routing + answer correctness automatically (e.g. with LangSmith or a
  simple pytest suite).
- No caching for repeated SQL/RAG lookups within a session.
- The supervisor is a single router; for much larger toolsets you'd want
  hierarchical supervisors (a "team lead" over sub-teams) rather than one
  flat router.
- No guardrail against the SQL agent being prompted to attempt destructive
  queries beyond the system-prompt instruction — for production I'd enforce
  this at the DB-connection level (read-only user/connection string) rather
  than relying on the prompt alone.

## AI tool disclosure

I used Claude (Anthropic) as a coding assistant to scaffold this project:
generating the initial file structure, the LangGraph supervisor boilerplate,
drafting docstrings/README text, and debugging the PDF parser against the
actual assignment file. API calls were verified against the
actually-installed package versions rather than memorized import paths (the
ecosystem has moved to a `langchain>=1.0` layout, and the DuckDuckGo backend
package was renamed to `ddgs`). The
architecture decisions — supervisor-vs-single-agent, tool choices, prompt
design, and the routing/synthesis logic — reflect my own design choices for
this task; I reviewed, tested, and adjusted the generated code rather than
using it unmodified.

*(Adjust this paragraph to accurately reflect how you actually used AI
tools before submitting — this is a starting draft, not a final claim on
your behalf.)*
