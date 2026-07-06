# Sample queries for demoing the system

## Single-agent (sql_agent)
- "How many movies were released in 1999?"
- "What is the average budget of movies in the database?"
- "List the actors who appear in [a movie title from your DB]."
- "Which language has the most movies in the database?"

## Single-agent (rag_agent)
- "What is the movie Titanic about and what genre is it?"
- "Recommend a movie about a heist gone wrong."
- "Which movie in the database deals with a dystopian future?"

## Single-agent (research_agent)
- "Who directed Inception and did it win any Oscars?"
- "What studio produced Fight Club?"

## Multi-agent (given in the assignment brief)
- "Which film talks about Project Mayhem, when was it released, and who produced it?"
  - Expected flow: `rag_agent` (identifies the movie from the plot) -> `sql_agent`
    (release year from the DB) -> `research_agent` (producer, since that's not
    in the DB schema) -> `final_answer` synthesizes all three.

## Additional multi-agent stress tests
- "Find the movie about a shark terrorizing a beach town, tell me its budget,
  and whether it won any major awards."
  - `rag_agent` -> `sql_agent` (budget) -> `research_agent` (awards)
- "What genre is [movie X], how many actors from it also appear in other
  movies in the database, and who directed it?"
  - `rag_agent` (genre) -> `sql_agent` (cross-movie actor overlap) ->
    `research_agent` (director)
