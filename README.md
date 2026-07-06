# InternationalmovieDBAgent
InternationalmovieDBAgent
## Setup

```bash
cd movie_intelligence_system
pip install -r requirements.txt
nano  .env        # fill in your API key(s)

# The db and the pdf has to be added manually for the setup ./data/

python ingestion.py         # one-time: parse the PDF, build the Chroma vector store 
python app_cli.py           # chat in the terminal
# or:
streamlit run app_streamlit.py # to run on webbrowser
```


Disclosure : The project uses my architecture decisions for example selecting a supervisor-sub agent design replicating on Master-slave, tool choices, prompt design, and the routing/synthesis logic. 

For coding and adding graphs I used Claude for generating the initial file structure, adding and debugging the PDF parser against the actual assignment file. 


## Few Known limitations in the current project / next steps with more time and money :p

- The API calls are limited. 
- No evaluation harness — I'd add a small set of golden Q&A pairs and check
  routing + answer correctness automatically ( uses a smoke test right now )
- No caching for repeated SQL/RAG lookups within a session.
- The supervisor is a single router; for much larger toolsets you'd want
  hierarchical supervisors (a "team lead" over sub-teams) rather than one
  flat router.


## Architecture

look for MovieAgent.png file