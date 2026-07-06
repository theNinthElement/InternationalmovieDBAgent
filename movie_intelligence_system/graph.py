import functools
from typing import Annotated, Literal, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from config import get_llm
from sql_tool import build_sql_agent
from rag_tool import build_rag_agent
from research_tool import build_research_agent

MEMBERS = ["sql_agent", "rag_agent", "research_agent"]

SUPERVISOR_PROMPT = f"""You are the supervisor of a movie-intelligence team \
with three specialists: {MEMBERS}.

- sql_agent: structured facts from the movie database - release years, box \
office / budget / revenue figures, actors, cast lists, spoken languages, \
counts and aggregations.
- rag_agent: plot/theme/genre lookups over a curated set of 100 detailed \
movie descriptions (use this to identify a movie from a description of its \
plot, or to summarize what a movie is about).
- research_agent: live web search for anything not covered by the above - \
directors, producers, awards, real-world trivia, recent news. (Note: studio \
IS in the SQL database, so prefer sql_agent for studio questions.)

The SQL database and the description corpus cover the same 100 movies and \
share Movie IDs (101-200), so an ID or title found by one agent can be used \
directly by the other.

If the user's question has nothing to do with movies (not answerable via \
the SQL database, the plot/description corpus, or movie-related web \
research), route to OFF_TOPIC immediately instead of calling a specialist.

Given the conversation so far, decide who should act next. Route to exactly \
one specialist per turn. Only choose FINISH once every part of the user's \
original question has actually been answered by the specialists' outputs \
above - do not FINISH early just because one sub-question has been \
answered while others (e.g. "who produced it") have not."""

FINAL_ANSWER_PROMPT = """Using only the information the specialists gathered
above, write one clear, direct, well-organized answer to the user's original
question. Do not mention agents, tools, or your internal process."""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next: str


class RouteDecision(BaseModel):
    next: Literal["sql_agent", "rag_agent", "research_agent", "OFF_TOPIC", "FINISH"]


def _agent_node(state: AgentState, agent, name: str):
    result = agent.invoke({"messages": state["messages"]})
    last_message = result["messages"][-1]
    # Re-tag as a named HumanMessage so the supervisor reads it as a report
    # "from sql_agent" rather than as the assistant's own final voice.
    return {"messages": [HumanMessage(content=last_message.content, name=name)]}


def _supervisor_node(state: AgentState, llm):
    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + list(state["messages"])
    decision = llm.with_structured_output(RouteDecision).invoke(messages)
    return {"next": decision.next}


def _off_topic_node(state: AgentState):
    message = (
        "I can only answer questions about the movies in this system "
        "(plot, cast, financials, directors, trivia). Try rephrasing your "
        "question around a movie."
    )
    return {"messages": [AIMessage(content=message)]}


def _final_answer_node(state: AgentState, llm):
    messages = list(state["messages"]) + [SystemMessage(content=FINAL_ANSWER_PROMPT)]
    response = llm.invoke(messages)
    return {"messages": [response]}


def build_graph():
    """Compile the full supervisor graph. Instantiates the LLM once and
    shares it across the supervisor + final-answer nodes; each specialist
    builds (and owns) its own LLM + tool set internally."""
    llm = get_llm()

    sql_agent = build_sql_agent()
    rag_agent = build_rag_agent()
    research_agent = build_research_agent()

    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", functools.partial(_supervisor_node, llm=llm))
    workflow.add_node("sql_agent", functools.partial(_agent_node, agent=sql_agent, name="sql_agent"))
    workflow.add_node("rag_agent", functools.partial(_agent_node, agent=rag_agent, name="rag_agent"))
    workflow.add_node(
        "research_agent", functools.partial(_agent_node, agent=research_agent, name="research_agent")
    )
    workflow.add_node("final_answer", functools.partial(_final_answer_node, llm=llm))
    workflow.add_node("off_topic_node", _off_topic_node)

    workflow.add_edge(START, "supervisor")
    for member in MEMBERS:
        workflow.add_edge(member, "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {**{m: m for m in MEMBERS}, "FINISH": "final_answer", "OFF_TOPIC": "off_topic_node"},
    )
    workflow.add_edge("final_answer", END)
    workflow.add_edge("off_topic_node", END)

    # MemorySaver gives the CLI/Streamlit apps multi-turn memory (follow-up
    # questions like "what about its sequel?") for free, keyed by thread_id.
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)
