import streamlit as st
from langchain_core.messages import HumanMessage

from graph import build_graph

st.set_page_config(page_title="International Movie Database Agent", page_icon="🎬")
st.title("🎬 International Movie Database Agent")
# st.caption("SQL analytics + RAG plot search + live web research, orchestrated with LangGraph.")

if "graph" not in st.session_state:
    with st.spinner("Loading agents..."):
        st.session_state.graph = build_graph()
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

query = st.chat_input(
    "e.g. 'Which film talks about Project Mayhem, when was it released, and who produced it?'"
)
if query:
    st.session_state.history.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)

    config = {"configurable": {"thread_id": "streamlit-session"}, "recursion_limit": 25}
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.graph.invoke(
                {"messages": [HumanMessage(content=query)]}, config=config
            )
            agents_used = [m.name for m in result["messages"] if getattr(m, "name", None)]
            if agents_used:
                st.caption(f"Consulted: {' → '.join(agents_used)}")
            answer = result["messages"][-1].content
            st.markdown(answer)
    st.session_state.history.append(("assistant", answer))
