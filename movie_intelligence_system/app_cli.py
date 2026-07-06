from langchain_core.messages import HumanMessage

from graph import build_graph

THREAD_ID = "cli-session"


def main():
    print("Building agents (LLM, SQL toolkit, vector store, search tool)...")
    graph = build_graph()
    config = {"configurable": {"thread_id": THREAD_ID}, "recursion_limit": 25}

    print("\nInternational Movie Database Agent - ask me anything about the movie catalog.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        result = graph.invoke({"messages": [HumanMessage(content=query)]}, config=config)

        # Show which specialists were consulted - useful for explaining the
        # routing decisions live during a demo.
        agent_steps = [m.name for m in result["messages"] if getattr(m, "name", None)]
        if agent_steps:
            print(f"   (consulted: {' -> '.join(agent_steps)})")

        print(f"Assistant: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
