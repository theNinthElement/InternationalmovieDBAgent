from langchain_core.messages import HumanMessage

from graph import build_graph

QUESTIONS = [
    "How many movies were released in 1999?",
    "What is the movie Titanic about and what genre is it?",
    "Which film talks about Project Mayhem, when was it released, and who produced it?",
]


def main():
    graph = build_graph()
    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n{'=' * 70}\nQ{i}: {q}\n{'-' * 70}")
        config = {"configurable": {"thread_id": f"smoke-{i}"}, "recursion_limit": 25}
        result = graph.invoke({"messages": [HumanMessage(content=q)]}, config=config)
        route = [m.name for m in result["messages"] if getattr(m, "name", None)]
        print(f"route: {' -> '.join(route) if route else '(none)'}")
        print(f"answer: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
