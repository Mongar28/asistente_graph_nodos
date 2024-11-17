from langgraph.graph import StateGraph, END
from agents_graph.agents import (
    search_agent, handle_general,
    categorize, email_egent, State
)


def route_query(state: State) -> str:
    """Route the query based on its sentiment and category."""
    if state["category"] == "General":
        return "handle_general"
    elif state["category"] == "Search":
        return "search_agent"
    elif state["category"] == "Email":
        return "email_egent"
    else:
        return "handle_general"


workflow = StateGraph(State)

# Agrega los nodos
workflow.add_node("categorize", categorize)
workflow.add_node("handle_general", handle_general)
workflow.add_node("search_agent", search_agent)
workflow.add_node("email_egent", email_egent)

# Añade las ramas condicionales
workflow.add_conditional_edges(
    "categorize",
    route_query,
    {
        "handle_general": "handle_general",
        "search_agent": "search_agent",
        "email_egent": "email_egent"
    }
)

workflow.add_edge("handle_general", END)
workflow.add_edge("search_agent", END)
workflow.add_edge("email_egent", END)


# Set entry point
workflow.set_entry_point("categorize")

# Compile the graph
app = workflow.compile()

mermaid_png = app.get_graph().draw_mermaid_png()

# # Guardar la imagen en un archivo local
with open("graph_output.png", "wb") as f:
    f.write(mermaid_png)

query: str = {
    "query": "De que se trata el último correo que recibí?"}

result = app.invoke(query)

print(
    f"""
    Query: {query}
    Category: {result['category']}
    Response: {result['response']}
    """
)
