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


# Crear el flujo de trabajo
workflow = StateGraph(State)

# Agregar nodos
workflow.add_node("categorize", categorize)
workflow.add_node("handle_general", handle_general)
workflow.add_node("search_agent", search_agent)
workflow.add_node("email_egent", email_egent)

# Agregar las ramas condicionales
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

# Punto de entrada
workflow.set_entry_point("categorize")

# Compilar el grafo
app = workflow.compile()


# Función para iniciar un chat en la terminal
def start_chat():
    print("💬 Bienvenido al asistente interactivo por terminal!")
    print("Escribe 'salir' para terminar la conversación.\n")

    # Inicializar el historial vacío
    conversation_history = []

    while True:
        # Entrada del usuario
        user_query = input("👤 Tú: ")
        if user_query.lower() == "salir":
            print("👋 ¡Adiós! Hasta la próxima.")
            break

        # Preparar estado inicial con historial
        query_state = {
            "query": user_query,
            "history": conversation_history  # Pasar el historial actual
        }

        # Invocar el grafo con la consulta y el historial
        result = app.invoke(query_state)

        # Mostrar respuesta del asistente
        category = result.get("category", "Desconocida")
        response = result.get(
            "response", "Lo siento, no entiendo la consulta.")

        print(f"🤖 Asistente (Categoría: {category}): {response}\n")

        # Actualizar el historial con la interacción
        conversation_history = result.get("history", conversation_history)


# Iniciar el chat
if __name__ == "__main__":
    start_chat()

