"""Demonstra o uso do chatbot com perguntas de exemplo."""
import os
from rag_chatbot.src.rag_pipeline import initialize_rag_components, create_rag_graph

QUESTIONS = [
    "O que é Task Decomposition?",
    "Quais são as conclusões sobre a decomposição de tarefas?",
]


def main():
    components = initialize_rag_components()
    rag_app = create_rag_graph(
        vector_store=components["vector_store"],
        llm=components["llm"],
        rag_prompt=components["rag_prompt"],
        structured_llm=components["structured_llm"],
    )

    for q in QUESTIONS:
        print(f"\nPergunta: {q}")
        state = rag_app.invoke({"question": q})
        print("Resposta:", state["answer"])


if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY não configurada. Consulte o README.")
    main()
