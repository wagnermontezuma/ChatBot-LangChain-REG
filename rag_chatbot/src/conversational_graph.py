"""Conversational graph using LangGraph."""

from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from .chat_nodes import query_or_respond, tools, generate


def create_conversational_graph(llm):
    """Return a LangGraph app wiring the conversational nodes."""
    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node("query_or_respond", lambda state: query_or_respond(state, llm))
    graph_builder.add_node("tools", tools)
    graph_builder.add_node("generate", lambda state: generate(state, llm))

    graph_builder.set_entry_point("query_or_respond")

    try:
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {
                "tools": "tools",
                "default": END,
            },
        )
    except AttributeError:
        # Fallback for simplified StateGraph used in tests
        graph_builder.add_edge("query_or_respond", "tools")

    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    graph = graph_builder.compile()
    return graph
