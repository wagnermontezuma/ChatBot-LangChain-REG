import importlib
from types import SimpleNamespace


def test_chat_nodes_flow(monkeypatch):
    chat_nodes = importlib.import_module('rag_chatbot.src.chat_nodes')
    msgs = importlib.import_module('langchain_core.messages')

    # stub llm
    def bind_tools(tools):
        def invoke(messages):
            return msgs.AIMessage(content="", additional_kwargs={"tool_calls": [{"id": "1", "name": "retrieve", "args": {"query": "hello"}}]})
        return SimpleNamespace(invoke=invoke)
    llm = SimpleNamespace(bind_tools=bind_tools, invoke=lambda x: "final")

    state = {"messages": [msgs.HumanMessage(content="Hello")]}
    state1 = chat_nodes.query_or_respond(state, llm)
    assert state1["messages"]

    state2 = chat_nodes.tools(state1)
    assert "messages" in state2

    final = chat_nodes.generate(state2, llm, SimpleNamespace(format_messages=lambda **k: "prompt"))
    assert final["messages"][0].content == "final"
