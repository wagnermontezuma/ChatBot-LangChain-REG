import importlib
from types import SimpleNamespace

def test_rag_pipeline(monkeypatch):
    pipeline = importlib.import_module('rag_chatbot.src.rag_pipeline')

    HumanMessage = importlib.import_module('langchain_core.messages').HumanMessage
    AIMessage = importlib.import_module('langchain_core.messages').AIMessage
    ToolMessage = importlib.import_module('langchain_core.messages').ToolMessage

    def dummy_analyze(state, structured_llm=None):
        msgs = state["messages"]
        ai = AIMessage(content="", additional_kwargs={"tool_calls": [{"id": "1", "args": {"query": "q", "section": "beginning"}}]})
        return {"messages": msgs + [ai]}

    def dummy_retrieve(state, vector_store=None):
        msgs = state["messages"]
        doc = SimpleNamespace(page_content="ctx", metadata={"section": "beginning"})
        tool = ToolMessage(content="", tool_call_id="1", additional_kwargs={"documents": [doc]})
        return {"messages": msgs + [tool]}

    def dummy_generate(state, llm=None, rag_prompt=None):
        msgs = state["messages"]
        return {"messages": msgs + [AIMessage(content="ans")]} 

    monkeypatch.setattr(pipeline, 'analyze_query', dummy_analyze)
    monkeypatch.setattr(pipeline, 'retrieve', dummy_retrieve)
    monkeypatch.setattr(pipeline, 'generate', dummy_generate)

    graph = pipeline.create_rag_graph(None, None, None, None)
    result = graph.invoke({"messages": [HumanMessage(content="hi")]})
    assert result["messages"][-1].content == "ans"
