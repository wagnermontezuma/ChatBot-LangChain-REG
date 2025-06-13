import importlib
from types import SimpleNamespace

def test_rag_pipeline(monkeypatch):
    pipeline = importlib.import_module('rag_chatbot.src.rag_pipeline')

    def dummy_analyze(state, structured_llm=None):
        return {"question": state["question"], "query": {"query": "q", "section": "beginning"}}

    def dummy_retrieve(state, vector_store=None):
        doc = SimpleNamespace(page_content="ctx", metadata={"section": "beginning"})
        return {"question": state["question"], "query": state["query"], "context": [doc]}

    def dummy_generate(state, llm=None, rag_prompt=None):
        return {"question": state["question"], "context": state["context"], "answer": "ans"}

    monkeypatch.setattr(pipeline, 'analyze_query', dummy_analyze)
    monkeypatch.setattr(pipeline, 'retrieve', dummy_retrieve)
    monkeypatch.setattr(pipeline, 'generate', dummy_generate)

    graph = pipeline.create_rag_graph(None, None, None, None)
    result = graph.invoke({"question": "hi"})
    assert result["answer"] == "ans"
