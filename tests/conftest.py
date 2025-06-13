import os
import sys
import types
import pytest

@pytest.fixture(autouse=True)
def stub_libraries(monkeypatch):
    """Provide minimal stubs for external libraries used by the project."""
    # bs4
    bs4_module = types.ModuleType("bs4")
    bs4_module.BeautifulSoup = lambda *args, **kwargs: None
    bs4_module.SoupStrainer = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "bs4", bs4_module)

    # langchain_community.document_loaders
    loaders_module = types.ModuleType("langchain_community.document_loaders")
    class DummyLoader:
        def __init__(self, *args, **kwargs):
            pass
        def load(self):
            return [types.SimpleNamespace(page_content="stub document", metadata={})]
    loaders_module.WebBaseLoader = DummyLoader
    monkeypatch.setitem(sys.modules, "langchain_community.document_loaders", loaders_module)

    # langchain_text_splitters
    splitters_module = types.ModuleType("langchain_text_splitters")
    class DummySplitter:
        def __init__(self, *args, **kwargs):
            pass
        def split_documents(self, docs):
            return docs
    splitters_module.RecursiveCharacterTextSplitter = DummySplitter
    monkeypatch.setitem(sys.modules, "langchain_text_splitters", splitters_module)

    # langchain_core.documents
    core_docs_module = types.ModuleType("langchain_core.documents")
    class DummyDocument:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    core_docs_module.Document = DummyDocument
    monkeypatch.setitem(sys.modules, "langchain_core.documents", core_docs_module)

    # dotenv
    dotenv_module = types.ModuleType("dotenv")
    def load_dotenv(*args, **kwargs):
        return True
    dotenv_module.load_dotenv = load_dotenv
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_module)

    # langchain_google_genai
    genai_module = types.ModuleType("langchain_google_genai")
    class DummyEmbeddings:
        def __init__(self, *args, **kwargs):
            pass
    class DummyChatModel:
        def __init__(self, *args, **kwargs):
            pass
        def with_structured_output(self, *args, **kwargs):
            return self
        def stream(self, prompt):
            yield "answer"
    genai_module.GoogleGenerativeAIEmbeddings = DummyEmbeddings
    genai_module.ChatGoogleGenerativeAI = DummyChatModel
    genai_module.GoogleGenerativeAIEmbeddings = DummyEmbeddings
    monkeypatch.setitem(sys.modules, "langchain_google_genai", genai_module)

    # langchain_community.vectorstores
    vector_module = types.ModuleType("langchain_community.vectorstores")
    class DummyChroma:
        def __init__(self):
            self.documents = []
        @classmethod
        def from_documents(cls, documents, embedding=None, **kwargs):
            obj = cls()
            obj.documents = list(documents)
            obj.embedding = embedding
            return obj
        def as_retriever(self, **kwargs):
            return self
        def invoke(self, query):
            return self.documents
    vector_module.Chroma = DummyChroma
    monkeypatch.setitem(sys.modules, "langchain_community.vectorstores", vector_module)

    # langchain_core.runnables
    runnables_module = types.ModuleType("langchain_core.runnables")
    class DummyRunnable:
        def __or__(self, other):
            return self
        def invoke(self, value):
            return value
    runnables_module.Runnable = DummyRunnable
    runnables_module.RunnablePassthrough = DummyRunnable
    monkeypatch.setitem(sys.modules, "langchain_core.runnables", runnables_module)

    # langchain_core.output_parsers
    parsers_module = types.ModuleType("langchain_core.output_parsers")
    class DummyParser:
        def invoke(self, chunk):
            return chunk
    parsers_module.StrOutputParser = DummyParser
    monkeypatch.setitem(sys.modules, "langchain_core.output_parsers", parsers_module)

    # langchain_core.prompts
    prompts_module = types.ModuleType("langchain_core.prompts")
    class DummyPrompt:
        def __init__(self, *args, **kwargs):
            pass
        def format_messages(self, **kwargs):
            return "formatted"
        @classmethod
        def from_template(cls, template):
            return cls()
    prompts_module.ChatPromptTemplate = DummyPrompt
    monkeypatch.setitem(sys.modules, "langchain_core.prompts", prompts_module)

    # streamlit
    st_module = types.ModuleType("streamlit")
    st_module.set_page_config = lambda **k: None
    st_module.title = lambda *a, **k: None
    st_module.header = lambda *a, **k: None
    st_module.info = lambda *a, **k: None
    st_module.subheader = lambda *a, **k: None
    st_module.chat_message = lambda *a, **k: types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, exc_type, exc, tb: None)
    st_module.chat_input = lambda *a, **k: None
    st_module.button = lambda *a, **k: False
    st_module.empty = lambda: types.SimpleNamespace(markdown=lambda x: None)
    st_module.markdown = lambda *a, **k: None
    st_module.rerun = lambda: None
    st_module.error = lambda *a, **k: None
    st_module.cache_resource = lambda func=None: (lambda *a, **k: func(*a, **k)) if func else (lambda x: x)
    monkeypatch.setitem(sys.modules, "streamlit", st_module)

    # langchain
    lc_module = types.ModuleType("langchain")
    lc_module.hub = types.SimpleNamespace(pull=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "langchain", lc_module)

    # langchain_core.messages
    messages_module = types.ModuleType("langchain_core.messages")
    class BaseMessage:
        def __init__(self, content="", **kwargs):
            self.content = content
            self.additional_kwargs = kwargs

    class HumanMessage(BaseMessage):
        pass

    class AIMessage(BaseMessage):
        pass

    class ToolMessage(BaseMessage):
        def __init__(self, content="", tool_call_id=None, **kwargs):
            super().__init__(content, **kwargs)
            self.tool_call_id = tool_call_id

    messages_module.BaseMessage = BaseMessage
    messages_module.HumanMessage = HumanMessage
    messages_module.AIMessage = AIMessage
    messages_module.ToolMessage = ToolMessage
    monkeypatch.setitem(sys.modules, "langchain_core.messages", messages_module)

    # langgraph.graph
    graph_module = types.ModuleType("langgraph.graph")
    class MessagesState(dict):
        pass
    class StateGraph:
        def __init__(self, _state):
            self.nodes = {}
            self.entry = None
        def add_node(self, name, func):
            self.nodes[name] = func
        def add_edge(self, a, b):
            pass
        def set_entry_point(self, name):
            self.entry = name
        def compile(self):
            class App:
                def __init__(self, nodes, entry):
                    self.nodes = nodes
                    self.entry = entry
                def invoke(self, state):
                    for func in self.nodes.values():
                        state.update(func(state))
                    return state
            return App(self.nodes, self.entry)
    graph_module.MessagesState = MessagesState
    graph_module.StateGraph = StateGraph
    graph_module.END = "end"
    monkeypatch.setitem(sys.modules, "langgraph.graph", graph_module)

    # Ensure GOOGLE_API_KEY exists so get_embeddings_model doesn't fail
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")



