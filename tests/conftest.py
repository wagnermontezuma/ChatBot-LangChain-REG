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

    # Ensure GOOGLE_API_KEY exists so get_embeddings_model doesn't fail
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
