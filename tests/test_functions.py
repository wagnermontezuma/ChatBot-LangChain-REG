import importlib
from types import SimpleNamespace


def test_load_documents():
    module = importlib.import_module('rag_chatbot.src.document_loader')
    docs = module.load_documents('http://example.com')
    assert len(docs) == 1
    assert docs[0].page_content == 'stub document'


def test_split_documents():
    text_module = importlib.import_module('rag_chatbot.src.text_splitter')
    Document = importlib.import_module('langchain_core.documents').Document
    docs = [Document(page_content=f'doc {i}') for i in range(3)]
    chunks = text_module.split_documents(docs)
    assert len(chunks) == 3
    sections = {c.metadata.get('section') for c in chunks}
    assert sections <= {'beginning', 'middle', 'end'}


def test_create_vector_store():
    vector_module = importlib.import_module('rag_chatbot.src.vector_store')
    Document = importlib.import_module('langchain_core.documents').Document
    docs = [Document(page_content='content')]
    store = vector_module.create_vector_store(docs)
    assert isinstance(store, vector_module.Chroma)
    assert store.documents == docs
