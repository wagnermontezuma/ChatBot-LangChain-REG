import importlib
import types
import sys

def test_streamlit_app_load(monkeypatch):
    stubs = types.SimpleNamespace(
        set_page_config=lambda **k: None,
        title=lambda *a, **k: None,
        header=lambda *a, **k: None,
        info=lambda *a, **k: None,
        subheader=lambda *a, **k: None,
        chat_message=lambda *a, **k: types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, exc_type, exc, tb: None),
        chat_input=lambda *a, **k: None,
        button=lambda *a, **k: False,
        empty=lambda: types.SimpleNamespace(markdown=lambda x: None),
        markdown=lambda *a, **k: None,
        rerun=lambda: None,
        cache_resource=lambda func=None: (lambda *a, **k: func(*a, **k)) if func else (lambda x: x),
        error=lambda *a, **k: None,
        stop=lambda: None,
        session_state=type('DummyState', (dict,), {
            '__getattr__': lambda self, k: self.get(k),
            '__setattr__': lambda self, k, v: dict.__setitem__(self, k, v)
        })(),
        sidebar=__import__('contextlib').nullcontext(),
    )
    monkeypatch.setitem(sys.modules, 'streamlit', stubs)
    module = importlib.import_module('rag_chatbot.src.streamlit_app')
    assert hasattr(module, 'setup_rag_pipeline')
