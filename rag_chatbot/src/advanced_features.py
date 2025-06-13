import os
import time
import logging
import streamlit as st
from typing import Iterator
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Importar componentes do pipeline RAG (apenas o necessário, LLM e prompt serão passados)
from typing import TypedDict, List


class Search(TypedDict):
    query: str
    section: str


class GraphState(TypedDict):
    question: str
    query: Search
    context: List[BaseMessage] | List
    answer: str

logger = logging.getLogger(__name__)

# --- Configuração de Logging com LangSmith ---
def configure_langsmith_logging() -> None:
    """Ativa o rastreamento do LangSmith se a chave estiver configurada."""
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        logger.info("LangSmith tracing habilitado")


class CachedEmbeddings:
    """Wrapper de embeddings com cache simples."""

    def __init__(self, base_model):
        self.base = base_model
        self.cache = {}

    def _embed(self, text: str):
        if text not in self.cache:
            embed_fn = getattr(self.base, "embed_query", None)
            if callable(embed_fn):
                self.cache[text] = embed_fn(text)
            else:
                # Fallback para um embedding fictício baseado no hash
                self.cache[text] = [hash(text) % 1000]
        return self.cache[text]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


_VECTOR_STORE = None


def get_lazy_vector_store(loader_func):
    """Carrega o vector store apenas na primeira chamada."""
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        _VECTOR_STORE = loader_func()
    return _VECTOR_STORE


def validate_question(question: str) -> bool:
    """Verifica se a pergunta é válida."""
    return bool(question and question.strip())

# --- Streaming de Respostas ---
def stream_rag_response(question: str, rag_app: Runnable, llm_model: any, rag_prompt_template: ChatPromptTemplate, vector_store: any) -> Iterator[str]:
    """Gera resposta em modo streaming com tratamento de erros e métricas."""

    if not validate_question(question):
        yield "Pergunta vazia."
        return

    start_time = time.time()

    # Para streaming, precisamos do contexto. Executa as etapas de análise e recuperação
    # de forma síncrona para obter o contexto.
    
    # Obter o LLM estruturado para análise de consulta
    structured_llm = llm_model.with_structured_output(GraphState.__annotations__['query']) # Reutiliza o schema Search do GraphState

    # Prompt para análise da consulta (duplicado de rag_pipeline para evitar dependência circular)
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", "Analise a pergunta do usuário e determine a consulta principal e a seção relevante do documento (beginning, middle, end)."),
        ("human", "Pergunta: {question}\n\nRetorne a consulta e a seção no formato JSON com os campos 'query' e 'section'.")
    ])
    analysis_chain = analysis_prompt | structured_llm
    
    # Analisar a consulta
    parsed_query = analysis_chain.invoke({"question": question})

    # Recuperar contexto com filtro
    retriever_with_filter = vector_store.as_retriever(
        search_kwargs={"filter": {"section": parsed_query["section"]}}
    )
    context = retriever_with_filter.invoke(parsed_query["query"])
    formatted_context = "\n\n".join([doc.page_content for doc in context])

    # Criar a cadeia de RAG para streaming
    # Formatar o prompt antes de passar para o LLM
    formatted_prompt = rag_prompt_template.format_messages(context=formatted_context, question=question)
    
    # Invocar o LLM com streaming
    try:
        for chunk in llm_model.stream(formatted_prompt):
            yield StrOutputParser().invoke(chunk)
    except Exception as exc:
        logger.error("Falha no streaming: %s", exc)
        yield "Desculpe, ocorreu um erro ao gerar a resposta."

    logger.info("Tempo de resposta: %.2fs", time.time() - start_time)

# --- Suporte para Múltiplos Modos de Invocação (Sync, Async) ---
# O LangChain e LangGraph já suportam isso nativamente com .invoke() e .ainvoke()
# e .stream() e .astream().
# A função `stream_rag_response` acima já demonstra um modo de invocação (streaming).
# O `rag_app.invoke` no streamlit_app.py já é síncrono.
# Para async, precisaríamos de um ambiente async (ex: FastAPI, ou `asyncio.run` no main).

# Exemplo de uso síncrono (já feito no main.py e streamlit_app.py)
def invoke_rag_sync(question: str, rag_app: Runnable) -> GraphState:
    return rag_app.invoke({"question": question})

# Exemplo de uso assíncrono (requer um loop de eventos)
async def invoke_rag_async(question: str, rag_app: Runnable) -> GraphState:
    return await rag_app.ainvoke({"question": question})

# --- Configurar Logging com LangSmith para Rastreamento ---
# Isso já está configurado via variáveis de ambiente no .env (LANGSMITH_API_KEY, LANGSMITH_TRACING=true)
# e é ativado automaticamente pelas bibliotecas LangChain/LangGraph.
# O erro 502 anterior indica um problema de conexão com o servidor LangSmith, não de configuração.

# --- Otimizações de Performance (Esboço) ---
# Cache para embeddings já calculados:
# O ChromaDB já faz um cache interno de embeddings para documentos que já foram adicionados.
# Para um cache mais explícito, poderíamos usar `langchain.globals.set_llm_cache(InMemoryCache())`
# ou um cache mais persistente.

# Lazy loading do vector store:
# Atualmente, o vector store é carregado e populado na inicialização do Streamlit (`@st.cache_resource`).
# Isso já é uma forma de lazy loading, pois só acontece uma vez na primeira execução.
# Para um lazy loading mais granular (ex: carregar chunks sob demanda), seria mais complexo.

# Configuração de timeouts apropriados:
# Pode ser configurado nos modelos LLM (ex: `timeout` no construtor do ChatGoogleGenerativeAI)
# ou em chamadas de rede subjacentes.

# --- Tratamento de Erros (Esboço) ---
# Try-catch para falhas de conexão: Já implementado em `main.py` e `streamlit_app.py`
# Fallback responses quando RAG falha: Pode ser implementado na função `generate` ou no `streamlit_app.py`
# para retornar uma mensagem padrão se o LLM falhar.
# Validação de entrada do usuário: Pode ser adicionada no `streamlit_app.py` para verificar
# se a pergunta não está vazia, etc.

# --- Métricas e Monitoramento (Esboço) ---
# Tempo de resposta: Pode ser medido envolvendo as chamadas `invoke` com `time.time()`.
# Qualidade das recuperações: Requer avaliação manual ou métricas mais avançadas (ex: RAGAS).
# Logs estruturados: Pode ser implementado usando a biblioteca `logging` do Python e configurando formatadores.

# --- Testes Unitários Básicos (Esboço) ---
# Seriam arquivos separados (ex: `tests/test_document_loader.py`) usando `unittest` ou `pytest`.
# Exemplo de estrutura:
# import unittest
# from src.document_loader import load_documents
# class TestDocumentLoader(unittest.TestCase):
#     def test_load_documents(self):
#         docs = load_documents(url="some_test_url")
#         self.assertGreater(len(docs), 0)

if __name__ == "__main__":
    # Exemplo de uso da função de streaming
    st.write("Demonstração de Streaming (apenas se executado diretamente no Streamlit)")
    # Este bloco não será executado diretamente via `python advanced_features.py`
    # mas sim quando importado e usado em `streamlit_app.py`.
