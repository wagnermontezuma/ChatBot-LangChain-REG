import sys
import os

# Adiciona o diretório raiz do projeto ao sys.path
# Isso é necessário para que as importações relativas funcionem corretamente
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import List, TypedDict, Literal
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate # Adicionar esta importação
from langgraph.graph import StateGraph, END

# Importar funções dos módulos criados
from rag_chatbot.src.document_loader import load_documents
from rag_chatbot.src.text_splitter import split_documents
from rag_chatbot.src.vector_store import create_vector_store, get_embeddings_model
from rag_chatbot.src.llm_config import get_chat_model
from rag_chatbot.src.prompt_template import get_rag_prompt_template

# 1. Definir o schema Search
class Search(TypedDict):
    """
    Schema para a análise da consulta.
    """
    query: str
    section: Literal["beginning", "middle", "end"]

# 1. Definir a classe State
class GraphState(TypedDict):
    """
    Representa o estado do nosso grafo.

    Atributos:
        question: Pergunta do usuário.
        query: Consulta analisada (com query e section).
        context: Documentos recuperados.
        answer: Resposta gerada pelo LLM.
    """
    question: str
    query: Search
    context: List[Document]
    answer: str

# Variáveis globais (serão inicializadas e retornadas por initialize_rag_components)
# Não serão mais None, mas sim os objetos reais.
# Removendo a necessidade de importá-los diretamente de rag_pipeline em outros módulos.
# Eles serão passados como argumentos ou obtidos do retorno de initialize_rag_components.

def initialize_rag_components(doc_url: str | None = None):
    """
    Inicializa e retorna os componentes RAG (LLM, Prompt, Vector Store, Retriever, Structured LLM).
    """
    print("Inicializando componentes RAG...")
    
    # Carregar e dividir documentos para o vector store
    documents = load_documents(doc_url)
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)
    
    # Configurar LLM e Prompt
    llm = get_chat_model()
    rag_prompt = get_rag_prompt_template()

    # Configurar LLM estruturado para análise de consulta
    structured_llm = llm.with_structured_output(Search)
    
    print("Componentes RAG inicializados.")
    return {
        "vector_store": vector_store,
        "llm": llm,
        "rag_prompt": rag_prompt,
        "structured_llm": structured_llm
    }

# 2. Implementar as funções do pipeline
def analyze_query(state: GraphState, structured_llm):
    """
    Analisa a pergunta do usuário para extrair a consulta e a seção.
    """
    print("---ANALISANDO CONSULTA---")
    question = state["question"]
    
    # Prompt para análise da consulta
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", "Analise a pergunta do usuário e determine a consulta principal e a seção relevante do documento (beginning, middle, end)."),
        ("human", "Pergunta: {question}\n\nRetorne a consulta e a seção no formato JSON com os campos 'query' e 'section'.")
    ])

    # Cadeia de análise
    analysis_chain = analysis_prompt | structured_llm
    
    # Invocar a cadeia de análise
    parsed_query = analysis_chain.invoke({"question": question})
    
    print(f"Consulta analisada: {parsed_query}")
    return {"question": question, "query": parsed_query}

def retrieve(state: GraphState, vector_store):
    """
    Busca documentos similares usando o retriever com filtros baseados na seção.
    """
    print("---RECUPERANDO CONTEXTO---")
    question = state["question"]
    parsed_query = state["query"]
    
    # Configurar o retriever com filtro
    # O filtro é aplicado no momento da busca
    retriever_with_filter = vector_store.as_retriever(
        search_kwargs={"filter": {"section": parsed_query["section"]}}
    )
    
    documents = retriever_with_filter.invoke(parsed_query["query"])
    return {"question": question, "query": parsed_query, "context": documents}

def generate(state: GraphState, llm, rag_prompt):
    """
    Gera resposta usando o LLM e prompt template.
    """
    print("---GERANDO RESPOSTA---")
    question = state["question"]
    context = state["context"]

    # Criar a cadeia de RAG
    rag_chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    # Formatar o contexto para o prompt
    formatted_context = "\n\n".join([doc.page_content for doc in context])

    answer = rag_chain.invoke({"context": formatted_context, "question": question})
    return {"question": question, "context": context, "answer": answer}

# 3. Configurar o grafo LangGraph
def create_rag_graph(vector_store, llm, rag_prompt, structured_llm):
    """
    Cria e compila o grafo LangGraph para o pipeline RAG.
    """
    workflow = StateGraph(GraphState)

    # Adicionar nós, passando os componentes necessários
    workflow.add_node("analyze_query", lambda state: analyze_query(state, structured_llm))
    workflow.add_node("retrieve", lambda state: retrieve(state, vector_store))
    workflow.add_node("generate", lambda state: generate(state, llm, rag_prompt))

    # Adicionar sequência
    workflow.add_edge("analyze_query", "retrieve")
    workflow.add_edge("retrieve", "generate")

    # Configurar edge do START
    workflow.set_entry_point("analyze_query")

    # Compilar o grafo
    app = workflow.compile()
    return app

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Testa o pipeline RAG.")
    parser.add_argument("--url", help="URL dos documentos", default=None)
    args = parser.parse_args()

    # Inicializar componentes RAG antes de criar e usar o grafo
    components = initialize_rag_components(args.url)
    vector_store = components["vector_store"]
    llm = components["llm"]
    rag_prompt = components["rag_prompt"]
    structured_llm = components["structured_llm"]
    
    rag_app = create_rag_graph(vector_store, llm, rag_prompt, structured_llm)

    # 4. Testar o pipeline com perguntas específicas sobre diferentes seções
    test_questions = [
        "Qual é a introdução sobre decomposição de tarefas?", # Espera "beginning"
        "Fale sobre os métodos de decomposição de tarefas.", # Espera "middle"
        "Quais são as conclusões sobre a decomposição de tarefas?", # Espera "end"
        "O que é Task Decomposition?" # Pergunta geral
    ]

    for question in test_questions:
        print(f"\n---TESTANDO PIPELINE RAG COM A PERGUNTA: '{question}'---")
        final_state = rag_app.invoke({"question": question})

        print("\n---CONTEXTO RECUPERADO---")
        for i, doc in enumerate(final_state["context"]):
            print(f"Documento {i+1} (parcial): {doc.page_content[:300]}...")
            print(f"Seção: {doc.metadata.get('section', 'N/A')}")

        print("\n---RESPOSTA GERADA---")
        print(final_state["answer"])
