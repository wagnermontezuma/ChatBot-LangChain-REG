import sys
import os
import logging

# Adiciona o diretório raiz do projeto ao sys.path
# Isso é necessário para que as importações relativas funcionem corretamente
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import TypedDict, Literal
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate # Adicionar esta importação
from langgraph.graph import MessagesState, StateGraph, END

# Importar funções dos módulos criados
from rag_chatbot.src.document_loader import load_documents
from rag_chatbot.src.text_splitter import split_documents
from rag_chatbot.src.vector_store import create_vector_store, get_embeddings_model
from rag_chatbot.src.llm_config import get_chat_model
from rag_chatbot.src.prompt_template import get_rag_prompt_template
from rag_chatbot.src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# 1. Definir o schema Search
class Search(TypedDict):
    """
    Schema para a análise da consulta.
    """
    query: str
    section: Literal["beginning", "middle", "end"]


# Variáveis globais (serão inicializadas e retornadas por initialize_rag_components)
# Não serão mais None, mas sim os objetos reais.
# Removendo a necessidade de importá-los diretamente de rag_pipeline em outros módulos.
# Eles serão passados como argumentos ou obtidos do retorno de initialize_rag_components.

def initialize_rag_components():
    """
    Inicializa e retorna os componentes RAG (LLM, Prompt, Vector Store, Retriever, Structured LLM).
    """
    logger.info("Inicializando componentes RAG...")
    
    # Carregar e dividir documentos para o vector store
    documents = load_documents()
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)
    
    # Configurar LLM e Prompt
    llm = get_chat_model()
    rag_prompt = get_rag_prompt_template()

    # Configurar LLM estruturado para análise de consulta
    structured_llm = llm.with_structured_output(Search)
    
    logger.info("Componentes RAG inicializados.")
    return {
        "vector_store": vector_store,
        "llm": llm,
        "rag_prompt": rag_prompt,
        "structured_llm": structured_llm
    }

# 2. Implementar as funções do pipeline
def analyze_query(state: MessagesState, structured_llm):
    """Analisa a mensagem do usuário e retorna uma chamada de ferramenta."""
    logger.info("---ANALISANDO CONSULTA---")
    messages = state["messages"]
    question = messages[-1].content

    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", "Analise a pergunta do usuário e determine a consulta principal e a seção relevante do documento (beginning, middle, end)."),
        ("human", "Pergunta: {question}\n\nRetorne a consulta e a seção no formato JSON com os campos 'query' e 'section'.")
    ])
    analysis_chain = analysis_prompt | structured_llm
    parsed_query = analysis_chain.invoke({"question": question})

    logger.info(f"Consulta analisada: {parsed_query}")
    tool_call = {"id": "vs_query", "name": "vector_search", "args": parsed_query}
    ai_msg = AIMessage(content="", additional_kwargs={"tool_calls": [tool_call]})
    return {"messages": messages + [ai_msg]}

def retrieve(state: MessagesState, vector_store):
    """Recupera documentos conforme a consulta analisada."""
    logger.info("---RECUPERANDO CONTEXTO---")
    messages = state["messages"]
    ai_msg = messages[-1]
    parsed_query = ai_msg.additional_kwargs["tool_calls"][0]["args"]

    retriever_with_filter = vector_store.as_retriever(
        search_kwargs={"filter": {"section": parsed_query["section"]}}
    )

    documents = retriever_with_filter.invoke(parsed_query["query"])
    tool_call_id = ai_msg.additional_kwargs["tool_calls"][0].get("id", "vs_query")
    tool_msg = ToolMessage(
        content="",
        tool_call_id=tool_call_id,
        additional_kwargs={"documents": documents},
    )
    return {"messages": messages + [tool_msg]}

def generate(state: MessagesState, llm, rag_prompt):
    """Gera a resposta final utilizando o contexto recuperado."""
    logger.info("---GERANDO RESPOSTA---")
    messages = state["messages"]
    tool_msg = messages[-1]
    documents = tool_msg.additional_kwargs.get("documents", [])

    # Encontrar a última pergunta do usuário
    question = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)),
        "",
    )

    rag_chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    formatted_context = "\n\n".join([doc.page_content for doc in documents])
    answer = rag_chain.invoke({"context": formatted_context, "question": question})
    ai_msg = AIMessage(content=answer)
    return {"messages": messages + [ai_msg]}

# 3. Configurar o grafo LangGraph
def create_rag_graph(vector_store, llm, rag_prompt, structured_llm):
    """
    Cria e compila o grafo LangGraph para o pipeline RAG.
    """
    workflow = StateGraph(MessagesState)

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
    # Inicializar componentes RAG antes de criar e usar o grafo
    components = initialize_rag_components()
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
        logger.info(f"---TESTANDO PIPELINE RAG COM A PERGUNTA: '{question}'---")
        final_state = rag_app.invoke({"messages": [HumanMessage(content=question)]})

        tool_msg = final_state["messages"][-2]
        for i, doc in enumerate(tool_msg.additional_kwargs.get("documents", [])):
            logger.info(f"Documento {i+1} (parcial): {doc.page_content[:300]}...")
            logger.info(f"Seção: {doc.metadata.get('section', 'N/A')}")

        answer_msg = final_state["messages"][-1]
        logger.info("---RESPOSTA GERADA---")
        logger.info(answer_msg.content)
