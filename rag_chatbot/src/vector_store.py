import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from .advanced_features import CachedEmbeddings
import logging
from .logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv(dotenv_path='rag_chatbot/.env')

def get_embeddings_model(api_key: str = None):
    """
    Configura e retorna o modelo de embeddings do Google Generative AI.
    """
    if api_key is None:
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("A variável de ambiente GOOGLE_API_KEY não está configurada ou não foi fornecida.")

    # O usuário especificou GOOGLE_API_KEY e pediu para mudar para Google GenAI.
    # O modelo de embeddings do Google é geralmente "models/embedding-001".
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

def create_vector_store(documents: list[Document]):
    """
    Cria e popula um vector store em memória com os documentos fornecidos.
    """
    base_embeddings = get_embeddings_model()
    embeddings = CachedEmbeddings(base_embeddings)
    # Usando Chroma como um exemplo de vector store em memória
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        # persist_directory="./chroma_db" # Para persistir em disco, se necessário
    )
    return vector_store

def add_documents_to_vector_store(vector_store: Chroma, documents: list[Document]):
    """
    Adiciona documentos a um vector store existente.
    """
    vector_store.add_documents(documents)
    logger.info(f"Adicionados {len(documents)} documentos ao vector store.")

if __name__ == "__main__":
    # Exemplo de uso (normalmente seria chamado por outro módulo)
    # Para testar, vamos criar alguns documentos dummy
    dummy_docs = [
        Document(page_content="Este é o primeiro documento de teste para o vector store."),
        Document(page_content="Este é o segundo documento, com conteúdo diferente."),
        Document(page_content="O terceiro documento também será adicionado ao vector store.")
    ]
    
    logger.info("Criando vector store com documentos iniciais...")
    vs = create_vector_store(dummy_docs)
    logger.info(f"Vector store criado com {vs._collection.count()} documentos.")

    # Adicionando mais documentos
    more_docs = [
        Document(page_content="Este é um documento adicional para testar a função de adição."),
        Document(page_content="Mais um documento para o vector store.")
    ]
    add_documents_to_vector_store(vs, more_docs)
    logger.info(f"Total de documentos no vector store após adição: {vs._collection.count()}")

    # Exemplo de busca (requer uma chave de API válida para embeddings)
    # query = "primeiro documento"
    # results = vs.similarity_search(query)
    # print(f"\nResultados da busca para '{query}':")
    # for doc in results:
    #     print(f"- {doc.page_content[:100]}...")
