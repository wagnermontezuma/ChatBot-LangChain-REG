import os
import logging
from dotenv import load_dotenv
from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.vector_store import create_vector_store
from src.rag_pipeline import initialize_rag_components, create_rag_graph
from src.logging_config import setup_logging

def main():
    # Carrega as variáveis de ambiente do arquivo .env
    load_dotenv(dotenv_path='rag_chatbot/.env')
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Iniciando o processo de indexação de documentos...")

    # 1. Carregar documentos
    logger.info("Carregando documentos da URL...")
    documents = load_documents()
    logger.info(f"Total de documentos carregados: {len(documents)}")

    if not documents:
        logger.error("Nenhum documento foi carregado. Verifique a URL ou a configuração do BeautifulSoup.")
        return

    # 2. Dividir documentos em chunks
    logger.info("Dividindo documentos em chunks...")
    chunks = split_documents(documents)
    logger.info(f"Total de chunks criados: {len(chunks)}")

    if not chunks:
        logger.error("Nenhum chunk foi criado. Verifique o text splitter.")
        return

    # 3. Criar e popular o vector store (opcional para este teste, mas importante para o RAG)
    logger.info("Criando e populando o vector store (isso pode levar um tempo e requer GOOGLE_API_KEY)...")
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY não está configurada no .env")

        vector_store = create_vector_store(chunks)  # create_vector_store já carrega a chave do ambiente
        logger.info(f"Vector store populado com {vector_store._collection.count()} chunks.")
        logger.info("Sistema de indexação de documentos configurado e testado com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao criar o vector store: {e}")
        logger.error("Certifique-se de que a variável de ambiente GOOGLE_API_KEY está configurada corretamente no arquivo .env.")
        logger.error("Se você pretende usar OpenAI, a configuração do modelo de embeddings e a dependência precisam ser ajustadas.")
        return # Sair se o vector store não puder ser criado

    # --- Teste do Pipeline RAG ---
    logger.info("--- INICIANDO TESTE DO PIPELINE RAG ---")
    # A inicialização dos componentes RAG em rag_pipeline.py já lida com a chave da API.
    initialize_rag_components()
    
    rag_app = create_rag_graph()

    test_question = "What is Task Decomposition?"
    logger.info(f"---TESTANDO PIPELINE RAG COM A PERGUNTA: '{test_question}'---")

    # Executar o grafo
    final_state = rag_app.invoke({"question": test_question})

    # Exibir o contexto recuperado
    logger.info("---CONTEXTO RECUPERADO---")
    for i, doc in enumerate(final_state["context"]):
        logger.info(f"Documento {i+1} (parcial): {doc.page_content[:300]}...")

    # Exibir a resposta gerada
    logger.info("---RESPOSTA GERADA---")
    logger.info(final_state["answer"])

if __name__ == "__main__":
    main()
