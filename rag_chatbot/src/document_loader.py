from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup, SoupStrainer
import logging
from .logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def load_documents(url: str = "https://lilianweng.github.io/posts/2023-06-23-agent/"):
    """
    Carrega documentos de uma URL usando WebBaseLoader e BeautifulSoup.
    """
    loader = WebBaseLoader(
        web_path=url,
        bs_kwargs=dict(
            parse_only=SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    return loader.load()

if __name__ == "__main__":
    docs = load_documents()
    logger.info(f"Documentos carregados: {len(docs)}")
    if docs:
        logger.info(f"Conteúdo do primeiro documento (parcial): {docs[0].page_content[:500]}...")
