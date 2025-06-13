from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup, SoupStrainer
from rag_chatbot.src.config import DOC_URL

def load_documents(url: str | None = None):
    """
    Carrega documentos de uma URL usando WebBaseLoader e BeautifulSoup.
    """
    if url is None:
        url = DOC_URL

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
    import argparse

    parser = argparse.ArgumentParser(description="Carrega documentos de uma URL")
    parser.add_argument("--url", help="URL dos documentos", default=None)
    args = parser.parse_args()

    docs = load_documents(args.url)
    print(f"Documentos carregados: {len(docs)}")
    if docs:
        print(f"Conteúdo do primeiro documento (parcial): {docs[0].page_content[:500]}...")
