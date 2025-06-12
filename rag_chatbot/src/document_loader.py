from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup, SoupStrainer

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
    print(f"Documentos carregados: {len(docs)}")
    if docs:
        print(f"Conteúdo do primeiro documento (parcial): {docs[0].page_content[:500]}...")
