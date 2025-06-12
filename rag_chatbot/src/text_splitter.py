from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_documents(documents: list[Document]):
    """
    Divide documentos em chunks usando RecursiveCharacterTextSplitter e adiciona metadados de seção.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)

    # Adicionar metadados de seção
    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        section = "middle"
        if i < total_chunks / 3:
            section = "beginning"
        elif i > 2 * total_chunks / 3:
            section = "end"
        chunk.metadata["section"] = section
    
    return chunks

if __name__ == "__main__":
    # Exemplo de uso (normalmente seria chamado por outro módulo)
    # Para testar, vamos criar um documento dummy
    dummy_doc = Document(page_content="Este é um texto de exemplo para testar o text splitter. Ele precisa ser longo o suficiente para ser dividido em múltiplos chunks. Vamos adicionar mais conteúdo para garantir que o chunk_size e chunk_overlap sejam efetivos. " * 20)
    
    chunks = split_documents([dummy_doc])
    print(f"Total de chunks criados: {len(chunks)}")
    if chunks:
        print(f"Primeiro chunk (parcial): {chunks[0].page_content[:200]}...")
        print(f"Segundo chunk (parcial): {chunks[1].page_content[:200]}...")
