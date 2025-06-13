# ChatBot-LangChain-REG

Este projeto demonstra um chatbot RAG (Retrieval-Augmented Generation) baseado no tutorial do LangChain. Ele carrega artigos de um blog, indexa o conteúdo em um vector store e utiliza o Google Gemini para responder perguntas a partir desse contexto.

## Instalação

1. Clone o repositório e crie um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r rag_chatbot/requirements.txt
```

2. Configure as variáveis de ambiente em um arquivo `.env` na raiz `rag_chatbot/`:

```
GOOGLE_API_KEY= sua-chave-google
LOG_LEVEL=INFO
```

## Uso Rápido

Execute a aplicação Streamlit para testar o chatbot interativamente:

```bash
streamlit run rag_chatbot/src/streamlit_app.py
```

Também é possível testar apenas o pipeline RAG via linha de comando:

```bash
python rag_chatbot/main.py
```

Para uma demonstração automática com perguntas de exemplo utilize o script `scripts/demo.py`:

```bash
python scripts/demo.py
```

## Docker (opcional)

O projeto pode ser executado em contêiner com Docker:

```bash
docker compose up --build
```

A aplicação ficará acessível em `http://localhost:8501`.

## Arquitetura

```
load_documents -> split_documents -> create_vector_store
                \                             /
                 -> LangGraph pipeline (analyze_query -> retrieve -> generate)
```

- **document_loader.py** – captura o HTML e filtra o conteúdo com BeautifulSoup.
- **text_splitter.py** – divide o texto em chunks e marca a seção (início, meio, fim).
- **vector_store.py** – cria o índice Chroma com embeddings do Google.
- **rag_pipeline.py** – monta o grafo LangGraph que liga análise de consulta,
  recuperação e geração.
- **streamlit_app.py** – interface web com streaming de respostas.

## Troubleshooting

- *`GOOGLE_API_KEY` não configurada*: verifique o arquivo `.env`.
- *Sem resposta no Streamlit*: cheque a conexão com a internet e a chave de API.
- *Erros de importação*: confirme que as dependências do `requirements.txt` foram instaladas.

## Testes

Rode os testes unitários para garantir o funcionamento básico:

```bash
pytest
```

