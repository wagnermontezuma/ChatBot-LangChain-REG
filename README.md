# ChatBot LangChain RAG

Este projeto demonstra um pipeline de Retrieval-Augmented Generation (RAG) utilizando LangChain e LangGraph. Documentos de uma fonte web são carregados, divididos em trechos, indexados em um vector store e consultados através de um LLM da Google (Gemini). Há também uma interface em Streamlit para interação via chat.

## Dependências
As dependências estão listadas em [`rag_chatbot/requirements.txt`](rag_chatbot/requirements.txt). Recomenda-se criar um ambiente virtual e instalar tudo com:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r rag_chatbot/requirements.txt
```

## Arquivo `.env`
Crie um arquivo `rag_chatbot/.env` contendo as seguintes variáveis:

```
GOOGLE_API_KEY=chave_da_api_do_gemini
# Opcional para rastreamento com LangSmith
LANGSMITH_API_KEY=sua_chave_langsmith
LANGSMITH_TRACING=true
```

## Como executar

### Indexação e teste básico
Execute o script principal para carregar documentos, criar o vector store e fazer um teste simples do pipeline:

```bash
python rag_chatbot/main.py
```

### Interface web
Para interagir via Streamlit:

```bash
streamlit run rag_chatbot/src/streamlit_app.py
```

## Rodando os testes
Os testes (quando existirem) podem ser executados com `pytest`:

```bash
pytest
```

