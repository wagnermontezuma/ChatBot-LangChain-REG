# ChatBot-LangChain-REG

Este repositório contém um exemplo simples de integração entre LangChain e recursos de RAG.

## Funcionalidades Avançadas

O módulo `src/advanced_features.py` adiciona otimizações ao chatbot:

- **Streaming de respostas** com medição de tempo e fallback em caso de falha.
- **Modos de invocação síncrono e assíncrono**.
- **Cache de embeddings** e carregamento preguiçoso do vector store.
- **Logging integrado ao LangSmith** e logs estruturados via `logging`.
- **Validação de entrada e tratamento de erros** durante o streaming.

As funções principais podem ser reutilizadas por outras interfaces, como a aplicação Streamlit.

## Testes

Os testes unitários estão localizados no diretório `tests/` e usam `pytest`.
Para executá-los, no diretório raiz do projeto, rode:

```bash
pytest
```
