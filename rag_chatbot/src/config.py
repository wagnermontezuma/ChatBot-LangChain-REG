import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env localizado em rag_chatbot/
load_dotenv(dotenv_path='rag_chatbot/.env')

# URL padrão do documento para indexação
DOC_URL = os.getenv('DOC_URL', 'https://lilianweng.github.io/posts/2023-06-23-agent/')
