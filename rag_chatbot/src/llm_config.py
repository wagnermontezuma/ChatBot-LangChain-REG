import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv(dotenv_path='rag_chatbot/.env')

def get_chat_model(model_name: str = "gemini-1.5-pro", temperature: float = 0.7, api_key: str = None):
    """
    Configura e retorna o modelo de chat do Google Gemini.
    """
    if api_key is None:
        api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("A variável de ambiente GOOGLE_API_KEY não está configurada ou não foi fornecida.")
        
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=api_key)
    return llm

if __name__ == "__main__":
    try:
        llm = get_chat_model()
        print(f"Modelo de chat configurado: {llm.model_name}")
        # Exemplo de como usar o modelo (requer GOOGLE_API_KEY válida)
        # response = llm.invoke("Olá, como você está?")
        # print(f"Resposta do LLM: {response.content}")
    except ValueError as e:
        print(f"Erro de configuração do LLM: {e}")
    except Exception as e:
        print(f"Ocorreu um erro ao testar o LLM: {e}")
        print("Certifique-se de que a GOOGLE_API_KEY está correta e tem permissões.")
