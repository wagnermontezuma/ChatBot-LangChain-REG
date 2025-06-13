from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
import logging
from .logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def get_rag_prompt_template():
    """
    Carrega o prompt RAG do hub e cria uma versão customizada.
    """
    # Carrega o prompt RAG do hub
    base_rag_prompt = hub.pull("rlm/rag-prompt")

    # Cria uma versão customizada do prompt
    # Instruções adicionais:
    # - Instrua o assistente a usar o contexto recuperado
    # - Limite respostas a 3 frases máximo
    # - Instrua para dizer "não sei" quando não souber
    # - Mantenha respostas concisas
    custom_template = """Você é um assistente útil e informativo. Use o seguinte contexto recuperado para responder à pergunta.
Sua resposta deve ser concisa, com no máximo 3 frases. Se você não souber a resposta, diga "Não sei".

Contexto: {context}
Pergunta: {question}
"""
    
    # Combina o template base com as instruções customizadas
    # Uma forma de fazer isso é criar um novo ChatPromptTemplate
    # que incorpore as instruções.
    # Para simplificar e garantir que as instruções sejam seguidas,
    # vamos criar um novo template com as instruções explícitas.
    custom_rag_prompt = ChatPromptTemplate.from_template(custom_template)
    
    return custom_rag_prompt

if __name__ == "__main__":
    prompt = get_rag_prompt_template()
    
    # Teste o template de prompt com dados de exemplo
    example_context = "A inteligência artificial é um campo da ciência da computação que se dedica ao estudo e ao desenvolvimento de máquinas e programas capazes de simular o raciocínio humano."
    example_question = "O que é inteligência artificial?"
    
    formatted_prompt = prompt.format(context=example_context, question=example_question)
    logger.info("--- Template de Prompt Customizado ---")
    logger.info(formatted_prompt)

    example_context_no_answer = "O céu é azul."
    example_question_no_answer = "Qual a cor do sol?"
    formatted_prompt_no_answer = prompt.format(context=example_context_no_answer, question=example_question_no_answer)
    logger.info("\n--- Template de Prompt Customizado (Cenário 'não sei') ---")
    logger.info(formatted_prompt_no_answer)
