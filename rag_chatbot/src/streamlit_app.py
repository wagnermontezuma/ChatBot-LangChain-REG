import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Adicionar o diretório pai (rag_chatbot) ao sys.path para importações relativas
# Isso é necessário quando o script é executado de dentro de um subdiretório (src)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Importar funções e componentes do pipeline RAG
from src.rag_pipeline import initialize_rag_components, create_rag_graph
from src.advanced_features import stream_rag_response # Importar a função de streaming

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv(dotenv_path='rag_chatbot/.env')

# --- Configuração da Página Streamlit ---
st.set_page_config(page_title="RAG Chatbot with LangChain", layout="wide")
st.title("🤖 RAG Chatbot with LangChain")

# --- Inicialização do Pipeline RAG (apenas uma vez) ---
@st.cache_resource
def setup_rag_pipeline():
    """
    Configura e retorna o pipeline RAG e seus componentes.
    Usa st.cache_resource para evitar re-execução a cada interação.
    """
    try:
        components = initialize_rag_components()
        # Passar os componentes para create_rag_graph
        rag_app = create_rag_graph(
            vector_store=components["vector_store"],
            llm=components["llm"],
            rag_prompt=components["rag_prompt"],
            structured_llm=components["structured_llm"]
        )
        return rag_app, components # Retorna o app e os componentes
    except Exception as e:
        st.error(f"Erro ao inicializar o pipeline RAG: {e}")
        st.stop()

rag_app, rag_components = setup_rag_pipeline() # Captura o app e os componentes

# Extrai os componentes para uso
llm = rag_components["llm"]
rag_prompt = rag_components["rag_prompt"]
vector_store = rag_components["vector_store"] # Pode ser útil para depuração ou futuras features

# --- Gerenciamento do Histórico de Mensagens ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.header("Sobre o Chatbot RAG")
    st.info(
        "Este é um chatbot RAG (Retrieval-Augmented Generation) construído com LangChain e LangGraph. "
        "Ele carrega documentos de um blog, divide-os em chunks, indexa-os em um vector store, "
        "e usa um modelo de linguagem (Google Gemini) para responder a perguntas com base no contexto recuperado."
    )
    st.subheader("Configurações (Ainda não implementadas)")
    # st.slider("Temperatura do LLM", 0.0, 1.0, 0.7) # Exemplo de configuração
    
    st.subheader("Exemplos de Perguntas")
    example_questions = [
        "O que é Task Decomposition?",
        "Fale sobre os métodos de decomposição de tarefas.",
        "Quais são as conclusões sobre a decomposição de tarefas?",
        "Como a reflexão ajuda os agentes?",
        "O que é Tree of Thoughts?"
    ]
    for q in example_questions:
        if st.button(q):
            st.session_state.messages.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q) # Exibir a pergunta do exemplo imediatamente

            # Processar a pergunta com streaming
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                try:
                    # Para usar stream_rag_response, precisamos do rag_app e da pergunta
                    # A função stream_rag_response agora recebe llm, rag_prompt e vector_store
                    for chunk in stream_rag_response(q, rag_app, llm, rag_prompt, vector_store):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"Ocorreu um erro: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Ocorreu um erro: {e}"})
            st.rerun()

    if st.button("Limpar Histórico"):
        st.session_state.messages = []
        st.rerun()

# --- Exibição do Histórico de Conversas ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Input de Texto para Perguntas do Usuário ---
if prompt := st.chat_input("Faça sua pergunta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            # Invocar o pipeline RAG com streaming
            # A função stream_rag_response agora recebe llm, rag_prompt e vector_store
            for chunk in stream_rag_response(prompt, rag_app, llm, rag_prompt, vector_store):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # Opcional: exibir contexto recuperado (requer uma forma de obter o contexto do stream)
            # Como stream_rag_response retorna apenas a resposta, para exibir o contexto,
            # precisaríamos de uma invocação separada ou modificar stream_rag_response para retornar
            # o estado completo ou o contexto. Por simplicidade, vou remover a exibição do contexto
            # no modo de streaming, a menos que o usuário peça explicitamente para refatorar.
            # Ou, podemos fazer uma invocação síncrona para obter o contexto após o streaming.
            # Para manter a simplicidade, vou remover a exibição do contexto por enquanto.
            # Se o usuário quiser, podemos refatorar stream_rag_response para retornar o estado completo.
            # Para este passo, vou remover a exibição do contexto no chat_input.

        except Exception as e:
            st.error(f"Ocorreu um erro ao processar sua pergunta: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Ocorreu um erro: {e}"})
    st.rerun()
