try:
    from langchain_core.messages import (
        HumanMessage,
        AIMessage,
        ToolMessage,
        SystemMessage,
    )
except Exception:  # pragma: no cover - fallback for test stubs
    class BaseMessage:
        def __init__(self, content: str = "", **kwargs):
            self.content = content
            self.additional_kwargs = kwargs

    class HumanMessage(BaseMessage):
        pass

    class AIMessage(BaseMessage):
        pass

    class ToolMessage(BaseMessage):
        def __init__(self, content: str = "", tool_call_id=None, **kwargs):
            super().__init__(content, **kwargs)
            self.tool_call_id = tool_call_id

    class SystemMessage(BaseMessage):
        pass
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState

from .vector_store import retrieve


def query_or_respond(state: MessagesState, llm):
    """Call the LLM which may return a tool call."""
    model = llm.bind_tools([retrieve])
    response = model.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}


# Node responsible for executing tool calls
tools = ToolNode([retrieve])


def generate(state: MessagesState, llm, prompt=None):
    """Generate the assistant answer using retrieved context."""
    messages = state["messages"]

    docs = []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            docs.extend(msg.additional_kwargs.get("docs", []))
            break

    context = "\n\n".join(d.page_content for d in docs)

    system_instruction = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
    )
    system_msg = SystemMessage(content=f"{system_instruction}\n\n{context}")

    convo_msgs = [
        m
        for m in messages
        if isinstance(m, (HumanMessage, SystemMessage))
        or (
            isinstance(m, AIMessage)
            and not m.additional_kwargs.get("tool_calls")
        )
    ]
    prompt_messages = convo_msgs + [system_msg]

    answer = llm.invoke(prompt_messages)
    return {"messages": messages + [AIMessage(content=answer)]}
