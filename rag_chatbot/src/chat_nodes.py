from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState

from .vector_store import retrieve


def query_or_respond(state: MessagesState, llm):
    model = llm.bind_tools([retrieve])
    response = model.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode([retrieve])

def tools(state: MessagesState):
    return tool_node.invoke(state)


def generate(state: MessagesState, llm, prompt):
    messages = state["messages"]
    docs = []
    if messages and isinstance(messages[-1], ToolMessage):
        docs = messages[-1].additional_kwargs.get("docs", [])
    question = next((m.content for m in messages if isinstance(m, HumanMessage)), "")
    context = "\n\n".join(d.page_content for d in docs)
    formatted = prompt.format_messages(context=context, question=question)
    answer = llm.invoke(formatted)
    return {"messages": [AIMessage(content=answer)]}
