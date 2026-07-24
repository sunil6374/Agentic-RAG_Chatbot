
import streamlit as st
from langchain_groq import ChatGroq
from typing import Annotated

from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from typing_extensions import TypedDict
from langchain_cohere import CohereEmbeddings




# ----------------------------------------------------
# Load Documents
# ----------------------------------------------------

loader = PyPDFLoader("Class 11 -2nd Holy Qurbana – A Study English.pdf")

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

documents = splitter.split_documents(docs)


# ----------------------------------------------------
# Embeddings + Vector Store
# ----------------------------------------------------

embeddings = embeddings = CohereEmbeddings(
    model="embed-v4.0")

vectorstore = FAISS.from_documents(
    documents,
    embeddings
)

retriever = vectorstore.as_retriever()


# ----------------------------------------------------
# Retriever Tool
# ----------------------------------------------------

@tool
def retrieve(query: str):
    """
    Retrieve relevant documents from the knowledge base.
    """

    docs = retriever.invoke(query)

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


tools = [retrieve]


# ----------------------------------------------------
# LLM
# ----------------------------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=st.secrets["GROQ_API_KEY"],
    temperature=0
)

llm = llm.bind_tools(tools)


# ----------------------------------------------------
# State
# ----------------------------------------------------

class State(TypedDict):
    messages: Annotated[list, add_messages]


# ----------------------------------------------------
# Agent Node
# ----------------------------------------------------

def chatbot(state: State):

    response = llm.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }


# ----------------------------------------------------
# Graph
# ----------------------------------------------------

builder = StateGraph(State)

builder.add_node(
    "chatbot",
    chatbot
)

builder.add_node(
    "tools",
    ToolNode(tools)
)

builder.add_edge(
    START,
    "chatbot"
)

builder.add_conditional_edges(
    "chatbot",
    tools_condition
)

builder.add_edge(
    "tools",
    "chatbot"
)

graph = builder.compile()


# ----------------------------------------------------
# Run
# ----------------------------------------------------

st.title("📄 Agentic RAG")

question = st.text_input("Ask a question")

if st.button("Ask"):

    result = graph.invoke(
        {
            "messages":[
                HumanMessage(content=question)
            ]
        }
    )

    st.write(result["messages"][-1].content)
