from dotenv import load_dotenv
import os

from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages

from langgraph.graph import StateGraph, END

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
)

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool


# =========================================================
# ENV
# =========================================================

load_dotenv()

PDF_PATH = "Stock_Market_Performance_2024.pdf"
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "stock_market_2024"


# =========================================================
# LLM
# =========================================================

llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

try:
    test_response = llm.invoke(
        "Hello Groq! Tell me a very short joke about fast chips."
    )
    print("LLM Connected Successfully:")
    print(test_response.content)
except Exception as e:
    print(f"LLM Connection Error: {e}")
    raise


# =========================================================
# EMBEDDINGS
# =========================================================

# Groq does not provide embeddings endpoint
# so we use local HuggingFace embeddings

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# =========================================================
# LOAD OR CREATE VECTORSTORE
# =========================================================

def get_vectorstore():
    """
    Production-ready logic:
    - If Chroma DB already exists → load it
    - Else → create it from PDF
    """

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF file not found: {PDF_PATH}")

    # If vector DB already exists, load it
    if os.path.exists(PERSIST_DIRECTORY):
        print("Loading existing ChromaDB...")

        return Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )

    # Else create fresh vector DB
    print("Creating new ChromaDB from PDF...")

    loader = PyPDFLoader(PDF_PATH)

    try:
        docs = loader.load()
        print(f"PDF loaded successfully with {len(docs)} pages")
    except Exception as e:
        print(f"PDF Loading Error: {e}")
        raise

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = text_splitter.split_documents(docs)

    try:
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY,
            collection_name=COLLECTION_NAME,
        )

        print("ChromaDB created and persisted successfully!")

        return vectorstore

    except Exception as e:
        print(f"Vectorstore Creation Error: {e}")
        raise


vectorstore = get_vectorstore()

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# =========================================================
# TOOL
# =========================================================

@tool
def retrieve_docs(question: str):
    """
    Retrieve relevant stock market information
    from the Stock Market Performance 2024 PDF.
    """

    try:
        docs = retriever.invoke(question)

        if not docs:
            return (
                "I found no relevant information in the "
                "Stock Market Performance 2024 document."
            )

        results = []

        for i, doc in enumerate(docs):
            page = doc.metadata.get("page", "Unknown")

            results.append(
                f"""
Document {i + 1}
Page Number: {page}

Content:
{doc.page_content}
"""
            )

        return "\n\n".join(results)

    except Exception as e:
        return f"Retriever tool error: {str(e)}"


tools = [retrieve_docs]

llm = llm.bind_tools(tools)


# =========================================================
# STATE
# =========================================================

class AgentState(TypedDict):
    messages: Annotated[
        Sequence[BaseMessage],
        add_messages
    ]


# =========================================================
# SYSTEM PROMPT
# =========================================================

system_prompt = """
You are an intelligent AI assistant specialized in answering
questions about Stock Market Performance in 2024.

STRICT RULES:

1. You MUST use the retriever tool whenever the answer depends
   on information from the PDF.

2. Never hallucinate or invent information.

3. Only answer using retrieved document context.

4. If the answer is not found, clearly say:
   "I could not find this information in the document."

5. Always cite:
   - page number
   - specific document evidence

6. You may call the retriever tool multiple times if needed.

7. Keep answers professional, concise, and accurate.
"""


# =========================================================
# NODE: CALL LLM
# =========================================================

def call_llm(state: AgentState):
    """
    LLM decides:
    - answer directly
    - OR call tool(s)
    """

    messages = [
        SystemMessage(content=system_prompt)
    ] + list(state["messages"])

    response = llm.invoke(messages)

    return {
        "messages": [response]
    }


# =========================================================
# NODE: TOOL EXECUTION
# =========================================================

def take_action(state: AgentState):
    """
    Handles MULTIPLE tool calls (production fix)
    """

    response = state["messages"][-1]

    if not response.tool_calls:
        return {"messages": []}

    tools_by_name = {
        tool.name: tool for tool in tools
    }

    tool_messages = []

    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        try:
            print(f"Executing Tool: {tool_name}")

            if tool_name not in tools_by_name:
                result = f"Tool '{tool_name}' not found."

            else:
                result = tools_by_name[tool_name].invoke(tool_args)

        except Exception as e:
            result = f"Tool execution error: {str(e)}"

        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call_id
            )
        )

    return {
        "messages": tool_messages
    }


# =========================================================
# ROUTER
# =========================================================

def should_continue(state: AgentState):
    """
    If LLM requested tool(s)
    → go to tool node

    Else
    → finish
    """

    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "retriever_agent"

    return END


# =========================================================
# GRAPH
# =========================================================

graph = StateGraph(AgentState)

graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.set_entry_point("llm")

graph.add_conditional_edges(
    "llm",
    should_continue
)

graph.add_edge(
    "retriever_agent",
    "llm"
)

app = graph.compile()


# =========================================================
# RUN AGENT
# =========================================================

def running_agent():
    print("\nStock Market RAG Agent Started")
    print("Type 'exit' to quit\n")

    while True:
        question = input("Ask your question: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            response = app.invoke({
                "messages": [
                    HumanMessage(content=question)
                ]
            })

            print("\nFinal Answer:\n")
            print(response["messages"][-1].content)
            print("\n" + "=" * 80 + "\n")

        except Exception as e:
            print(f"Agent Error: {e}")


if __name__ == "__main__":
    running_agent()