import os
import re
from typing import List, TypedDict, Annotated, Sequence
from operator import add as add_messages

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document as LCDocument
from langgraph.graph import StateGraph, END

# Import our new components
from ingestion import IngestionPipeline
from connectors import load_pdf, load_tavily, load_duckduckgo

# =========================================================
# 1. ENV & CONFIG
# =========================================================

load_dotenv()

PDF_PATH = "Stock_Market_Performance_2024.pdf"
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "stock_market_2024"

# =========================================================
# 2. STATE DEFINITION
# =========================================================

class AgentState(TypedDict):
    """
    Standardized state for our Hybrid RAG.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    documents: List[LCDocument]
    generation: str
    search_needed: bool

# =========================================================
# 3. INITIALIZE COMPONENTS
# =========================================================

# LLM (Groq)
llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# Ingestion Pipeline
pipeline = IngestionPipeline(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIRECTORY,
    chunk_size=1000,
    chunk_overlap=200
)

# Initialize or Load VectorStore
def initialize_vectorstore():
    if not os.path.exists(PERSIST_DIRECTORY) and os.path.exists(PDF_PATH):
        print("Initializing Vector DB with Ingestion Pipeline...")
        return pipeline.run(loader_func=load_pdf, file_path=PDF_PATH)
    else:
        # Load existing collection
        print("Loading existing ChromaDB...")
        return pipeline.get_vectorstore()

vectorstore = initialize_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# =========================================================
# 4. NODES (Logic Blocks)
# =========================================================

def retrieve(state: AgentState):
    """
    Step 1: Retrieve relevant documents.
    """
    print("--- [Node] RETRIEVING FROM INTERNAL DOCS ---")
    last_message = state["messages"][-1].content
    docs = retriever.invoke(last_message)
    return {"documents": docs, "search_needed": False}

def grade_documents(state: AgentState):
    """
    Step 2: Use LLM to determine if the retrieved documents are actually relevant.
    If zero relevant docs remain, flag for web search.
    """
    print("--- [Node] GRADING DOCUMENTS WITH LLM ---")
    question = state["messages"][-1].content
    documents = state["documents"]
    
    filtered_docs = []
    
    grading_prompt = """You are a grader assessing relevance of a retrieved document to a user question. 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    
    Retrieved Document:
    {doc_content}
    
    User Question: {question}
    
    Answer only 'yes' or 'no':"""

    for d in documents:
        # Ask LLM to grade the document
        messages = [HumanMessage(content=grading_prompt.format(doc_content=d.page_content, question=question))]
        response = llm.invoke(messages)
        score = response.content.strip().lower()
        
        if 'yes' in score:
            print("  - Document: RELEVANT")
            filtered_docs.append(d)
        else:
            print("  - Document: NOT RELEVANT")
    
    search_needed = len(filtered_docs) == 0
    print(f"Total relevant docs: {len(filtered_docs)}. Web search needed: {search_needed}")
    
    return {"documents": filtered_docs, "search_needed": search_needed}

def web_search(state: AgentState):
    """
    Step 3 (Optional): Search the web if internal docs failed.
    """
    print("--- [Node] SEARCHING THE WEB ---")
    question = state["messages"][-1].content
    
    # Try Tavily first, fallback to DuckDuckGo
    web_results = load_tavily(question)
    if not web_results:
        web_results = load_duckduckgo(question)
        
    # Convert to LangChain Documents
    new_docs = [
        LCDocument(page_content=d["content"], metadata=d["metadata"])
        for d in web_results
    ]
    
    return {"documents": new_docs}

def generate(state: AgentState):
    """
    Step 4: Generate the final answer.
    """
    print("--- [Node] GENERATING ---")
    question = state["messages"][-1].content
    docs = state["documents"]
    
    context = "\n\n".join([d.page_content for d in docs])
    source_info = "INTERNAL KNOWLEDGE" if not state.get("search_needed") else "EXTERNAL WEB SEARCH"
    
    system_prompt = f"""You are an expert financial assistant. 
    Information Source: {source_info}
    
    Use the following context to answer the user's question accurately. 
    If the context is insufficient, state that clearly.
    
    CONTEXT:
    {{context}}
    """
    
    messages = [
        SystemMessage(content=system_prompt.format(context=context)),
        HumanMessage(content=question)
    ]
    
    response = llm.invoke(messages)
    return {"generation": response.content}

# =========================================================
# 5. GRAPH CONSTRUCTION
# =========================================================

def decide_to_generate(state: AgentState):
    """
    Router: Should we search the web or generate?
    """
    if state["search_needed"]:
        return "web_search"
    return "generate"

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_docs", grade_documents)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)

# Set Flow
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_docs")

# Conditional Routing
workflow.add_conditional_edges(
    "grade_docs",
    decide_to_generate,
    {
        "web_search": "web_search",
        "generate": "generate"
    }
)

workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

# =========================================================
# 6. RUNNER
# =========================================================

def running_agent():
    print("\nHybrid RAG Agent Started (Internal Docs + Web Search)")
    print("Type 'exit' to quit\n")

    while True:
        question = input("Ask your question: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            # Invoke graph
            result = app.invoke({
                "messages": [HumanMessage(content=question)]
            })

            print("\nFinal Answer:\n")
            print(result["generation"])
            print("\n" + "=" * 80 + "\n")

        except Exception as e:
            print(f"Agent Error: {e}")

if __name__ == "__main__":
    running_agent()