import os
from typing import List, TypedDict, Annotated, Sequence
from operator import add as add_messages

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document as LCDocument
from langgraph.graph import StateGraph, END
import json

# Custom modules
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
# 2. STATE
# =========================================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    documents: List[LCDocument]
    generation: str
    rewrite_query: str
    reranked_documents: List[LCDocument]
    expended_queries: List[str]
    eval_decision: str  # correct / incorrect / ambiguous

# =========================================================
# 3. LLM & VECTORSTORE
# =========================================================

llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

pipeline = IngestionPipeline(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIRECTORY,
    chunk_size=1000,
    chunk_overlap=200
)

def initialize_vectorstore():
    if not os.path.exists(PERSIST_DIRECTORY) and os.path.exists(PDF_PATH):
        print("Initializing Vector DB...")
        return pipeline.run(loader_func=load_pdf, file_path=PDF_PATH)
    else:
        print("Loading existing ChromaDB...")
        return pipeline.get_vectorstore()

vectorstore = initialize_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def rewrite_query_node(state: AgentState):
    """Rewrite the query to be more specific."""
    print("--- [Node] Rewrite Query ---")

    question = state["messages"][-1].content
    prompt = f"""
    Rewrite this question to be more specific and detailed:

    {question}

    Do not answer the question. Just rewrite it.
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    new_query = response.content.strip()

    print(f"New Query: {new_query}")
    return {"rewrite_query": new_query}


def generate_queries(state: AgentState):
    print("--- [Node] Generate Queries ---")

    query = state["rewrite_query"]
    prompt = f"""
    Generate 4 queries for the following question:

    {query}

    Do not answer the question. Just generate queries.
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        queries = json.loads(response.content)
    except:
        queries = [query]   
    print("Expanded Queries:", queries)
    return { "expended_queries": queries}

# =========================================================
# 4. NODES
# =========================================================

def retrieve(state: AgentState):
    print("--- [Node] RETRIEVE ---")
    queries= state["expended_queries"]
    all_docs=[]
    for query in queries:
        docs=retriever.invoke(query)
        all_docs.extend(docs)
    
    #-----------------------
    # Deduplicate based on metadata
    unique_docs = {}
    for d in all_docs:
        key = (d.page_content, d.metadata.get("source"), d.metadata.get("page"))
        if key not in unique_docs:
            unique_docs[key] = d

    final_docs = list(unique_docs.values())
    
    #-----------------------

    print("Retrieved Documents:", len(all_docs), "| Unique:", len(final_docs))
    #-----------------------

    return {"documents": final_docs}


def rerank_documents(state: AgentState):
    print("--- [Node] Rerank Documents ---")
    documents=state["documents"]
    question= state["messages"][-1].content
    scored_docs=[]
    for doc in documents:
        prompt = f"""
        Rate relevance from 1-10.

        Question:
        {question}

        Document:
        {doc.page_content[:1500]}

        Return ONLY a number.
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        try:
            score = float(response.content.strip())
        except:
            score = 0.0
        
        print(f"  Score: {score}")
        scored_docs.append((score, doc))
        
    # Sort by score
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Top 3
    reranked = [d[1] for d in scored_docs[:3]]
    
    return {"reranked_documents": reranked, "documents": reranked}

# ---------------- CRAG EVALUATOR ---------------- #

def grade_documents(state: AgentState):
    print("--- [Node] CRAG EVALUATOR ---")

    question = state["messages"][-1].content
    documents = state["documents"]

    context = "\n\n".join([d.page_content for d in documents])

    prompt = f"""
    You are a CRAG evaluator.

    Classify the retrieved context:

    - correct → fully answers the question
    - incorrect → irrelevant or useless
    - ambiguous → partially helpful but insufficient

    Question:
    {question}

    Context:
    {context}

    Answer ONLY one word: correct, incorrect, or ambiguous.
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    decision = response.content.strip().lower()

    print(f"Decision: {decision}")

    return {
        "documents": documents,
        "eval_decision": decision
    }

# ---------------- WEB SEARCH ---------------- #

def web_search(state: AgentState):
    print("--- [Node] WEB SEARCH ---")

    question = state["messages"][-1].content
    decision = state.get("eval_decision", "")

    results = load_tavily(question)
    if not results:
        results = load_duckduckgo(question)

    new_docs = [
        LCDocument(page_content=d["content"], metadata=d["metadata"])
        for d in results
    ]

    # CRAG behavior
    if decision == "incorrect":
        return {"documents": new_docs}

    elif decision == "ambiguous":
        return {"documents": state["documents"] + new_docs}

    return {"documents": new_docs}

# ---------------- GENERATION ---------------- #

def generate(state: AgentState):
    print("--- [Node] GENERATE ---")

    question = state["messages"][-1].content
    docs = state["documents"]

    context = "\n\n".join([d.page_content for d in docs])

    decision = state.get("eval_decision", "")
    if decision == "correct":
        source = "INTERNAL KNOWLEDGE"
    else:
        source = "HYBRID (INTERNAL + WEB)"

    system_prompt = f"""
    You are an expert financial assistant.

    Source: {source}

    Use the context to answer the question.
    If insufficient, say so clearly.

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
# 5. ROUTER
# =========================================================

def router(state: AgentState):
    decision = state.get("eval_decision", "").strip().lower()

    if decision == "correct":
        return "generate"

    elif decision in ["incorrect", "ambiguous"]:
        return "web_search"

    return "generate"

# =========================================================
# 6. GRAPH
# =========================================================

workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade_documents)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)
workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("generate_queries", generate_queries)
workflow.add_node("rerank", rerank_documents)

# Entry
workflow.set_entry_point("rewrite_query")

# Flow
workflow.add_edge("rewrite_query", "generate_queries")
workflow.add_edge("generate_queries", "retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "grade")

workflow.add_conditional_edges(
    "grade",
    router,
    {
        "generate": "generate",
        "web_search": "web_search"
    }
)

workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# =========================================================
# 7. RUNNER
# =========================================================

def run():
    print("\nCRAG RAG Agent Running...\n(Type 'exit' to quit)\n")

    while True:
        q = input("Ask: ").strip()

        if q.lower() in ["exit", "quit"]:
            break

        result = app.invoke({
            "messages": [HumanMessage(content=q)]
        })

        print("\nAnswer:\n", result["generation"])
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    run()