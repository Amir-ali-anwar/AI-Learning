import os
from typing import Optional
from langchain_community.tools.tavily_search import TavilySearchResults
from .schema import DocumentList

def load_tavily(query: str, max_results: int = 5) -> DocumentList:
    """
    Search the internet using Tavily (Optimized for LLMs).
    Requires TAVILY_API_KEY in environment variables.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("Warning: TAVILY_API_KEY not found. Skipping Tavily search.")
        return []

    try:
        search = TavilySearchResults(max_results=max_results)
        results = search.invoke(query)
        
        return [
            {
                "content": r["content"],
                "metadata": {
                    "source": "tavily",
                    "url": r["url"],
                    "file_name": "web_search",
                    "author": "Internet"
                }
            }
            for r in results
        ]
    except Exception as e:
        print(f"Tavily Search Error: {e}")
        return []
