from langchain_community.tools import DuckDuckGoSearchRun
from .schema import DocumentList

def load_duckduckgo(query: str) -> DocumentList:
    """
    Search the internet using DuckDuckGo (Free, no API key required).
    """
    try:
        search = DuckDuckGoSearchRun()
        result = search.invoke(query)
        
        return [
            {
                "content": result,
                "metadata": {
                    "source": "duckduckgo",
                    "url": "https://duckduckgo.com",
                    "file_name": "web_search",
                    "author": "Internet"
                }
            }
        ]
    except Exception as e:
        print(f"DuckDuckGo Search Error: {e}")
        return []
