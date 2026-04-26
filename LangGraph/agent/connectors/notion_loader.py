from .schema import DocumentList

def load_notion(page_id: str) -> DocumentList:
    """
    Logic to load and process data from Notion pages.
    Returns:
        DocumentList: A list of standardized Document dictionaries.
    """
    return [
        {
            "content": "",
            "metadata": {
                "source": "notion",
                "file_name": page_id,
                "author": "Notion User",
                "created_at": ""
            }
        }
    ]
