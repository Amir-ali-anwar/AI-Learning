from .schema import DocumentList

def load_sql(connection_string: str, query: str) -> DocumentList:
    """
    Logic to load and process data from SQL databases.
    Returns:
        DocumentList: A list of standardized Document dictionaries.
    """
    return [
        {
            "content": "",
            "metadata": {
                "source": "sql",
                "file_name": f"query_result_{query[:10]}",
                "author": "Database",
                "created_at": ""
            }
        }
    ]
