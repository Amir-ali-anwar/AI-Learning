from .schema import DocumentList

def load_csv(file_path: str) -> DocumentList:
    """
    Logic to load and process CSV files.
    Returns:
        DocumentList: A list of standardized Document dictionaries.
    """
    return [
        {
            "content": "",
            "metadata": {
                "source": "csv",
                "file_name": file_path,
                "author": "",
                "created_at": ""
            }
        }
    ]
