from .schema import DocumentList

def load_docx(file_path: str) -> DocumentList:
    """
    Logic to load and process DOCX files.
    Returns:
        DocumentList: A list of standardized Document dictionaries.
    """
    return [
        {
            "content": "",
            "metadata": {
                "source": "docx",
                "file_name": file_path,
                "author": "",
                "created_at": ""
            }
        }
    ]
