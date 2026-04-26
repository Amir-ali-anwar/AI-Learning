from .schema import DocumentList

def load_github(repo_url: str) -> DocumentList:
    """
    Logic to load and process data from GitHub repositories.
    Returns:
        DocumentList: A list of standardized Document dictionaries.
    """
    return [
        {
            "content": "",
            "metadata": {
                "source": "github",
                "repo": repo_url,
                "file_name": repo_url,
                "author": "GitHub",
                "created_at": ""
            }
        }
    ]
