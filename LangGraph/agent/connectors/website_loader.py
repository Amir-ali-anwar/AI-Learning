from .schema import DocumentList

def load_website(url: str) -> DocumentList:
    """
    Logic to load and process data from websites (web scraping).
    Returns:
        DocumentList: A list of standardized Document dictionaries.
    """
    return [
        {
            "content": "",
            "metadata": {
                "source": "website",
                "url": url,
                "file_name": url,
                "author": "Web Scraper",
                "created_at": ""
            }
        }
    ]
