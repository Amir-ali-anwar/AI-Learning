from typing import TypedDict, List, Optional, Dict, Any

class DocumentMetadata(TypedDict, total=False):
    """
    Standard metadata format for all connectors.
    """
    source: str
    file_name: Optional[str]
    page: Optional[int]
    author: Optional[str]
    created_at: Optional[str]
    url: Optional[str]
    repo: Optional[str]
    # Add any other relevant fields here

class Document(TypedDict):
    """
    Standard output document format for all connectors.
    """
    content: str
    metadata: DocumentMetadata

# Type alias for a list of documents
DocumentList = List[Document]
