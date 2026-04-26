import os
import re
from typing import List, Callable, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document as LCDocument

# Import the standardized format
try:
    from connectors.schema import DocumentList, Document
except ImportError:
    # Fallback if running from a different context
    from .connectors.schema import DocumentList, Document

class IngestionPipeline:
    """
    Standardized Ingestion Pipeline:
    Extract -> Clean -> Chunk -> Embed -> Store
    
    This pipeline is reusable for any source that follows the 
    standardized DocumentList format.
    """

    def __init__(
        self, 
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db",
        collection_name: str = "rag_collection",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def extract(self, loader_func: Callable, *args, **kwargs) -> DocumentList:
        """
        Step 1: Extract data using a standardized connector.
        """
        print(f"--- [1/5] Extracting data ---")
        docs = loader_func(*args, **kwargs)
        print(f"Extracted {len(docs)} document units.")
        return docs

    def clean(self, docs: DocumentList) -> DocumentList:
        """
        Step 2: Clean the text content of documents.
        """
        print(f"--- [2/5] Cleaning text ---")
        for doc in docs:
            text = doc["content"]
            # Remove redundant whitespaces and newlines
            text = re.sub(r'\s+', ' ', text).strip()
            # Remove non-printable characters if any
            text = "".join(char for char in text if char.isprintable())
            doc["content"] = text
        return docs

    def chunk(self, docs: DocumentList) -> List[LCDocument]:
        """
        Step 3: Chunk the documents into smaller segments.
        """
        print(f"--- [3/5] Chunking documents ---")
        # Convert to LangChain Document format for splitting
        lc_docs = [
            LCDocument(page_content=d["content"], metadata=d["metadata"])
            for d in docs if d["content"]
        ]
        splits = self.text_splitter.split_documents(lc_docs)
        print(f"Created {len(splits)} chunks.")
        return splits

    def embed_and_store(self, splits: List[LCDocument]) -> Chroma:
        """
        Step 4 & 5: Embed the chunks and store them in the Vector DB.
        """
        if not splits:
            return self.get_vectorstore()

        print(f"--- [4/5] Embedding & [5/5] Storing in {self.collection_name} ---")
        
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        
        print("Ingestion Pipeline completed successfully.")
        return vectorstore

    def get_vectorstore(self) -> Chroma:
        """
        Simply loads and returns the existing vectorstore.
        """
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )

    def run(self, loader_func: Optional[Callable] = None, docs: Optional[DocumentList] = None, *args, **kwargs) -> Chroma:
        """
        Runs the full pipeline. 
        You can either provide a loader function or a list of already extracted documents.
        """
        # Step 1: Extract
        if loader_func:
            raw_docs = self.extract(loader_func, *args, **kwargs)
        elif docs is not None:
            raw_docs = docs
        else:
            raise ValueError("Either loader_func or docs must be provided.")

        # Step 2: Clean
        cleaned_docs = self.clean(raw_docs)

        # Step 3: Chunk
        splits = self.chunk(cleaned_docs)

        # Step 4 & 5: Embed & Store
        return self.embed_and_store(splits)

if __name__ == "__main__":
    # Example usage (dry run or testing)
    # from connectors.pdf_loader import load_pdf
    # pipeline = IngestionPipeline()
    # vectorstore = pipeline.run(loader_func=load_pdf, file_path="sample.pdf")
    pass
