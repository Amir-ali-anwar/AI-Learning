from sentence_transformers import SentenceTransformer

def load_model(model_name="all-MiniLM-L6-v2"):
    """
    Loads a SentenceTransformer model.
    """
    return SentenceTransformer(model_name)

def get_embeddings(model, sentences):
    """
    Encodes a list of sentences into embeddings.
    """
    return model.encode(sentences)
