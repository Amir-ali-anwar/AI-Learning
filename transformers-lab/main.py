import numpy as np
from src.embeddings import load_model, get_embeddings
from src.similarity import cosine_similarity

def main():
    # 1. Setup
    model = load_model("all-MiniLM-L6-v2")

    # 2. Sentences
    sentences = [
        # AI
        "Artificial intelligence is transforming the world.",
        "Machine learning enables computers to learn from data.",
        "Deep learning uses neural networks.",

        # Food
        "Pizza is my favorite food.",
        "I love eating pasta.",
        "Burgers taste delicious.",

        # Sports
        "Cricket is very popular in Pakistan.",
        "Football is played worldwide.",
        "Lionel Messi is a famous footballer.",

        # Tech
        "Cloud computing is scalable.",
        "Azure provides AI services.",
        "Kubernetes manages containers.",

        # Random
        "The sky is blue.",
        "I enjoy reading books.",
        "Dogs are loyal animals."
    ]

    # 3. Generate Embeddings
    embeddings = get_embeddings(model, sentences)
    print(f"Embeddings shape: {embeddings.shape}")

    # 4. Similarity Matrix
    n = len(embeddings)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            similarity_matrix[i][j] = cosine_similarity(embeddings[i], embeddings[j])
    
    print("Similarity matrix calculated.")

    # 5. Query Search
    query = "I enjoy playing football."
    query_embedding = get_embeddings(model, [query])[0]

    print(f"\nQuery: {query}\n")
    scores = []
    for i, emb in enumerate(embeddings):
        score = cosine_similarity(query_embedding, emb)
        scores.append((sentences[i], score))
        print(f"{sentences[i]} -> {score:.3f}")

if __name__ == "__main__":
    main()
