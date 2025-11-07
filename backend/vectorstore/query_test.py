import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(MODEL_NAME)

# Load FAISS index + metadata
index = faiss.read_index("data/vector_index/financial_index.faiss")
with open("data/vector_index/metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

def search(query, top_k=5):
    query_vector = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vector, top_k)
    print(f"\n🔎 Query: {query}\n")
    for i, idx in enumerate(indices[0]):
        result = metadata[idx]
        print(f"Result {i+1}: {result['file']} ({result['source']})")
        print(f"Chunk ID: {result['chunk_id']} | Distance: {distances[0][i]:.2f}")
        print("-" * 60)

if __name__ == "__main__":
    search("Compare Apple and Nvidia R&D expenditure trends")
