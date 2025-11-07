import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------
# CONFIGURATION
# ---------------------------
CHUNKS_DIR = "data/chunks"
INDEX_DIR = "data/vector_index"
MODEL_NAME = "all-MiniLM-L6-v2"  # Fast + reliable embedding model

os.makedirs(INDEX_DIR, exist_ok=True)

# Load embedding model
print("🔍 Loading embedding model...")
embedder = SentenceTransformer(MODEL_NAME)


def load_chunks_from_dir(root_dir):
    """Load all text chunks into memory"""
    texts, metadata = [], []

    for root, _, files in os.walk(root_dir):
        for file in files:
            if not file.endswith(".json"):
                continue
            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                for chunk in data:
                    texts.append(chunk["text"])
                    metadata.append({
                        "file": file,
                        "chunk_id": chunk["chunk_id"],
                        "source": os.path.relpath(root, CHUNKS_DIR)
                    })
    print(f"📦 Loaded {len(texts)} text chunks.")
    return texts, metadata


def build_faiss_index(texts, metadata):
    """Create FAISS index and save it to disk"""
    print("⚙️ Generating embeddings...")
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save embeddings and metadata
    faiss.write_index(index, os.path.join(INDEX_DIR, "financial_index.faiss"))
    with open(os.path.join(INDEX_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ FAISS index built and saved ({len(texts)} chunks indexed).")


if __name__ == "__main__":
    texts, metadata = load_chunks_from_dir(CHUNKS_DIR)
    build_faiss_index(texts, metadata)
    print("🎯 Embedding + Indexing Complete!")
