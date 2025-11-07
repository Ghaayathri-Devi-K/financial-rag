"""
RAG Query Engine – Groq Llama-3 Edition
Author: Ghaayathri Devi Kannan
Description:
This script retrieves relevant chunks from the FAISS vector index,
constructs a context window, and queries Groq's Llama-3-70B model
for natural language synthesis of financial insights.
"""

import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv


# ================================================================
# CONFIGURATION
# ================================================================
load_dotenv()

INDEX_PATH = "data/vector_index/financial_index.faiss"
META_PATH = "data/vector_index/metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

# ================================================================
# INITIALIZE MODELS
# ================================================================
print("🚀 Loading embedding model + Groq client...")

# Load embedding model for semantic retrieval
embedder = SentenceTransformer(MODEL_NAME)

# Initialize Groq client (OpenAI compatible)
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Load FAISS index and metadata
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)


# ================================================================
# HELPER FUNCTIONS
# ================================================================
def get_context_text(query: str, top_k: int = TOP_K) -> str:
    """
    Retrieve top-k relevant context snippets from FAISS index.
    """
    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)

    context_blocks = []
    for i, idx in enumerate(indices[0]):
        meta = metadata[idx]
        file_path = f"data/chunks/{meta['source']}/{meta['file']}"
        if not os.path.exists(file_path):
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            chunk_data = json.load(f)
        text_snippet = chunk_data[meta["chunk_id"]]["text"]
        context_blocks.append(f"[{meta['source']}] {text_snippet}")

    if not context_blocks:
        return "No relevant context found."

    return "\n\n".join(context_blocks)


def rag_query_groq(query: str) -> str:
    """
    Perform Retrieval-Augmented Generation using Groq's Llama-3 model.
    """
    context = get_context_text(query)

    system_prompt = (
        "You are a financial analysis assistant. "
        "Use the CONTEXT to answer the QUESTION factually, concisely, "
        "and with reasoning. Always cite sources as [sec], [news], or [market]."
    )

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"}
            ],
            temperature=0.2,
            max_tokens=600
        )

        return completion.choices[0].message.content

    except Exception as e:
        return f"⚠️ Error while querying Groq: {str(e)}"


# ================================================================
# MAIN EXECUTION
# ================================================================
if __name__ == "__main__":
    query = "Compare Apple and Nvidia R&D expenditure trends and investor sentiment."

    print("\n🔎 Querying Groq RAG...")
    answer = rag_query_groq(query)

    print("\n🧭 Answer:\n")
    print(answer)




