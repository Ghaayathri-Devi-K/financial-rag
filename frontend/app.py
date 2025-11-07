import os
import json
import faiss
import numpy as np
import streamlit as st
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# =========================================================
# CONFIGURATION
# =========================================================
load_dotenv()
INDEX_PATH = "data/vector_index/financial_index.faiss"
META_PATH  = "data/vector_index/metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"

# =========================================================
# DEVICE SETUP (FIX FOR META TENSOR BUG)
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"🧠 Using device: `{device}`")

# Load embedding model safely
try:
    embedder = SentenceTransformer(MODEL_NAME, device=device)
    print(f"✅ Loaded SentenceTransformer on {device}")
except Exception as e:
    st.error(f"❌ Failed to load SentenceTransformer: {e}")
    st.stop()

# Initialize Groq client
try:
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )
    print("✅ Groq client initialized.")
except Exception as e:
    st.error(f"❌ Failed to initialize Groq client: {e}")
    st.stop()

# Load FAISS index
try:
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"✅ Loaded FAISS index: {INDEX_PATH}")
except Exception as e:
    st.error(f"❌ Failed to load FAISS index or metadata: {e}")
    st.stop()


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def get_context(query, top_k=5):
    """Return top-k retrieved chunks with metadata."""
    qvec = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(qvec, top_k)
    results = []
    for idx in I[0]:
        meta = metadata[idx]
        file_path = f"data/chunks/{meta['source']}/{meta['file']}"
        if not os.path.exists(file_path):
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        snippet = data[meta["chunk_id"]]["text"]
        results.append((meta["source"], meta["file"], snippet))
    return results


def generate_answer(query, context_text):
    """Call Groq Llama model for final synthesis."""
    system_prompt = (
        "You are a financial analysis assistant. "
        "Use the CONTEXT to answer the QUESTION factually and concisely. "
        "Always cite sources like [sec], [news], or [market]."
    )
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CONTEXT:\n{context_text}\n\nQUESTION:\n{query}"}
            ],
            temperature=0.2,
            max_tokens=600
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error while querying Groq API: {e}"


# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="Financial RAG Assistant", page_icon="📈", layout="wide")

st.title("📊 Financial RAG Assistant")
st.caption("Ask questions using real data from SEC filings, market feeds and news.")

query = st.text_input("🔎 Ask a question:", placeholder="Compare Apple and Nvidia R&D trends over 2023 …")

if st.button("Run Analysis") and query.strip():
    with st.spinner("Retrieving context and generating answer 💭 …"):
        retrieved = get_context(query)
        context_text = "\n\n".join([r[2] for r in retrieved])
        answer = generate_answer(query, context_text)

    st.subheader("🧭 AI Answer")
    st.write(answer)

    with st.expander("📂 Retrieved Sources (Top 5)"):
        for src, file, snip in retrieved:
            st.markdown(f"**Source:** `{src}` | **File:** `{file}`")
            st.write(snip)
            st.divider()
