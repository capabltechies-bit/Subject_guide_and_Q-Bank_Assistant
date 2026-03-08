"""
vector_store.py
In-memory FAISS vector store using Google Gemini embeddings.
"""

import os
import numpy as np
import faiss
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# ── In-memory stores ──────────────────────────────────────────────────────────
chunks_store   = []   # actual text of each chunk
metadata_store = []   # source filename + content type for each chunk
faiss_index    = None # the FAISS search index


# ── Embedding ─────────────────────────────────────────────────────────────────
def embed_text(text):
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text
    )
    return result["embedding"]


# ── Indexing ──────────────────────────────────────────────────────────────────
def add_documents(processed_docs: list) -> None:
    """Embed all chunks from a list of processed document dicts and add to FAISS."""
    global faiss_index

    new_chunks, new_meta = [], []

    for doc in processed_docs:
        for chunk in doc["chunks"]:
            new_chunks.append(chunk)
            new_meta.append({
                "source":       doc["filename"],
                "content_type": doc["content_type"],
            })

    if not new_chunks:
        print("No chunks to add!")
        return

    print(f"Embedding {len(new_chunks)} chunks...")
    vectors = [embed_text(c) for c in new_chunks]
    matrix  = np.array(vectors, dtype="float32")

    if faiss_index is None:
        dim = matrix.shape[1]          # 768 for text-embedding-004
        faiss_index = faiss.IndexFlatL2(dim)

    faiss_index.add(matrix)
    chunks_store.extend(new_chunks)
    metadata_store.extend(new_meta)
    print(f"✅ Total chunks indexed: {len(chunks_store)}")


# ── Search ────────────────────────────────────────────────────────────────────
def search(query: str, k: int = 5) -> list:
    """Find the k most relevant chunks for a query string."""
    if faiss_index is None or len(chunks_store) == 0:
        return []

    query_vector = np.array([embed_text(query)], dtype="float32")
    distances, indices = faiss_index.search(query_vector, min(k, len(chunks_store)))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx >= 0:
            results.append({
                "text":         chunks_store[idx],
                "source":       metadata_store[idx]["source"],
                "content_type": metadata_store[idx]["content_type"],
                "score":        float(dist),
            })
    return results


# ── Stats (used by Streamlit sidebar) ────────────────────────────────────────
def get_stats() -> dict:
    sources = list({m["source"] for m in metadata_store})
    return {
        "documents": len(sources),
        "chunks":    len(chunks_store),
        "sources":   sources,
    }