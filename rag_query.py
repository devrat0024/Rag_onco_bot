#!/usr/bin/env python3
# =========================================================
# 🧠 Dr. Nova – RAG Query (Short Conversational Replies)
# Author: Devvrat Shukla + GPT-5
# =========================================================

import os, json, faiss, numpy as np, torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

# =========================================================
# ⚙️ Environment Setup
# =========================================================
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    print("🔑 GROQ_API_KEY loaded successfully.")
else:
    print("⚠️ GROQ_API_KEY not found. Running in local-only mode.")

# =========================================================
# 📦 Configuration
# =========================================================
OUTPUT_DIR = "embeddings_biomed"
EMBED_MODEL = "pritamdeka/S-BioBert-snli-multinli-stsb"
GENERATOR_MODEL = "google/flan-t5-large"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🧠 Using device: {device}")

# =========================================================
# 📂 Load FAISS + Data
# =========================================================
def _load_data():
    index_path = os.path.join(OUTPUT_DIR, "pdf_faiss.index")
    text_path = os.path.join(OUTPUT_DIR, "pdf_texts.npy")
    meta_path = os.path.join(OUTPUT_DIR, "pdf_metadata.json")

    if not all(os.path.exists(p) for p in [index_path, text_path, meta_path]):
        raise FileNotFoundError("❌ Missing FAISS or embedding data. Run 'rag_llm.py' first.")

    index = faiss.read_index(index_path)
    texts = np.load(text_path, allow_pickle=True)
    metadata = json.load(open(meta_path, encoding="utf-8"))
    return index, texts, metadata

# =========================================================
# 🔍 Retrieve Relevant Context
# =========================================================
def retrieve_context(query: str, top_k: int = 3):
    index, texts, metadata = _load_data()
    embedder = SentenceTransformer(EMBED_MODEL, device=device)
    q_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, len(texts))

    sem_scores = 1 / (1 + D[0])
    sem_scores = (sem_scores - sem_scores.min()) / (sem_scores.max() - sem_scores.min() + 1e-10)

    bm25 = BM25Okapi([t.lower().split() for t in texts])
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)

    combined = 0.6 * sem_scores + 0.4 * bm25_scores
    top_idx = np.argsort(combined)[::-1][:top_k]

    chosen_texts = [texts[i] for i in top_idx]
    chosen_meta = [metadata[i] for i in top_idx]
    return chosen_texts, chosen_meta

# =========================================================
# 🤖 Generate Short Conversational Answer
# =========================================================
def generate_answer(query: str, context: str, chat_context: str = ""):
    style_instruction = (
        "Reply like a friendly doctor. Keep it very short (1–2 sentences). "
        "Be warm, natural, and empathetic — not robotic or overly formal."
    )

    prompt = f"""
Chat history:
{chat_context[:1000]}

User question: {query}

Relevant info:
{context[:2000]}

Instruction: {style_instruction}
"""

    # Try Groq API first
    if groq_api_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_api_key)
            chat = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a kind, concise medical counselor."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=150,
            )
            answer = chat.choices[0].message.content.strip()
            print("✅ Groq API generated a response.")
            return answer
        except Exception as e:
            print("⚠️ Groq API failed:", e)
            print("💻 Using local model fallback...")

    # Local model fallback
    try:
        generator = pipeline("text2text-generation", model=GENERATOR_MODEL, device_map="auto")
        result = generator(prompt, max_new_tokens=100, do_sample=False)[0]["generated_text"].strip()
        # Keep it short — 1–2 sentences
        sentences = result.split(". ")
        return ". ".join(sentences[:2]) + "."
    except Exception as e:
        print("❌ Local model generation failed:", e)
        return "⚠️ Sorry, I couldn’t create a reply right now."

# =========================================================
# 🩺 Full RAG Query
# =========================================================
def rag_query(query: str, top_k: int = 3, chat_context: str = ""):
    """Retrieve → Generate → Return short conversational answer."""
    chunks, meta = retrieve_context(query, top_k)
    if not chunks:
        return {"answer": "⚠️ No relevant information found.", "metadata": []}
    answer = generate_answer(query, "\n".join(chunks), chat_context)
    return {"answer": answer, "metadata": meta}

# =========================================================
# 🧪 Test Run
# =========================================================
if __name__ == "__main__":
    q = "What is oral cancer?"
    result = rag_query(q)
    print("\n🩺 Final Answer:\n", result["answer"])
